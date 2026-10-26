"""Coverage for formatting.py: compact rendering of tool results.

Fixtures mirror the shapes of real Fivetran responses (list envelopes,
schema-config maps, setup_tests) with made-up values.
"""
import json

import httpx
import pytest

import formatting
import server
from formatting import (
    MAX_ROWS,
    clamp_limit,
    render_api_result,
    render_endpoint_list,
    render_schema,
)
from server import SCOPE_TIERS, Credentials, call_tool, configure, set_credentials_resolver


def _envelope(data, message="Retrieved successfully"):
    return {"code": "Success", "message": message, "data": data}


def _connection(i, **extra):
    return {
        "id": f"conn_{i}", "group_id": "grp", "service": "sap_odata", "schema": "a4h",
        "paused": False, "config": None, "setup_tests": None,
        "status": {"setup_state": "connected", "sync_state": "scheduled",
                   "update_state": "on_schedule", "tasks": [], "warnings": []},
        **extra,
    }


# --- list results -------------------------------------------------------------

def test_curated_list_renders_selected_columns_only():
    text = render_api_result("list_connections", _envelope({"items": [_connection(1)]}))
    lines = text.splitlines()
    assert lines[0] == "Retrieved successfully"
    assert lines[1].startswith("id|service|schema|group_id|paused|status.setup_state")
    assert lines[2].startswith("conn_1|sap_odata|a4h|grp|false|connected")
    assert "tasks" not in text and "config" not in text
    assert lines[-1] == "1 row"


def test_list_footer_carries_next_cursor():
    text = render_api_result(
        "list_all_groups", _envelope({"items": [{"id": "g1", "name": "W"}], "next_cursor": "abc"})
    )
    assert text.splitlines()[-1] == '1 row · more available: call again with query={"cursor": "abc"}'


def test_unpaged_list_is_capped_with_hint():
    items = [{"id": f"svc_{i}", "name": f"S{i}", "type": "T", "icons": ["x.png"]} for i in range(250)]
    text = render_api_result("metadata_public_connectors", _envelope(items))
    assert "svc_99|" in text and "svc_100|" not in text
    assert f"{MAX_ROWS} of 250 rows shown" in text
    assert "metadata_connectors" in text.splitlines()[-1]
    assert "x.png" not in text


def test_generic_list_drops_empty_columns_and_noise_keys():
    items = [{"id": "a", "note": None, "link_to_docs": "http://d", "tags": ["x", "y"]}]
    text = render_api_result("some_other_list", _envelope({"items": items}))
    assert text.splitlines()[1] == "id|tags"
    assert text.splitlines()[2] == "a|x,y"


def test_keyed_map_collapses_to_key_names():
    items = [
        {"id": "p1", "configurable_variables": {"var_a": {"type": "bool"}, "var_b": {"type": "str"}}},
        {"id": "p2", "configurable_variables": {"var_c": {"type": "int"}}},
    ]
    text = render_api_result("transformation_package_metadata_list", _envelope({"items": items}))
    assert text.splitlines()[1] == "id|configurable_variables"
    assert "p1|var_a,var_b" in text


def test_pipe_in_value_is_quoted():
    text = render_api_result("x", _envelope({"items": [{"id": "a|b"}]}))
    assert '"a|b"' in text


def test_empty_list():
    assert render_api_result("list_all_groups", _envelope({"items": []})).endswith("0 rows")


def test_long_cells_are_truncated_in_lists():
    text = render_api_result("x", _envelope({"items": [{"id": "a", "d": "z" * 500}]}))
    assert "z" * 500 not in text
    assert "..." in text


# --- single objects -------------------------------------------------------------

def test_object_is_compact_json_without_empties_and_setup_tests_table():
    data = {
        "id": "d1", "private_link_id": None, "config": {"password": "******", "port": "443"},
        "paused": False,
        "setup_tests": [{"title": "Validate Permissions", "status": "PASSED", "message": ""}],
    }
    text = render_api_result("create_destination", _envelope(data, "Destination has been created"))
    lines = text.splitlines()
    assert lines[0] == "Destination has been created"
    assert json.loads(lines[1]) == {"id": "d1", "config": {"password": "******", "port": "443"}, "paused": False}
    assert lines[2:] == ["setup_tests:", "title|status", "Validate Permissions|PASSED"]


def test_schema_config_renders_one_row_per_table():
    data = {
        "enable_new_by_default": True,
        "schemas": {"SRV": {"enabled": True, "tables": {
            "T1": {"enabled": False, "name_in_destination": "t1",
                   "columns": {"A": {"enabled": True}, "B": {"enabled": True}},
                   "enabled_patch_settings": {"allowed": False}},
        }}},
    }
    text = render_api_result("connection_schema_config", _envelope(data))
    assert 'settings: {"enable_new_by_default":true}' in text
    assert "SRV|T1|false||t1|2|false" in text
    assert "connection_column_config" in text


def test_column_config_renders_columns():
    data = {"columns": {"ID": {"enabled": True, "hashed": False, "is_primary_key": True}}}
    text = render_api_result("connection_column_config", _envelope(data))
    assert "ID|true|false|true|||" in text


def test_errors_and_envelope_less_results_stay_json():
    err = {"error": "UPSTREAM_ERROR", "status": 400, "message": "bad"}
    assert json.loads(render_api_result("x", err)) == err
    assert json.loads(render_api_result("x", {"status": "success", "code": 204})) == {
        "status": "success", "code": 204}


# --- limit clamp ----------------------------------------------------------------

def test_clamp_limit():
    assert clamp_limit(None) == (None, None)
    assert clamp_limit({"limit": 50}) == ({"limit": 50}, None)
    q, note = clamp_limit({"limit": 1000, "cursor": "c"})
    assert q == {"limit": MAX_ROWS, "cursor": "c"}
    assert note == f"limit capped at {MAX_ROWS} (requested 1000)"
    assert clamp_limit({"limit": "abc"}) == ({"limit": "abc"}, None)


class _RecordingClient:
    def __init__(self, response):
        self.response, self.params = response, None

    async def request(self, *, method, url, headers, params, json, timeout=None):
        self.params = params
        return self.response


@pytest.mark.asyncio
async def test_call_tool_clamps_limit_before_upstream(monkeypatch):
    configure(SCOPE_TIERS["read"], set())

    async def _resolver():
        return Credentials(authorization="Basic test")

    set_credentials_resolver(_resolver)
    req = httpx.Request("GET", "https://api.fivetran.com/v1/metadata/connector-types")
    client = _RecordingClient(httpx.Response(200, request=req, json=_envelope({"items": []})))
    monkeypatch.setattr(server, "_http_client", client)

    tool = next(n for n, pair in server.TOOLS_BY_NAME.items() if pair == ("metadata", "read"))
    result = await call_tool(tool, {"name": "metadata_connectors", "query": {"limit": 1000}})

    assert client.params == {"limit": MAX_ROWS}
    assert "note: limit capped" in result[0].text


# --- discovery ----------------------------------------------------------------

def test_endpoint_list_and_summary():
    text = render_endpoint_list({"endpoints": [
        {"name": "create_group", "method": "POST", "path": "/v1/groups",
         "summary": "WRITE OPERATION - Create a Group", "scope": "write", "callable": True}]})
    assert text.splitlines() == [
        "name|method|path|scope|callable|summary",
        "create_group|POST|/v1/groups|write|true|WRITE OPERATION - Create a Group",
        "1 endpoints",
    ]
    summary = render_endpoint_list({"categories": {"groups": 3}, "callable_counts": {"groups": 2}, "total": 3})
    assert summary.splitlines() == ["category|endpoints|callable", "groups|3|2", "total endpoints: 3"]


_SCHEMA = {
    "description": "⚠️ WRITE OPERATION - Creates. See [docs](https://x/y).",
    "path": "/v1/destinations",
    "method": "POST",
    "request_body": {"content": {"application/json": {
        "required": ["service"],
        "properties": {
            "service": {"type": "string", "enum": [f"svc_{i}" for i in range(60)]},
            "config": {"type": "object", "required": ["host"], "properties": {
                "host": {"type": "string", "description": "Host"},
                "password": {"type": "string", "format": "password"},
            }},
            "tags": {"type": "array", "items": {"type": "string"}, "default": []},
        },
    }}},
    "response": {"content": {"application/json": {"properties": {"code": {"type": "string"}}}}},
}


def test_schema_renders_field_table_and_omits_response():
    text = render_schema(_SCHEMA)
    lines = text.splitlines()
    assert lines[0] == "POST /v1/destinations"
    assert lines[1] == "⚠️ WRITE OPERATION - Creates. See docs."
    assert "field|type|required|allowed|description" in lines
    assert "config.host|string|yes||Host" in lines
    assert "config.password|string/password|||" in lines
    assert "tags|array<string>||default=[]|" in lines
    assert "response schema omitted" in lines[-1]
    # Enum lists are never truncated: every allowed value survives.
    assert all(f"svc_{i}" in text for i in range(60))


def test_schema_service_narrows_service_enum():
    text = render_schema(_SCHEMA, service="svc_7")
    assert "service|string|yes|svc_7|" in text
    assert "svc_8" not in text


def test_schema_include_response():
    text = render_schema(_SCHEMA, include_response=True)
    assert "response:" in text and "code|string|||" in text


def test_schema_error_stays_json():
    err = {"error": "UNKNOWN_ENDPOINT", "endpoint": "x", "message": "m"}
    assert json.loads(render_schema(err)) == err


def test_real_schema_files_render():
    # Every endpoint in the manifest renders without raising.
    for ep in server.ENDPOINTS:
        render_schema(server.do_get_schema(ep["name"]))


# --- unknown service suggestions ---------------------------------------------

@pytest.mark.parametrize("kind,service,expected", [
    ("connectors", "databricks", "databricks_db"),
    ("connectors", "sap_odta", "sap_odata"),
    ("destinations", "snowflak", "snowflake"),
])
def test_unknown_service_suggests_close_ids(kind, service, expected):
    with pytest.raises(ValueError, match=f"Did you mean: .*{expected}"):
        server._load_service_config(kind, service)


def test_unknown_service_without_matches_has_no_hint():
    with pytest.raises(ValueError) as exc:
        server._load_service_config("destinations", "zzzz")
    assert "Did you mean" not in str(exc.value)
