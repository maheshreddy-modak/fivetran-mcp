"""Compact text rendering of tool results for LLM context windows.

The do_* handlers in server.py return plain dicts; call_tool hands them to
render_* here right before they become TextContent. Nothing in this module
changes what is fetched — only how it is written out:

- Lists of records become pipe-delimited rows (header + values, no padding),
  capped at MAX_ROWS, with a footer saying how to fetch more.
- Single records become JSON without indentation or null/empty fields.
- get_schema becomes a flat request-field table instead of raw OpenAPI.
- Error dicts stay JSON so agents can still branch on `error`.
"""
import csv
import io
import json
import re
from typing import Any

MAX_ROWS = 100
MAX_CELL_CHARS = 120
MAX_FIELD_DESC_CHARS = 200
_MAX_SCHEMA_DEPTH = 6

# URLs and image links an agent never needs to reason over in a listing.
_LIST_NOISE_KEYS = frozenset({"icon_url", "icons", "link_to_docs", "link_to_erd"})

_CONNECTION_COLUMNS = [
    "id", "service", "schema", "group_id", "paused", "status.setup_state",
    "status.sync_state", "status.update_state", "succeeded_at", "failed_at",
    "sync_frequency", "schedule_type",
]
_CONNECTOR_TYPE_COLUMNS = ["id", "name", "type", "service_status"]

# Hand-picked columns for the listings agents hit most; everything else uses
# every non-empty column.
CURATED_COLUMNS: dict[str, list[str]] = {
    "metadata_connectors": _CONNECTOR_TYPE_COLUMNS,
    "metadata_public_connectors": _CONNECTOR_TYPE_COLUMNS,
    "list_all_groups": ["id", "name", "created_at"],
    "list_connections": _CONNECTION_COLUMNS,
    "list_all_connections_in_group": _CONNECTION_COLUMNS,
    "list_destinations": [
        "id", "group_id", "service", "region", "setup_status", "time_zone_offset",
    ],
}

# Where to send the agent when a listing without a cursor is capped.
_UNPAGED_HINTS: dict[str, str] = {
    "metadata_public_connectors": (
        "This endpoint has no paging. Page the same catalog with "
        "metadata_read(name='metadata_connectors', query={'cursor': ...}), or check "
        "one service id directly with get_schema(name='create_connection', service=<id>)."
    ),
}


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

def compact_json(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False, default=str)


def drop_empty(value: Any) -> Any:
    """Recursively remove None, "", [] and {} (False and 0 are kept)."""
    if isinstance(value, dict):
        out = {k: drop_empty(v) for k, v in value.items()}
        return {k: v for k, v in out.items() if v not in (None, "", [], {})}
    if isinstance(value, list):
        return [drop_empty(v) for v in value]
    return value


def _cell(value: Any, max_chars: int | None = MAX_CELL_CHARS) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, list):
        value = ",".join(
            str(v.get("id", compact_json(v))) if isinstance(v, dict) else str(v)
            for v in value
        )
    elif isinstance(value, dict):
        value = compact_json(value)
    s = str(value).replace("\n", " ")
    if max_chars is None or len(s) <= max_chars:
        return s
    return s[:max_chars - 3] + "..."


def _is_keyed_map(value: dict[str, Any]) -> bool:
    """A dict of objects keyed by name (e.g. configurable_variables), not a sub-record."""
    return all(isinstance(v, dict) for v in value.values())


def _flatten(record: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Sub-records become dotted columns; keyed maps collapse to their key names,
    so per-row dynamic keys can't explode into hundreds of sparse columns."""
    out: dict[str, Any] = {}
    for k, v in record.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict) and v:
            if _is_keyed_map(v):
                out[key] = list(v)
            else:
                out.update(_flatten(v, key + "."))
        else:
            out[key] = v
    return out


def serialize_table(
    rows: list[list[Any]], headers: list[str], max_cell_chars: int | None = MAX_CELL_CHARS
) -> str:
    """Pipe-delimited rows (header + values, no padding)."""
    buf = io.StringIO()
    writer = csv.writer(buf, delimiter="|", lineterminator="\n")
    writer.writerow(headers)
    writer.writerows([_cell(v, max_cell_chars) for v in row] for row in rows)
    return buf.getvalue().rstrip("\n")


def records_table(records: list[dict[str, Any]], columns: list[str] | None = None) -> str:
    """Render dict records as a table, keeping only columns with any value."""
    flat = [
        _flatten({k: v for k, v in r.items() if k not in _LIST_NOISE_KEYS})
        for r in records
    ]
    if columns is None:
        columns = list(dict.fromkeys(k for r in flat for k in r))
    columns = [c for c in columns if any(r.get(c) not in (None, "", [], {}) for r in flat)]
    return serialize_table([[r.get(c) for c in columns] for r in flat], columns)


def _footer(shown: int, total: int, next_cursor: str | None, hint: str | None) -> str:
    noun = "row" if shown == 1 else "rows"
    parts = [f"{shown} {noun}" if shown == total else f"{shown} of {total} rows shown"]
    if next_cursor:
        parts.append(f"more available: call again with query={{\"cursor\": \"{next_cursor}\"}}")
    elif shown < total and hint:
        parts.append(hint)
    return " · ".join(parts)


# ---------------------------------------------------------------------------
# API results
# ---------------------------------------------------------------------------

def clamp_limit(query: dict[str, Any] | None) -> tuple[dict[str, Any] | None, str | None]:
    """Cap a caller-supplied page size at MAX_ROWS; return (query, note)."""
    if not query or "limit" not in query:
        return query, None
    try:
        requested = int(query["limit"])
    except (TypeError, ValueError):
        return query, None
    if requested <= MAX_ROWS:
        return query, None
    return {**query, "limit": MAX_ROWS}, f"limit capped at {MAX_ROWS} (requested {requested})"


def _render_schema_config(data: dict[str, Any]) -> str:
    rows = []
    for schema, s in (data.get("schemas") or {}).items():
        tables = (s or {}).get("tables") or {}
        if not tables:
            rows.append([schema, "", s.get("enabled"), "", s.get("name_in_destination"), "", ""])
        for table, t in tables.items():
            t = t or {}
            rows.append([
                schema, table, t.get("enabled"), t.get("sync_mode"),
                t.get("name_in_destination"), len(t["columns"]) if t.get("columns") else "",
                (t.get("enabled_patch_settings") or {}).get("allowed"),
            ])
    settings = {k: v for k, v in data.items() if k != "schemas"}
    lines = [f"settings: {compact_json(drop_empty(settings))}"] if settings else []
    headers = ["schema", "table", "enabled", "sync_mode", "name_in_destination",
               "columns", "patch_allowed"]
    lines.append(serialize_table(rows[:MAX_ROWS], headers))
    lines.append(_footer(min(len(rows), MAX_ROWS), len(rows), None, None))
    lines.append(
        "per-table columns: connections_read(name='connection_column_config', "
        "path_params={'connectionId': ..., 'schema': ..., 'table': ...})"
    )
    return "\n".join(lines)


def _render_column_config(data: dict[str, Any]) -> str:
    rows = []
    for name, c in (data.get("columns") or {}).items():
        c = c or {}
        rows.append([
            name, c.get("enabled"), c.get("hashed"), c.get("is_primary_key"),
            c.get("target_data_type"), c.get("name_in_destination"),
            (c.get("enabled_patch_settings") or {}).get("allowed"),
        ])
    headers = ["column", "enabled", "hashed", "is_primary_key", "target_data_type",
               "name_in_destination", "patch_allowed"]
    shown = rows[:MAX_ROWS]
    return serialize_table(shown, headers) + "\n" + _footer(len(shown), len(rows), None, None)


_OBJECT_RENDERERS = {
    "connection_schema_config": _render_schema_config,
    "reload_connection_schema_config": _render_schema_config,
    "modify_connection_schema_config": _render_schema_config,
    "pre_create_connection_schema_config": _render_schema_config,
    "connection_column_config": _render_column_config,
}


def _render_object(name: str, data: dict[str, Any]) -> str:
    renderer = _OBJECT_RENDERERS.get(name)
    if renderer and ("schemas" in data or "columns" in data):
        return renderer(data)
    data = dict(data)
    tests = data.pop("setup_tests", None)
    lines = [compact_json(drop_empty(data))]
    if tests:
        lines.append("setup_tests:")
        lines.append(records_table(tests, ["title", "status", "message"]))
    return "\n".join(lines)


def render_api_result(name: str, result: Any, note: str | None = None) -> str:
    """Render a do_call result for the endpoint `name`."""
    if not isinstance(result, dict) or "error" in result or "data" not in result:
        return compact_json(result)

    data = result["data"]
    lines = []
    message = result.get("message")
    if message:
        lines.append(message)
    if note:
        lines.append(f"note: {note}")

    items, next_cursor = None, None
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict) and isinstance(data.get("items"), list):
        items, next_cursor = data["items"], data.get("next_cursor")

    if items is None:
        lines.append(_render_object(name, data) if isinstance(data, dict) else compact_json(data))
    elif not items:
        lines.append("0 rows")
    elif not all(isinstance(i, dict) for i in items):
        lines.append(compact_json(items[:MAX_ROWS]))
        lines.append(_footer(min(len(items), MAX_ROWS), len(items), next_cursor, None))
    else:
        shown = items[:MAX_ROWS]
        lines.append(records_table(shown, CURATED_COLUMNS.get(name)))
        lines.append(_footer(len(shown), len(items), next_cursor, _UNPAGED_HINTS.get(name)))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Discovery results
# ---------------------------------------------------------------------------

def render_endpoint_list(result: dict[str, Any]) -> str:
    if "endpoints" in result:
        eps = result["endpoints"]
        if not eps:
            return "0 endpoints"
        table = records_table(eps, ["name", "method", "path", "scope", "callable", "summary"])
        return f"{table}\n{len(eps)} endpoints"
    if "categories" in result:
        callable_counts = result.get("callable_counts", {})
        rows = [[c, n, callable_counts.get(c, 0)] for c, n in result["categories"].items()]
        table = serialize_table(rows, ["category", "endpoints", "callable"])
        return f"{table}\ntotal endpoints: {result.get('total')}"
    return compact_json(result)


def _plain(text: str | None) -> str:
    """Markdown links -> link text; collapse whitespace."""
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text or "")
    return re.sub(r"\s+", " ", text).strip()


def _field_rows(schema: dict[str, Any], prefix: str = "", depth: int = 0) -> list[list[Any]]:
    """Flatten an OpenAPI object schema into path|type|required|allowed|description rows."""
    if depth > _MAX_SCHEMA_DEPTH:
        return []
    rows: list[list[Any]] = []
    for part in schema.get("allOf", []):
        rows += _field_rows(part, prefix, depth)
    required = set(schema.get("required", []))
    for name, prop in (schema.get("properties") or {}).items():
        if not isinstance(prop, dict):
            continue
        path = prefix + name
        items = prop.get("items") if isinstance(prop.get("items"), dict) else None
        ptype = prop.get("type") or ("object" if "properties" in prop else "")
        if ptype == "array" and items:
            ptype = f"array<{items.get('type') or 'object'}>"
        if prop.get("format"):
            ptype += f"/{prop['format']}"
        enum = prop.get("enum") or (items or {}).get("enum")
        allowed = ",".join(map(str, enum)) if enum else ""
        if "default" in prop:
            allowed = f"{allowed} default={prop['default']}".strip()
        desc = _plain(prop.get("description"))
        if len(desc) > MAX_FIELD_DESC_CHARS:
            desc = desc[:MAX_FIELD_DESC_CHARS - 3] + "..."
        if prop.get("deprecated"):
            desc = f"[deprecated] {desc}"
        rows.append([path, ptype, "yes" if name in required else "", allowed, desc])
        if "properties" in prop or "allOf" in prop:
            rows += _field_rows(prop, path + ".", depth + 1)
        if items and ("properties" in items or "allOf" in items):
            rows += _field_rows(items, path + "[].", depth + 1)
    return rows


_FIELD_HEADERS = ["field", "type", "required", "allowed", "description"]


def _json_body(section: Any) -> dict[str, Any] | None:
    body = (section or {}).get("content", {}).get("application/json")
    return body if isinstance(body, dict) else None


def _schema_table(body: dict[str, Any], service: str | None = None) -> str:
    rows = _field_rows(body)
    if service:
        # The caller already chose a service; listing every other id is noise.
        for row in rows:
            if row[0] == "service" and row[3]:
                row[3] = service
    # Uncapped cells: a truncated `allowed` list would silently drop valid values.
    return serialize_table(rows, _FIELD_HEADERS, max_cell_chars=None)


def render_schema(
    schema: dict[str, Any], include_response: bool = False, service: str | None = None
) -> str:
    if "error" in schema:
        return compact_json(schema)
    lines = [f"{schema.get('method', '')} {schema.get('path', '')}".strip()]
    if schema.get("description"):
        lines.append(_plain(schema["description"]))

    params = schema.get("parameters") or []
    if params:
        lines.append("parameters:")
        lines.append(serialize_table(
            [[p.get("name"), p.get("in"), "yes" if p.get("required") else "",
              _plain(p.get("description"))[:MAX_FIELD_DESC_CHARS]] for p in params],
            ["name", "in", "required", "description"],
        ))

    body = _json_body(schema.get("request_body"))
    if body:
        lines.append("request body:")
        lines.append(_schema_table(body, service))

    response = _json_body(schema.get("response"))
    if response:
        if include_response:
            lines.append("response:")
            lines.append(_schema_table(response))
        else:
            lines.append("response schema omitted; pass include_response=true to see it.")
    return "\n".join(lines)
