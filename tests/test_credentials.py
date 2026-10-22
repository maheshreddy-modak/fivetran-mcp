"""Coverage for select_credentials_resolver's mode dispatch.

Nothing previously exercised env_resolver/header_resolver/oauth_resolver
directly; these tests pin the dispatch contract: stdio always gets
env_resolver; streamable-http gets header_resolver when FIVETRAN_AUTH_ISSUER
is unset (the interim, non-OAuth path) or oauth_resolver when it's set, and
refuses to start if a shared API key is set regardless; an unrecognized mode
is a hard error.
"""
import base64

import pytest

import server
from server import CredentialsError, select_credentials_resolver


@pytest.mark.asyncio
async def test_stdio_mode_dispatches_env_resolver(monkeypatch):
    monkeypatch.setenv("FIVETRAN_API_KEY", "key123")
    monkeypatch.setenv("FIVETRAN_API_SECRET", "secret456")
    resolver = select_credentials_resolver("stdio")
    creds = await resolver()
    assert creds.authorization == f"Basic {base64.b64encode(b'key123:secret456').decode()}"


@pytest.mark.asyncio
async def test_stdio_mode_ignores_shared_key_guard(monkeypatch):
    monkeypatch.setenv("FIVETRAN_API_KEY", "key123")
    monkeypatch.setenv("FIVETRAN_API_SECRET", "secret456")
    select_credentials_resolver("stdio")  # must not raise


@pytest.mark.asyncio
async def test_streamable_http_mode_dispatches_header_resolver_when_issuer_unset(monkeypatch):
    monkeypatch.delenv("FIVETRAN_API_KEY", raising=False)
    monkeypatch.delenv("FIVETRAN_API_SECRET", raising=False)
    monkeypatch.delenv("FIVETRAN_AUTH_ISSUER", raising=False)
    resolver = select_credentials_resolver("streamable-http")
    with pytest.raises(CredentialsError, match="No request context available"):
        await resolver()


@pytest.mark.asyncio
async def test_streamable_http_mode_dispatches_oauth_resolver_when_issuer_set(monkeypatch):
    monkeypatch.delenv("FIVETRAN_API_KEY", raising=False)
    monkeypatch.delenv("FIVETRAN_API_SECRET", raising=False)
    monkeypatch.setenv("FIVETRAN_AUTH_ISSUER", "https://auth.fivetran.com")
    resolver = select_credentials_resolver("streamable-http")
    # oauth_resolver and header_resolver share the same request-context guard,
    # so this error is what distinguishes "some resolver ran" from a crash;
    # test_oauth.py covers oauth_resolver's actual header-forwarding body.
    with pytest.raises(CredentialsError, match="No request context available"):
        await resolver()


def test_streamable_http_mode_fails_startup_if_api_key_set(monkeypatch):
    monkeypatch.setenv("FIVETRAN_API_KEY", "key123")
    monkeypatch.delenv("FIVETRAN_API_SECRET", raising=False)
    with pytest.raises(ValueError, match="FIVETRAN_API_KEY"):
        select_credentials_resolver("streamable-http")


def test_streamable_http_mode_fails_startup_if_api_secret_set(monkeypatch):
    monkeypatch.delenv("FIVETRAN_API_KEY", raising=False)
    monkeypatch.setenv("FIVETRAN_API_SECRET", "secret456")
    with pytest.raises(ValueError, match="FIVETRAN_API_KEY"):
        select_credentials_resolver("streamable-http")


def test_streamable_http_mode_fails_startup_if_both_set(monkeypatch):
    monkeypatch.setenv("FIVETRAN_API_KEY", "key123")
    monkeypatch.setenv("FIVETRAN_API_SECRET", "secret456")
    with pytest.raises(ValueError, match="FIVETRAN_API_KEY"):
        select_credentials_resolver("streamable-http")


def test_unknown_mode_raises(monkeypatch):
    monkeypatch.delenv("FIVETRAN_API_KEY", raising=False)
    monkeypatch.delenv("FIVETRAN_API_SECRET", raising=False)
    with pytest.raises(ValueError, match="Unknown transport mode"):
        select_credentials_resolver("http")


def test_bearer_as_basic_relabels_bearer_header():
    assert server._bearer_as_basic("Bearer abc123==") == "Basic abc123=="
    assert server._bearer_as_basic("bearer abc123==") == "Basic abc123=="


def test_bearer_as_basic_leaves_other_schemes_untouched():
    assert server._bearer_as_basic("Basic abc123==") == "Basic abc123=="
    assert server._bearer_as_basic("Bearer ") == "Bearer "
    assert server._bearer_as_basic("") == ""


@pytest.mark.asyncio
@pytest.mark.parametrize("flag, expected", [("true", "Basic abc123=="), (None, "Bearer abc123==")])
async def test_header_resolver_bearer_as_basic_is_opt_in(monkeypatch, flag, expected):
    async def fake_forward():
        return server.Credentials(authorization="Bearer abc123==")

    monkeypatch.setattr(server, "_forward_authorization_header", fake_forward)
    if flag is None:
        monkeypatch.delenv("FIVETRAN_BEARER_AS_BASIC", raising=False)
    else:
        monkeypatch.setenv("FIVETRAN_BEARER_AS_BASIC", flag)
    creds = await server.header_resolver()()
    assert creds.authorization == expected
