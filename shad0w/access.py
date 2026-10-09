"""Who may use `shad0w proxy` and `shad0w serve`: an optional access token, and no LLM key open to the network.

With a token set (`access_token=`, `--token-env NAME`, `--token-file PATH`, or $SHAD0W_PROXY_TOKEN) every route except
GET /v1/health needs it, sent one of three ways:
  - X-Shad0w-Token: <token>              any client; never forwarded upstream
  - Authorization: Bearer <token>        an OpenAI SDK pointed at a proxy that holds its own upstream key:
                                         OpenAI(base_url=..., api_key=<token>)
  - Basic auth, password <token>         a browser: the dashboard asks for it once (any user name)
The header that carried the token is never sent upstream.

Listening beyond this machine (--host 0.0.0.0, a LAN address) with the proxy's own LLM key and no token would let
anyone who reaches the port spend that key and read the dashboard, so it is refused unless allow_open=True
(`--insecure-open`). Any other non-local address without a token prints a warning.
"""
from __future__ import annotations

import base64
import hmac
import ipaddress
import os
import sys

from .config import Secret

TOKEN_ENV = "SHAD0W_PROXY_TOKEN"
OPEN_PATHS = ("/v1/health",)  # liveness probes need no token


def token_secret(token=None, env: str | None = None, file: str | None = None) -> Secret | None:
    """The access token as a Secret, from (first match) `token`, the variable `env`, the file `file` (re-read on every
    request so a mounted secret can rotate) or $SHAD0W_PROXY_TOKEN. None means no token."""
    if token is not None:
        return token if isinstance(token, Secret) else Secret(token, source="access_token=")
    if env:
        s = Secret(env=env)
        if not s:
            raise ValueError(f"{env} is not set (it should hold the proxy's access token)")
        return s
    if file:
        def read():
            with open(file, encoding="utf-8") as f:
                return f.read().strip()
        if not read():
            raise ValueError(f"--token-file {file} is empty")
        return Secret(read, source=f"file {file}")
    if os.environ.get(TOKEN_ENV):
        return Secret(env=TOKEN_ENV)
    return None


def is_local(host: str) -> bool:
    """True when `host` only accepts connections from this machine."""
    if host == "localhost":
        return True
    try:  # "" and "0.0.0.0" mean every interface
        return ipaddress.ip_address(host.strip("[]")).is_loopback
    except ValueError:  # a host name: only "localhost" is known to be local
        return False


def check_exposure(host: str, token: Secret | None, own_key: bool, allow_open: bool, what: str = "proxy") -> None:
    """Refuse an LLM key open to the network; warn about any other open, non-local address."""
    if token is not None or is_local(host):
        return
    if own_key and not allow_open:
        raise ValueError(
            f"refusing to listen on {host} with the {what}'s own LLM key and no access token: anyone who can reach the "
            f"port could spend that key and read the dashboard. Set a token ({TOKEN_ENV}, --token-env NAME or "
            "--token-file PATH), listen on 127.0.0.1, or pass --insecure-open if the network itself is private.")
    print(f"shad0w {what}: WARNING listening on {host} with no access token; anyone who can reach the port can use it "
          f"and read the dashboard. Set {TOKEN_ENV} (or --token-env / --token-file) to require one.",
          file=sys.stderr, flush=True)


def authorize(headers, path: str, token: Secret | None) -> tuple[bool, str | None]:
    """(allowed, header that carried the token). That header must not be forwarded upstream."""
    if token is None or path in OPEN_PATHS:
        return True, None
    want = token.get() or ""
    if not want:  # a token file emptied after start-up: fail closed
        return False, None
    got = headers.get("x-shad0w-token")
    if got is not None and _same(got, want):
        return True, "x-shad0w-token"
    auth = headers.get("authorization") or ""
    kind, _, cred = auth.partition(" ")
    if kind.lower() == "bearer" and _same(cred.strip(), want):
        return True, "authorization"
    if kind.lower() == "basic":
        try:
            _, _, password = base64.b64decode(cred.strip()).decode("utf-8").partition(":")
        except (ValueError, UnicodeDecodeError):
            password = None
        if password is not None and _same(password, want):
            return True, "authorization"
    return False, None


def _same(a: str, b: str) -> bool:
    return hmac.compare_digest(a.encode("utf-8"), b.encode("utf-8"))


DENIED = {"error": {"message": "shad0w: missing or wrong access token. Send 'X-Shad0w-Token: <token>', "
                               "'Authorization: Bearer <token>', or log in with the token as the password.",
                    "type": "invalid_request_error", "code": "invalid_access_token"}}
CHALLENGE = {"www-authenticate": 'Basic realm="shad0w", charset="UTF-8"'}
