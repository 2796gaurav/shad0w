"""Signed certificates: an HMAC-SHA256 over the canonical JSON of a bundle's certificate.json.

    SHAD0W_CERT_KEY=... shad0w certify --bundle B --data fresh.jsonl --sign      # writes B/certificate.sig
    SHAD0W_CERT_KEY=... shad0w sign   --bundle B                                   # sign the certificate as it is
    SHAD0W_CERT_KEY=... shad0w verify --bundle B [--teacher gpt-6-luna]            # exit 0 when it checks out

The key is read from the environment variable named by KEY_ENV (SHAD0W_CERT_KEY) and never from a default: with no
key, signing and verifying refuse. The signature covers certificate.json byte-for-byte after canonicalisation
(sorted keys, no whitespace, UTF-8), so re-indenting the file does not break it while changing any value does.
`verify` also checks that the bundle's manifest (what the tables were compiled from) names the same teacher as the
certificate and, when asked, the teacher you expect: a certificate against gpt-6-luna says nothing about a bundle
compiled from another model's answers.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import hmac
import json
import os

KEY_ENV = "SHAD0W_CERT_KEY"
SIG_FILE = "certificate.sig"
CERT_FILE = "certificate.json"
ALGORITHM = "hmac-sha256"

__all__ = ["KEY_ENV", "SIG_FILE", "canonical_json", "sign_bundle", "verify_bundle", "signing_key"]


class SignatureError(ValueError):
    """The signature is missing, malformed, or does not match the certificate (or the teacher differs)."""


def canonical_json(obj) -> bytes:
    """The one byte string every equal JSON document maps to: sorted keys, no whitespace, no ASCII escaping."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def signing_key(key: str | bytes | None = None) -> bytes:
    """The HMAC key: the argument, else $SHAD0W_CERT_KEY. Never a built-in default."""
    if key is None:
        key = os.environ.get(KEY_ENV, "")
    if isinstance(key, str):
        key = key.encode("utf-8")
    if not key:
        raise ValueError(f"no signing key: set {KEY_ENV} (any long random string; keep it out of the bundle)")
    if len(key) < 16:
        raise ValueError(f"{KEY_ENV} is too short: use at least 16 bytes (32+ recommended)")
    return key


def _mac(key: bytes, cert: dict) -> str:
    return hmac.new(key, canonical_json(cert), hashlib.sha256).hexdigest()


def _read_cert(bundle: str) -> dict:
    p = os.path.join(bundle, CERT_FILE)
    if not os.path.exists(p):
        raise FileNotFoundError(f"no {CERT_FILE} in {bundle} (run shad0w train / shadow / certify first)")
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def manifest_teacher(bundle: str) -> str | None:
    """The teacher named in the bundle's manifest (what the tables were compiled from), or None."""
    p = os.path.join(bundle, "manifest.json")
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        man = json.load(f)
    t = (man.get("meta") or {}).get("teacher")
    return None if t is None else str(t)


def sign_bundle(bundle: str, key: str | bytes | None = None) -> dict:
    """Write <bundle>/certificate.sig for the current certificate.json and return the signature record."""
    k = signing_key(key)
    cert = _read_cert(bundle)
    sig = {"format": "shad0w-signature/1", "algorithm": ALGORITHM, "signature": _mac(k, cert),
           "certificate_sha256": hashlib.sha256(canonical_json(cert)).hexdigest(),
           "key_id": hashlib.sha256(k).hexdigest()[:16],  # which key signed (not reversible)
           "teacher": cert.get("teacher"), "signed_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")}
    with open(os.path.join(bundle, SIG_FILE), "w", encoding="utf-8") as f:
        json.dump(sig, f, indent=2)
    return sig


def verify_bundle(bundle: str, key: str | bytes | None = None, teacher: str | None = None) -> dict:
    """Check the signature and the teacher. Returns {"ok": True, ...details} or raises SignatureError (bad or
    missing signature, teacher mismatch) / ValueError (no key)."""
    k = signing_key(key)
    cert = _read_cert(bundle)
    p = os.path.join(bundle, SIG_FILE)
    if not os.path.exists(p):
        raise SignatureError(f"no {SIG_FILE} in {bundle}: the certificate is unsigned")
    with open(p, encoding="utf-8") as f:
        sig = json.load(f)
    if not isinstance(sig, dict) or sig.get("algorithm") != ALGORITHM or not isinstance(sig.get("signature"), str):
        raise SignatureError(f"{SIG_FILE} is not a shad0w {ALGORITHM} signature")
    if not hmac.compare_digest(sig["signature"], _mac(k, cert)):
        raise SignatureError("signature does not match certificate.json (changed since signing, or a different key)")
    cert_teacher = cert.get("teacher")
    man_teacher = manifest_teacher(bundle)
    if man_teacher is not None and cert_teacher is not None and man_teacher != cert_teacher:
        raise SignatureError(f"the bundle was compiled from {man_teacher!r} but the certificate names {cert_teacher!r}")
    if teacher is not None:
        for what, got in (("certificate", cert_teacher), ("manifest", man_teacher)):
            if got is not None and got != teacher:
                raise SignatureError(f"{what} names teacher {got!r}, expected {teacher!r}")
        if cert_teacher is None and man_teacher is None:
            raise SignatureError(f"neither the certificate nor the manifest names a teacher; cannot confirm {teacher!r}")
    return {"ok": True, "teacher": cert_teacher or man_teacher, "key_id": sig.get("key_id"), "signed_utc": sig.get("signed_utc"),
            "certificate_sha256": hashlib.sha256(canonical_json(cert)).hexdigest()}
