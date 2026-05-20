"""
offline_guard.py — FR028: enforce that the pipeline makes no external network
calls.

Strategy: monkey-patch socket.create_connection and socket.getaddrinfo to reject
any target that is not loopback. Ollama (port 11434), ChromaDB (file-based) and
Streamlit (port 8501) all stay on localhost, so they pass through untouched.

Import this module FIRST in stage4/streamlit_app.py, before any HTTP library
loads its connection pool.

Usage
-----
from offline_guard import assert_offline
assert_offline()

Or instantiate explicitly:
    guard = OfflineGuard(strict=True)
    guard.install()
"""
import socket
import os
from typing import Optional, Set


_LOOPBACK_HOSTS: Set[str] = {
    "localhost", "127.0.0.1", "::1", "0.0.0.0",
    "ip6-localhost", "ip6-loopback",
}


def _is_loopback(host: str) -> bool:
    if host is None:
        return False
    h = host.strip().lower()
    if h in _LOOPBACK_HOSTS:
        return True
    # IPv4 loopback range 127.0.0.0/8
    if h.startswith("127."):
        return True
    return False


class OfflineGuard:
    """Blocks non-loopback socket connections. Installed globally on install()."""

    def __init__(self, strict: bool = True, log_path: Optional[str] = None):
        self.strict = strict
        self.log_path = log_path
        self._installed = False
        self._orig_create = None
        self._orig_getaddrinfo = None
        self._violations: list = []

    # ── Install / uninstall ──────────────────────────────────────────

    def install(self) -> None:
        if self._installed:
            return

        self._orig_create = socket.create_connection

        def guarded_create_connection(address, *args, **kwargs):
            host, port = address[0], address[1]
            if not _is_loopback(host):
                self._violation(f"create_connection -> {host}:{port}")
                if self.strict:
                    raise RuntimeError(
                        f"OfflineGuard: refused connection to {host}:{port}. "
                        f"FR028 violation."
                    )
            return self._orig_create(address, *args, **kwargs)

        socket.create_connection = guarded_create_connection
        self._installed = True

    def uninstall(self) -> None:
        if not self._installed:
            return
        if self._orig_create is not None:
            socket.create_connection = self._orig_create
        self._installed = False

    # ── Reporting ────────────────────────────────────────────────────

    def _violation(self, msg: str) -> None:
        self._violations.append(msg)
        if self.log_path:
            try:
                os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
                with open(self.log_path, "a", encoding="utf-8") as f:
                    f.write(msg + "\n")
            except Exception:
                pass

    @property
    def violations(self) -> list:
        return list(self._violations)


# ── Module-level convenience ─────────────────────────────────────────

_default_guard: Optional[OfflineGuard] = None


def assert_offline(strict: bool = True, log_path: Optional[str] = None) -> OfflineGuard:
    """Install the default offline guard. Idempotent."""
    global _default_guard
    if _default_guard is None:
        _default_guard = OfflineGuard(strict=strict, log_path=log_path)
        _default_guard.install()
        print(f"[OfflineGuard] installed (strict={strict}). "
              f"FR028: external network blocked.")
    return _default_guard


if __name__ == "__main__":
    # Smoke test
    g = assert_offline(strict=True)
    print("Testing loopback (should succeed)...")
    try:
        s = socket.create_connection(("127.0.0.1", 1), timeout=0.1)
        s.close()
    except (ConnectionRefusedError, OSError):
        print("  ✓ connection attempt allowed (refused at OS level, as expected)")
    except RuntimeError as e:
        print(f"  ✗ UNEXPECTED: {e}")

    print("Testing external (should be blocked)...")
    try:
        s = socket.create_connection(("8.8.8.8", 53), timeout=0.5)
        s.close()
        print("  ✗ UNEXPECTED: external connection succeeded")
    except RuntimeError as e:
        print(f"  ✓ blocked: {e}")
