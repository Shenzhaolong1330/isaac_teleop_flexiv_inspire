"""Single-owner policy lease gated by robot-local authority."""

from __future__ import annotations

from dataclasses import dataclass
import secrets
import threading
import time


@dataclass(frozen=True)
class LocalControlState:
    session_id: str
    ft_zeroed: bool
    local_policy_authorized: bool
    pedal_valid: bool
    arms_online: bool
    hands_online: bool
    state: str

    @property
    def policy_lease_allowed(self) -> bool:
        return (
            self.state in {"READY", "POLICY_ARMED", "ACTIVE"}
            and self.ft_zeroed
            and self.local_policy_authorized
            and self.pedal_valid
            and self.arms_online
            and self.hands_online
        )


@dataclass(frozen=True)
class Lease:
    token: str
    client_id: str
    peer: str
    session_id: str
    expires_ns: int


class ControlLeaseManager:
    def __init__(self, *, max_lease_ms: int = 2000, clock_ns=time.monotonic_ns) -> None:
        self.max_lease_ns = int(max_lease_ms * 1e6)
        self.clock_ns = clock_ns
        self._lock = threading.Lock()
        self._lease: Lease | None = None

    def acquire(
        self,
        *,
        client_id: str,
        peer: str,
        requested_ms: int,
        local_state: LocalControlState,
    ) -> Lease:
        now = self.clock_ns()
        if not local_state.policy_lease_allowed:
            raise PermissionError("robot-local policy authorization is not satisfied")
        if not client_id or not peer:
            raise PermissionError("client identity and authenticated peer are required")
        duration_ns = min(max(1, requested_ms) * 1_000_000, self.max_lease_ns)
        with self._lock:
            if self._lease is not None and self._lease.expires_ns > now:
                if self._lease.client_id != client_id or self._lease.peer != peer:
                    raise PermissionError("another policy client owns the control lease")
            self._lease = Lease(
                token=secrets.token_urlsafe(32),
                client_id=client_id,
                peer=peer,
                session_id=local_state.session_id,
                expires_ns=now + duration_ns,
            )
            return self._lease

    def validate(
        self, token: str, peer: str, session_id: str, *, refresh_ms: int | None = None
    ) -> Lease:
        now = self.clock_ns()
        with self._lock:
            lease = self._lease
            if (
                lease is None
                or lease.expires_ns <= now
                or not secrets.compare_digest(lease.token, token)
                or lease.peer != peer
                or lease.session_id != session_id
            ):
                self._lease = None if lease is not None and lease.expires_ns <= now else lease
                raise PermissionError("invalid or expired policy lease")
            if refresh_ms is not None:
                lease = Lease(
                    **{
                        **lease.__dict__,
                        "expires_ns": now
                        + min(max(1, refresh_ms) * 1_000_000, self.max_lease_ns),
                    }
                )
                self._lease = lease
            return lease

    def current_valid(self, token: str, session_id: str) -> bool:
        """Check liveness without refreshing or exposing the lease token."""
        now = self.clock_ns()
        with self._lock:
            lease = self._lease
            if lease is None or lease.expires_ns <= now:
                if lease is not None and lease.expires_ns <= now:
                    self._lease = None
                return False
            return (
                secrets.compare_digest(lease.token, token)
                and lease.session_id == session_id
            )

    def release(self, token: str, peer: str) -> bool:
        with self._lock:
            lease = self._lease
            if (
                lease is None
                or lease.peer != peer
                or not secrets.compare_digest(lease.token, token)
            ):
                return False
            self._lease = None
            return True

    def invalidate(self) -> None:
        with self._lock:
            self._lease = None
