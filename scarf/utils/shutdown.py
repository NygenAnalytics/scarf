import signal
import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from types import FrameType
from typing import Any, Iterator


@dataclass(frozen=True, slots=True)
class ShutdownRequest:
    requested_at_ns: int
    reason: str
    signal_number: int | None = None
    signal_name: str | None = None

    def to_dict(self) -> dict[str, int | str | None]:
        return {
            "requestedAtNs": self.requested_at_ns,
            "reason": self.reason,
            "signalNumber": self.signal_number,
            "signalName": self.signal_name,
        }


class ShutdownRequested(BaseException):
    """Raised at a safe checkpoint after cooperative shutdown was requested."""

    def __init__(self, request: ShutdownRequest) -> None:
        self.request = request
        super().__init__(request.reason)


class ShutdownToken:
    """Thread-safe, runtime-neutral cooperative shutdown state."""

    __slots__ = ("_event", "_frame", "_lock", "_previous", "_request")

    def __init__(self) -> None:
        self._event = threading.Event()
        self._frame: FrameType | None = None
        self._lock = threading.Lock()
        self._previous: Any = None
        self._request: ShutdownRequest | None = None

    @property
    def requested(self) -> bool:
        return self._event.is_set()

    @property
    def request_record(self) -> ShutdownRequest | None:
        with self._lock:
            return self._request

    def request(
        self,
        *,
        reason: str = "shutdown requested",
        signal_number: int | None = None,
        previous_handler: Any = None,
        frame: FrameType | None = None,
    ) -> bool:
        """Record the first request and return whether this was the first one."""

        if not isinstance(reason, str) or not reason:
            raise TypeError("reason must be a non-empty string")
        signal_name: str | None = None
        if signal_number is not None:
            try:
                signal_name = signal.Signals(signal_number).name
            except ValueError:
                signal_name = f"SIGNAL_{signal_number}"
        with self._lock:
            if self._request is not None:
                return False
            self._request = ShutdownRequest(
                requested_at_ns=time.time_ns(),
                reason=reason,
                signal_number=signal_number,
                signal_name=signal_name,
            )
            self._previous = previous_handler
            self._frame = frame
            self._event.set()
            return True

    def checkpoint(self) -> None:
        request = self.request_record
        if request is not None:
            raise ShutdownRequested(request)

    def propagate(self) -> None:
        """Continue with the prior signal behavior after durable cleanup."""

        request = self.request_record
        if request is None:
            return
        if request.signal_number is None:
            raise ShutdownRequested(request)
        previous = self._previous
        if callable(previous):
            previous(request.signal_number, self._frame)
            raise ShutdownRequested(request)
        if previous == signal.SIG_IGN:
            raise ShutdownRequested(request)
        signal.raise_signal(request.signal_number)
        raise ShutdownRequested(request)


_CURRENT_TOKEN: ContextVar[ShutdownToken | None] = ContextVar(
    "scarf_shutdown_token",
    default=None,
)


@contextmanager
def shutdown_scope(token: ShutdownToken) -> Iterator[ShutdownToken]:
    if not isinstance(token, ShutdownToken):
        raise TypeError("token must be a ShutdownToken")
    context_token = _CURRENT_TOKEN.set(token)
    try:
        yield token
    finally:
        _CURRENT_TOKEN.reset(context_token)


def current_shutdown_token() -> ShutdownToken | None:
    return _CURRENT_TOKEN.get()


def shutdown_checkpoint() -> None:
    token = current_shutdown_token()
    if token is not None:
        token.checkpoint()


class TemporarySignalGuard:
    """Temporarily translate catchable termination signals into token requests."""

    __slots__ = ("_installed", "_token", "available", "unavailable_reason")

    def __init__(self, token: ShutdownToken) -> None:
        if not isinstance(token, ShutdownToken):
            raise TypeError("token must be a ShutdownToken")
        self._token = token
        self._installed: dict[int, Any] = {}
        self.available = False
        self.unavailable_reason: str | None = None

    def __enter__(self) -> "TemporarySignalGuard":
        if threading.current_thread() is not threading.main_thread():
            self.unavailable_reason = "signal handlers require the main thread"
            return self
        candidates = tuple(
            candidate
            for name in ("SIGTERM", "SIGINT", "SIGHUP")
            if (candidate := getattr(signal, name, None)) is not None
        )
        for signum in candidates:
            previous = signal.getsignal(signum)
            if previous == signal.SIG_IGN:
                continue

            def handler(
                received: int,
                frame: FrameType | None,
                *,
                prior: Any = previous,
            ) -> None:
                first = self._token.request(
                    reason=f"received {signal.Signals(received).name}",
                    signal_number=received,
                    previous_handler=prior,
                    frame=frame,
                )
                if not first:
                    self._escalate(received, frame, prior)

            signal.signal(signum, handler)
            self._installed[int(signum)] = previous
        self.available = bool(self._installed)
        if not self.available:
            self.unavailable_reason = "no catchable termination signals are available"
        return self

    @staticmethod
    def _escalate(signum: int, frame: FrameType | None, prior: Any) -> None:
        signal.signal(signum, prior)
        if callable(prior):
            prior(signum, frame)
            return
        if prior == signal.SIG_IGN:
            return
        signal.raise_signal(signum)

    def __exit__(self, *_exc: object) -> None:
        for signum, previous in self._installed.items():
            signal.signal(signum, previous)
        self._installed.clear()
