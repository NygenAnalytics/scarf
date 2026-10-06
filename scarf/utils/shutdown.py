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


class ShutdownRequested(BaseException):
    """Raised at a safe checkpoint after cooperative shutdown was requested."""

    def __init__(self, request: ShutdownRequest) -> None:
        self.request = request
        super().__init__(request.reason)


@dataclass(frozen=True, slots=True)
class _Claim:
    """The first shutdown request with the signal behavior it interrupted."""

    request: ShutdownRequest
    previous_handler: Any
    frame: FrameType | None


# The one key of a token's claims: the first request.
_FIRST_REQUEST = "first"


class ShutdownToken:
    """Thread-safe, runtime-neutral cooperative shutdown state."""

    __slots__ = ("_claims",)

    def __init__(self) -> None:
        # Holds at most the first request's claim, which is never replaced.
        self._claims: dict[str, _Claim] = {}

    @property
    def requested(self) -> bool:
        return _FIRST_REQUEST in self._claims

    @property
    def request_record(self) -> ShutdownRequest | None:
        claim = self._claims.get(_FIRST_REQUEST)
        return None if claim is None else claim.request

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
        claim = _Claim(
            request=ShutdownRequest(
                requested_at_ns=time.time_ns(),
                reason=reason,
                signal_number=signal_number,
                signal_name=signal_name,
            ),
            previous_handler=previous_handler,
            frame=frame,
        )
        return self._claims.setdefault(_FIRST_REQUEST, claim) is claim

    def checkpoint(self) -> None:
        claim = self._claims.get(_FIRST_REQUEST)
        if claim is not None:
            raise ShutdownRequested(claim.request)

    def propagate(self) -> None:
        """Continue with the prior signal behavior after durable cleanup."""

        claim = self._claims.get(_FIRST_REQUEST)
        if claim is None:
            return
        request = claim.request
        if request.signal_number is None:
            raise ShutdownRequested(request)
        previous = claim.previous_handler
        if callable(previous):
            previous(request.signal_number, claim.frame)
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
        try:
            for signum in candidates:
                previous = signal.getsignal(signum)
                # A handler installed outside Python reads as None and could
                # not be restored, so that signal keeps its current handler.
                if previous is None or previous == signal.SIG_IGN:
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

                # Record the prior handler first, so that a handler installed
                # just before an error is always put back.
                self._installed[int(signum)] = previous
                signal.signal(signum, handler)
        except BaseException:
            # A guard that fails to enter is never exited, so it puts back the
            # handlers it had already replaced before the error propagates.
            self._restore()
            raise
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

    def _restore(self) -> None:
        for signum, previous in self._installed.items():
            signal.signal(signum, previous)
        self._installed.clear()

    def __exit__(self, *_exc: object) -> None:
        self._restore()
