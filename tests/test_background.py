"""Worker-thread tasks that pipeline stages overlap with."""

import contextvars
import threading

import pytest

import scarf.utils.background as background
from scarf.utils.background import BackgroundTask

_VALUE: contextvars.ContextVar[str] = contextvars.ContextVar("value", default="unset")


def test_background_task_runs_on_a_worker_in_the_callers_context() -> None:
    token = _VALUE.set("caller")
    try:
        task = BackgroundTask(
            lambda: (threading.current_thread(), _VALUE.get()),
            name="probe",
        )
        thread, value = task.result()
    finally:
        _VALUE.reset(token)
    assert thread is not threading.current_thread()
    assert value == "caller"

    def fail() -> None:
        raise ValueError("deliberate failure")

    with pytest.raises(ValueError, match="deliberate failure"):
        BackgroundTask(fail, name="probe").result()


def test_inline_tasks_run_before_the_constructor_returns(monkeypatch) -> None:
    ran: list[threading.Thread] = []
    task = BackgroundTask(
        lambda: ran.append(threading.current_thread()),
        name="probe",
        inline=True,
    )
    assert task.done()
    assert ran == [threading.current_thread()]

    monkeypatch.setattr(background, "threadsafe_threading_layer", lambda: False)
    task = BackgroundTask(threading.current_thread, name="probe")
    assert task.done()
    assert task.result() is threading.current_thread()
