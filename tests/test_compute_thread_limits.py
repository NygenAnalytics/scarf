import pytest
from threadpoolctl import ThreadpoolController, threadpool_limits

from scarf.utils.compute import (
    enter_thread_limit,
    exit_thread_limit,
    process_thread_limit,
)


def _blas_threads() -> set[int]:
    return {
        lib.num_threads
        for lib in ThreadpoolController().lib_controllers
        if lib.user_api == "blas"
    }


@pytest.fixture
def two_blas_threads():
    # The test session pins BLAS to one thread; raise it so a change shows.
    with threadpool_limits(limits=2, user_api="blas"):
        if not _blas_threads() or max(_blas_threads()) < 2:
            pytest.skip("needs a multi-threaded BLAS")
        yield


def test_the_latest_remaining_limit_applies_after_a_middle_exit(two_blas_threads):
    first = enter_thread_limit(1)
    second = enter_thread_limit(2)
    third = enter_thread_limit(1)
    exit_thread_limit(third)
    assert _blas_threads() == {2}
    exit_thread_limit(second)
    assert _blas_threads() == {1}
    exit_thread_limit(first)
    assert _blas_threads() == {2}


def test_a_scope_without_a_limit_still_restores_the_original(two_blas_threads):
    unchanged = enter_thread_limit(None)
    assert _blas_threads() == {2}
    limited = enter_thread_limit(1)
    exit_thread_limit(limited)
    # Only the unchanged scope remains, so the original limits apply.
    assert _blas_threads() == {2}
    exit_thread_limit(unchanged)
    assert _blas_threads() == {2}


def test_the_context_manager_restores_after_an_exception(two_blas_threads):
    with pytest.raises(RuntimeError, match="stage failed"):
        with process_thread_limit(1):
            assert _blas_threads() == {1}
            raise RuntimeError("stage failed")
    assert _blas_threads() == {2}
