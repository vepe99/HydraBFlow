"""OOM-backoff retry helper: halve the batch size on a RESOURCE_EXHAUSTED error and retry."""

from __future__ import annotations

import pytest

from hydrabflow.utils.oom import is_oom_error, run_with_oom_backoff


class _FakeOOM(RuntimeError):
    pass


def test_is_oom_error_matches_jax_and_cuda_messages():
    assert is_oom_error(RuntimeError("RESOURCE_EXHAUSTED: Out of memory while trying to allocate"))
    assert is_oom_error(RuntimeError("CUDA_ERROR_OUT_OF_MEMORY"))
    assert not is_oom_error(ValueError("shape mismatch"))


def test_backoff_halves_until_it_fits():
    # Succeeds only at batch_size <= 64; starts at 256 -> should try 256,128,64 and return at 64.
    tried = []

    def fn(bs):
        tried.append(bs)
        if bs > 64:
            raise _FakeOOM("RESOURCE_EXHAUSTED: Out of memory")
        return bs

    out = run_with_oom_backoff(fn, 256, min_batch=16)
    assert out == 64
    assert tried == [256, 128, 64]


def test_backoff_reraises_when_even_min_batch_oorms():
    def fn(bs):
        raise _FakeOOM("RESOURCE_EXHAUSTED")

    with pytest.raises(_FakeOOM):
        run_with_oom_backoff(fn, 64, min_batch=16)


def test_backoff_does_not_swallow_non_oom_errors():
    def fn(bs):
        raise ValueError("bad shape")

    with pytest.raises(ValueError):
        run_with_oom_backoff(fn, 256)


def test_no_retry_on_success():
    calls = []
    assert run_with_oom_backoff(lambda bs: (calls.append(bs), "ok")[1], 128) == "ok"
    assert calls == [128]
