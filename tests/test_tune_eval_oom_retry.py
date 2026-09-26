import subprocess
from types import SimpleNamespace

from hydrabflow.pipeline import tune


def _run(monkeypatch, results):
    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        rc, err = results.pop(0)
        return SimpleNamespace(returncode=rc, stderr=err)

    monkeypatch.setattr(tune.subprocess, "run", fake_run)
    cfg = SimpleNamespace(eval=SimpleNamespace(batch_size=8))
    trial = SimpleNamespace(number=0, params={})
    return calls, lambda: tune._evaluate_subprocess(cfg, trial, "t", ["eval.batch_size=32"], "o")


def test_halves_eval_batch_on_oom(monkeypatch):
    calls, go = _run(monkeypatch, [(1, "RESOURCE_EXHAUSTED: Out of memory"), (0, "")])
    go()
    assert "eval.batch_size=32" in calls[0] and "eval.batch_size=16" in calls[1]
    assert calls[1].index("eval.batch_size=16") > calls[1].index("eval.batch_size=32")


def test_non_oom_failure_raises(monkeypatch):
    calls, go = _run(monkeypatch, [(1, "KeyError: nope")])
    try:
        go()
    except subprocess.CalledProcessError:
        assert len(calls) == 1
    else:
        raise AssertionError("expected CalledProcessError")
