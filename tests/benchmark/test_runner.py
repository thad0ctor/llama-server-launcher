"""Headless tests for :class:`BenchRunner` that don't need a real binary.

The runner spawns real subprocesses, so we use ``sys.executable -c "..."`` as a
portable, cross-platform stub that prints known stdout + stderr. Tests assert
the captured stdout and the emitted StepResult / lifecycle events, plus that
``_emit_event`` can't wedge the worker forever when cancelled with a full queue.
"""

from __future__ import annotations

import queue
import sys
import time

import pytest

from modules.benchmark.bench_runner import (
    EVENT_CANCELLED,
    EVENT_DONE,
    EVENT_ERROR,
    EVENT_LINE,
    EVENT_STEP_RESULT,
    EVENT_STEP_START,
    BenchPlan,
    BenchRunner,
    BenchStep,
    StepInfo,
    StepResult,
)

STUB_TOOL = "llama-bench"


def _drain_until_terminal(runner: BenchRunner, timeout: float = 5.0) -> list[tuple[str, object]]:
    """Collect events until a terminal event (done/cancelled/error) or timeout."""
    events: list[tuple[str, object]] = []
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            kind, payload = runner.events.get(timeout=0.1)
        except queue.Empty:
            if not runner.is_running and runner.events.empty():
                break
            continue
        events.append((kind, payload))
        if kind in (EVENT_DONE, EVENT_CANCELLED, EVENT_ERROR):
            break
    return events


def _py_stub(stdout_text: str, stderr_text: str, rc: int = 0) -> list[str]:
    code = (
        "import sys;"
        f"sys.stdout.write({stdout_text!r});"
        "sys.stdout.flush();"
        f"sys.stderr.write({stderr_text!r});"
        "sys.stderr.flush();"
        f"sys.exit({int(rc)})"
    )
    return [sys.executable, "-c", code]


def test_runner_captures_stdout_and_emits_result(tmp_path):
    stdout_text = "RESULT_MARKER_STDOUT\nsecond line\n"
    stderr_text = "PROGRESS_MARKER_STDERR\n"
    step = BenchStep(cmd=_py_stub(stdout_text, stderr_text), combo={"ngl": "10"}, label="ngl=10")
    plan = BenchPlan(tool=STUB_TOOL, steps=[step], cwd=str(tmp_path))

    runner = BenchRunner()
    assert runner.start(plan) is True
    events = _drain_until_terminal(runner)

    kinds = [k for k, _ in events]
    assert EVENT_STEP_START in kinds
    assert EVENT_STEP_RESULT in kinds
    assert kinds[-1] == EVENT_DONE

    # step_start payload
    start_info = next(p for k, p in events if k == EVENT_STEP_START)
    assert isinstance(start_info, StepInfo)
    assert start_info.index == 0
    assert start_info.total == 1
    assert start_info.combo == {"ngl": "10"}

    # step_result payload: stdout captured intact, rc == 0, tool preserved
    result = next(p for k, p in events if k == EVENT_STEP_RESULT)
    assert isinstance(result, StepResult)
    assert result.rc == 0
    assert result.tool == STUB_TOOL
    assert result.combo == {"ngl": "10"}
    assert "RESULT_MARKER_STDOUT" in result.stdout
    assert "second line" in result.stdout

    # stderr is streamed as line events (progress noise), never into stdout
    line_events = [str(p) for k, p in events if k == EVENT_LINE]
    assert any("PROGRESS_MARKER_STDERR" in line for line in line_events)
    assert "PROGRESS_MARKER_STDERR" not in result.stdout

    # done payload is the failed count (0 here)
    done_payload = next(p for k, p in events if k == EVENT_DONE)
    assert int(done_payload) == 0

    assert not runner.is_running


def test_runner_reports_nonzero_rc_as_failure(tmp_path):
    step = BenchStep(cmd=_py_stub("out\n", "err\n", rc=3), combo={}, label="")
    plan = BenchPlan(tool=STUB_TOOL, steps=[step], cwd=str(tmp_path))

    runner = BenchRunner()
    assert runner.start(plan) is True
    events = _drain_until_terminal(runner)

    result = next(p for k, p in events if k == EVENT_STEP_RESULT)
    assert result.rc == 3
    done_payload = next(p for k, p in events if k == EVENT_DONE)
    assert int(done_payload) == 1


def test_runner_signals_truncation_when_capture_exceeds_cap(tmp_path, monkeypatch):
    # With a small injected cap, an over-long stdout must be capped AND a
    # truncation WARNING emitted, so a silently-truncated JSON array (which would
    # parse to zero rows) is at least signalled rather than reported as a clean
    # Done with no results.
    monkeypatch.setattr("modules.benchmark.bench_runner.MAX_CAPTURE_BYTES", 16)
    big = "X" * 4096
    step = BenchStep(cmd=_py_stub(big, "err\n"), combo={}, label="")
    plan = BenchPlan(tool=STUB_TOOL, steps=[step], cwd=str(tmp_path))

    runner = BenchRunner()
    assert runner.start(plan) is True
    events = _drain_until_terminal(runner)

    lines = [str(p) for k, p in events if k == EVENT_LINE]
    assert any("truncated" in line.lower() for line in lines)
    result = next(p for k, p in events if k == EVENT_STEP_RESULT)
    assert len(result.stdout) <= 16


def test_runner_empty_plan_emits_error():
    runner = BenchRunner()
    plan = BenchPlan(tool=STUB_TOOL, steps=[])
    assert runner.start(plan) is True
    events = _drain_until_terminal(runner)
    assert events and events[-1][0] == EVENT_ERROR


def test_emit_event_does_not_block_when_cancelled_with_full_queue():
    """With a full queue and no consumer, _emit_event must return promptly once
    the cancel flag is set instead of parking the worker thread forever."""
    runner = BenchRunner()
    # Fill the event queue to capacity so any further put would block.
    filled = 0
    try:
        while True:
            runner.events.put_nowait((EVENT_LINE, f"x{filled}"))
            filled += 1
    except queue.Full:
        pass
    assert runner.events.full()

    # Simulate a cancel: the consumer has stopped and won't drain the queue.
    runner._cancel.set()

    start = time.monotonic()
    runner._emit_event(EVENT_DONE, 0)  # would block forever without the fix
    elapsed = time.monotonic() - start

    assert elapsed < 2.0, f"_emit_event blocked for {elapsed:.2f}s under cancel"
    # The dropped terminal event must not have displaced anything already queued.
    assert runner.events.full()


def test_emit_event_delivers_when_space_frees_up():
    """When not cancelled, _emit_event should retry and succeed once the
    consumer drains a slot, rather than dropping the event."""
    runner = BenchRunner()
    filled = 0
    try:
        while True:
            runner.events.put_nowait((EVENT_LINE, f"y{filled}"))
            filled += 1
    except queue.Full:
        pass
    assert runner.events.full()

    import threading

    def _drain_one_after_delay():
        time.sleep(0.25)
        runner.events.get_nowait()

    t = threading.Thread(target=_drain_one_after_delay, daemon=True)
    t.start()

    start = time.monotonic()
    runner._emit_event(EVENT_STEP_RESULT, "delivered")
    elapsed = time.monotonic() - start
    t.join(timeout=2.0)

    assert elapsed < 3.0
    # Our event is somewhere in the queue now (the freed slot got refilled).
    drained = []
    try:
        while True:
            drained.append(runner.events.get_nowait())
    except queue.Empty:
        pass
    assert (EVENT_STEP_RESULT, "delivered") in drained


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))


def test_env_unset_removes_inherited_var(tmp_path, monkeypatch):
    # env_unset must drop an inherited variable from the child's environment
    # (the resolver's "unset CUDA_VISIBLE_DEVICES" action).
    monkeypatch.setenv("BENCH_TEST_UNSET", "INHERITED")
    code = "import os,sys;sys.stdout.write('VAL=' + os.environ.get('BENCH_TEST_UNSET','<absent>'))"
    step = BenchStep(cmd=[sys.executable, "-c", code])
    plan = BenchPlan(tool=STUB_TOOL, steps=[step], cwd=str(tmp_path), env_unset=["BENCH_TEST_UNSET"])

    runner = BenchRunner()
    assert runner.start(plan) is True
    events = _drain_until_terminal(runner)

    results = [p for k, p in events if k == EVENT_STEP_RESULT]
    assert results and results[0].stdout.strip() == "VAL=<absent>"
