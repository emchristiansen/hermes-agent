"""The gateway stop's own call, ``shutdown_all(reason, resumable=True)`` (gateway/run_shutdown.py), fences new
dispatches and interrupts every running child, and marks them resumable for the restart.

The v0.23.2 merge paired upstream's ``resumable=True`` call site with the candidate's fenced ``shutdown_all``,
which had no such parameter: the TypeError was swallowed by the stop step, so the stop neither fenced nor
interrupted. This makes the stop step's exact call against a real dispatched delegation."""
import threading

from tools import async_delegation as ad
from tools import detached_writers as dw
from gateway import run_shutdown_budget as budget


def test_the_stop_steps_resumable_call_fences_interrupts_and_marks(monkeypatch):
    from hermes_state_registry import acquire, close_all
    import tools.delegate_tool_resumable as resumable
    ad._reset_for_tests(); dw._reset_for_tests(); budget._reset_maintenance_for_tests()
    db = acquire(ad._db_path())
    entered, release, interrupted = threading.Event(), threading.Event(), []
    marked = []
    real_note = resumable.note_gateway_shutdown
    monkeypatch.setattr(resumable, 'note_gateway_shutdown', lambda: marked.append(real_note()) or marked[-1])

    def work():
        entered.set()
        assert release.wait(30)
        return {'status': 'interrupted'}

    h = ad.dispatch_async_delegation(goal='live child at the stop', context=None, toolsets=None, role='leaf',
        model=None, session_key='agent:main:telegram:dm:resumable', runner=work,
        interrupt_fn=lambda: (interrupted.append(True), release.set()))
    assert h['status'] == 'dispatched', h
    try:
        assert entered.wait(30)
        # Exactly the stop step's call (GatewayShutdownMixin._interrupt_delegations).
        count = ad.shutdown_all(reason='gateway shutdown (test)', resumable=True)
        assert count == 1
        assert interrupted == [True]  # the running child was interrupted
        assert len(marked) == 1  # and marked as a restart (resumable), not a user stop
        late = ad.dispatch_async_delegation(goal='after the stop', context=None, toolsets=None, role='leaf',
            model=None, session_key='agent:main:telegram:dm:resumable-late', runner=dict)
        assert late['status'] == 'rejected' and 'shutting down' in late['error'], late
    finally:
        release.set()
        if ad._executor is not None:
            ad._executor.shutdown(wait=True)
        if ad._monitor_thread is not None:
            ad._monitor_stop.set(); ad._monitor_thread.join(30)
        db.close(); close_all(); ad._reset_for_tests(); dw._reset_for_tests(); budget._reset_maintenance_for_tests()
