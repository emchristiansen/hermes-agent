"""A managed correction is never lost with the process that accepted it (real owner daemon, real
``agent.managed_worker`` child, loopback model).

- The child is killed after it answered ``queued`` (A's reply lost and A resent, B on its own). Once
  the owner's supported recovery (``prompt.resolve_unknown``) releases the session, every kept
  correction runs and reaches the model; the killed turn's own input is never replayed.
- A failed final usage flush is reported as the failed turn it is: the owner settles it ``failed``
  (or, when the store cannot finish either, ``unknown`` until resolved), never completed, and no
  correction is confirmed consumed, so each one still runs as its own turn.
- A correction the child confirmed consumed, in an execution whose store then cannot finish, settles
  with the lost claim's fence: it ran, so ``interrupted``, never run again and never left in transfer.
  One whose hand-over write committed and then raised is not recorded as consumed: the lost claim's
  fence queues it again, so it runs again (a duplicate) and is never left in transfer.
- When the lost claim's own transfer settlement cannot be written, the supported resolution of its
  ``unknown`` turn settles them instead: a confirmed correction is never run again, an unconfirmed
  one runs again, and the turn queued behind them runs (NC8-OWN-01).
"""
import asyncio
import json
import signal
import sqlite3
import textwrap

import psutil
import pytest

from tests.gateway.fixtures.local_recovery_probe import rpc, websocket
from tests.gateway.test_managed_worker_corrections import (
    MANAGED, ManagedCorrectionModel, _blocked, _generation, _kept, _managed_session, _peer, _transcript,
    _turns, _user_text,
)
from tests.gateway.test_session_busy_controls import owner, settled
from tests.waiting import async_wait_until


def _turn_row(home):
    with sqlite3.connect(f'file:{home / "state.db"}?mode=ro', uri=True) as db:
        db.row_factory = sqlite3.Row
        return dict(db.execute("SELECT * FROM session_admissions WHERE request_id='turn'").fetchone())


def _worker(peer):
    """The owner's live managed child (exactly one)."""
    children = [p for p in psutil.Process(peer.owner_pid).children(recursive=True)
                if p.cmdline()[-2:] == ['-m', 'agent.managed_worker']]
    assert len(children) == 1, children
    return children[0]


def _prompted(peer, text):
    """Provider requests whose newest user message is exactly ``text``: it ran as its own turn."""
    return [r for r in peer.requests
            if next((_user_text(m) for m in reversed(r['messages']) if m['role'] == 'user'), None) == text]


async def _resolve(ws, sid, row):
    reply = await rpc(ws, 'prompt.resolve_unknown', session_id=sid, admission_id=row['admission_id'],
                      execution_generation=row['generation'])
    assert reply.get('result', {}).get('status') == 'terminal', reply


@pytest.mark.platforms("linux")
def test_corrections_a_killed_child_accepted_reach_the_model_after_the_supported_recovery(tmp_path):
    with owner(tmp_path, MANAGED, handler=ManagedCorrectionModel) as (home, peer, desc):
        _peer(peer, tmp_path / 'unused-marker')

        async def probe():
            async with websocket(home, desc) as ws:
                sid = await _managed_session(ws, home, peer)
                await rpc(ws, 'prompt.submit', session_id=sid, input_id='turn', text='BLOCK_CRASH')
                await _blocked(peer, 1)
                generation = await _generation(ws, sid)
                async with websocket(home, desc) as lost:  # A's reply is never read
                    await lost.send(json.dumps({'jsonrpc': '2.0', 'id': 'lost', 'method': 'session.steer',
                        'params': {'session_id': sid, 'execution_generation': generation, 'text': 'CRASH_A'}}))
                b = await rpc(ws, 'session.steer', session_id=sid, execution_generation=generation, text='CRASH_B')
                again = await rpc(ws, 'session.steer', session_id=sid, execution_generation=generation, text='CRASH_A')
                # An ordinary prompt queued behind the turn: it must run too once the turn is resolved.
                await rpc(ws, 'prompt.submit', session_id=sid, input_id='follower', text='FOLLOWER_PLAIN')
                for reply in (b, again):
                    assert reply.get('result', {}).get('status') == 'queued', reply
                    assert reply['result']['admission_id'], reply  # durable before this ACK
                # B and the resent A were kept before their ACKs. The first A, whose sender gave up on
                # its reply, may or may not have been kept: either is fine, its resend covers it.
                await async_wait_until(lambda: len(_kept(home)) >= 2, timeout=40, interval=.05, what='kept')
                # The child holds them in memory, unconsumed (its model request is parked). Kill it.
                child = _worker(peer)
                child.send_signal(signal.SIGKILL)
                await async_wait_until(lambda: _turn_row(home)['status'] == 'unknown', timeout=40, interval=.05,
                                       what='the lost turn fenced unknown')
                # Recovery may proceed only once the killed worker can no longer compete.
                assert not child.is_running() or child.status() == psutil.STATUS_ZOMBIE
                kept_a = [text for *_, text in _kept(home)].count('CRASH_A')
                assert kept_a in (1, 2) and [text for *_, text in _kept(home)].count('CRASH_B') == 1, _kept(home)
                assert all(status == 'queued' for status, *_ in _kept(home)), _kept(home)
                # Resolved at once, while the owner may still be tearing the lost worker down: the
                # resolution must still release the FIFO (it once found that drain running and was lost).
                await _resolve(ws, sid, _turn_row(home))
                await settled(home)
            assert _turns(home) == [('turn', 'terminal'), ('follower', 'terminal')]
            assert _turn_row(home)['outcome'] == 'interrupted'
            # Every kept correction ran as its own turn and reached the model: B once, A once per kept
            # copy (a duplicate is delivery). The killed turn's own input was never sent again.
            kept = _kept(home)
            assert [(status, outcome) for status, outcome, _ in kept] == [('terminal', 'completed')] * len(kept)
            assert len(_prompted(peer, 'CRASH_A')) == kept_a and len(_prompted(peer, 'CRASH_B')) == 1
            assert len(peer.blocked) == 1 and len(_prompted(peer, 'FOLLOWER_PLAIN')) == 1
            users = [content for role, content in _transcript(home, sid) if role == 'user']
            assert users.count('CRASH_A') == kept_a and users.count('CRASH_B') == 1, users
        asyncio.run(probe())


def _failing_store_site(tmp_path, *, flush_fails=True, finish_fails=False):
    """A ``sitecustomize`` for managed children only: with ``flush_fails`` the first final usage flush
    (from ``managed_worker._flushed``) fails with the transport's uncertain-ACK error; with
    ``finish_fails`` that store's first finish fails too, as a latched store's would. Each fault
    fires once (a flag file it consumes), so later turns run normally."""
    site = tmp_path / 'site'
    site.mkdir()
    flush_flag, finish_flag = tmp_path / 'fail-flush', tmp_path / 'fail-finish'
    if flush_fails:
        flush_flag.touch()
    if finish_fails:
        finish_flag.touch()
    (site / 'sitecustomize.py').write_text(textwrap.dedent(f'''
        import importlib.abc, importlib.util, os, sys

        def _once(flag):
            try:
                os.unlink(flag)
                return True
            except FileNotFoundError:
                return False

        class _Hook(importlib.abc.MetaPathFinder):
            def find_spec(self, name, path, target=None):
                if name != 'agent.runtime_session_store':
                    return None
                sys.meta_path.remove(self)
                spec = importlib.util.find_spec(name)
                exec_module = spec.loader.exec_module

                def patched(module):
                    exec_module(module)
                    store, flush, finish = module.RuntimeSessionStore, module.RuntimeSessionStore.flush_token_counts, \\
                        module.RuntimeSessionStore.finish

                    def flush_token_counts(self, *args, **kwargs):
                        if sys._getframe(1).f_code.co_name == '_flushed' and _once({str(flush_flag)!r}):
                            raise TimeoutError('worker_receipt_timeout')
                        return flush(self, *args, **kwargs)

                    def failing_finish(self, *args, **kwargs):
                        if _once({str(finish_flag)!r}):
                            raise TimeoutError('worker_receipt_timeout')
                        return finish(self, *args, **kwargs)
                    store.flush_token_counts = flush_token_counts
                    store.finish = failing_finish
                spec.loader.exec_module = patched
                return spec

        if sys.argv[-2:] == ['-m', 'agent.managed_worker'] or 'agent.managed_worker' in sys.orig_argv:
            sys.meta_path.insert(0, _Hook())
    '''))
    return site


@pytest.mark.platforms("linux")
@pytest.mark.parametrize('finish_fails', [False, True], ids=['finished', 'finish-lost'])
def test_a_failed_final_flush_settles_failed_and_confirms_no_correction(tmp_path, finish_fails):
    site = _failing_store_site(tmp_path, finish_fails=finish_fails)
    with owner(tmp_path, MANAGED, handler=ManagedCorrectionModel, site=site) as (home, peer, desc):
        _peer(peer, tmp_path / 'unused-marker')

        async def probe():
            async with websocket(home, desc) as ws:
                sid = await _managed_session(ws, home, peer)
                await rpc(ws, 'prompt.submit', session_id=sid, input_id='turn', text='BLOCK_FINAL')
                await _blocked(peer, 1)
                generation = await _generation(ws, sid)
                steer = await rpc(ws, 'session.steer', session_id=sid, execution_generation=generation,
                                  text='FLUSH_STEER')
                assert steer.get('result', {}).get('status') == 'queued', steer
                peer.release.set()
                if finish_fails:
                    await async_wait_until(lambda: _turn_row(home)['status'] == 'unknown', timeout=40,
                                           interval=.05, what='the unfinished execution fenced unknown')
                    assert _kept(home) == [('queued', None, 'FLUSH_STEER')]  # never confirmed consumed
                    await _resolve(ws, sid, _turn_row(home))
                await settled(home)
                receipt = (await rpc(ws, 'prompt.receipt', session_id=sid,
                                     admission_id=_turn_row(home)['admission_id']))['result']
            row = _turn_row(home)
            if finish_fails:
                # The owner never learned the execution finished: unknown, resolved as interrupted.
                assert (row['status'], row['outcome']) == ('terminal', 'interrupted'), row
            else:
                assert (row['status'], row['outcome']) == ('terminal', 'failed'), row
                from hermes_state_terminal import RESULT_PREFIX
                with sqlite3.connect(f'file:{home / "state.db"}?mode=ro', uri=True) as db:
                    stored = db.execute("SELECT value FROM state_meta WHERE key=?",
                                        (RESULT_PREFIX + row['admission_id'],)).fetchone()
                assert stored and 'session_persistence_failed' in stored[0], stored
            assert receipt['outcome'] == row['outcome'], receipt
            # The steer reached the model inside the turn (its follow-up), but the turn's persistence
            # failed, so it was not confirmed: it also ran as its own turn. A duplicate, never a loss.
            assert [(status, outcome) for status, outcome, _ in _kept(home)] == [('terminal', 'completed')]
            assert len(_prompted(peer, 'FLUSH_STEER')) == 2, [r['messages'][-1] for r in peer.requests]
        asyncio.run(probe())


@pytest.mark.platforms("linux")
def test_a_consumed_correction_of_an_execution_that_cannot_finish_settles_and_never_runs_again(tmp_path):
    site = _failing_store_site(tmp_path, flush_fails=False, finish_fails=True)
    with owner(tmp_path, MANAGED, handler=ManagedCorrectionModel, site=site) as (home, peer, desc):
        _peer(peer, tmp_path / 'unused-marker')

        async def probe():
            async with websocket(home, desc) as ws:
                sid = await _managed_session(ws, home, peer)
                await rpc(ws, 'prompt.submit', session_id=sid, input_id='turn', text='BLOCK_FINAL')
                await _blocked(peer, 1)
                generation = await _generation(ws, sid)
                steer = await rpc(ws, 'session.steer', session_id=sid, execution_generation=generation,
                                  text='FINISH_STEER')
                assert steer.get('result', {}).get('status') == 'queued', steer
                peer.release.set()
                # The follow-up turn ran the steer and its flush succeeded, so the child confirmed it
                # (``consumed``); then its store could not finish: the owner fences the turn unknown.
                await async_wait_until(lambda: _turn_row(home)['status'] == 'unknown', timeout=40,
                                       interval=.05, what='the unfinished execution fenced unknown')
                await async_wait_until(lambda: _kept(home) == [('terminal', 'interrupted', 'FINISH_STEER')],
                                       timeout=10, interval=.05, what='the consumed correction settled with it')
                pending = (await rpc(ws, 'session.resume', session_id=sid))['result']['pending']
                assert 'FINISH_STEER' not in json.dumps(pending), pending  # not shown as still in transfer
                await _resolve(ws, sid, _turn_row(home))
                await settled(home)
            assert (_turn_row(home)['status'], _turn_row(home)['outcome']) == ('terminal', 'interrupted')
            assert _kept(home) == [('terminal', 'interrupted', 'FINISH_STEER')]
            # Reached the model once, in the follow-up turn; never run again as its own turn.
            assert len(_prompted(peer, 'FINISH_STEER')) == 1, [r['messages'][-1] for r in peer.requests]
        asyncio.run(probe())


def _failing_handover_site(tmp_path, *, handover_fails=True, settlement_fails=False):
    """``_failing_store_site`` (the child's store cannot finish) plus, in the OWNER only: with
    ``handover_fails`` the first ``hermes_state_runtime.steer_session_input`` (the consumed hand-over
    write) commits and then raises, as a write whose reply was lost after its commit would; with
    ``settlement_fails`` the first transfer settlement of a lost claim (``settle_steer_transfers``,
    outcome ``unknown``) fails inside its transaction, as a full disk would (a held writer lock
    would only be retried by the store). Each
    fault fires once; returns the site and the hand-over flag (consumed once it fired)."""
    site = _failing_store_site(tmp_path, flush_fails=False, finish_fails=True)
    flag, settlement_flag = tmp_path / 'fail-handover', tmp_path / 'fail-settlement'
    if handover_fails:
        flag.touch()
    if settlement_fails:
        settlement_flag.touch()
    with (site / 'sitecustomize.py').open('a') as f:
        f.write(textwrap.dedent(f'''
            class _OwnerHook(importlib.abc.MetaPathFinder):
                def find_spec(self, name, path, target=None):
                    if name != 'hermes_state_runtime':
                        return None
                    sys.meta_path.remove(self)
                    spec = importlib.util.find_spec(name)
                    exec_module = spec.loader.exec_module

                    def patched(module):
                        exec_module(module)
                        original = module.steer_session_input

                        def steer_session_input(*args, **kwargs):
                            result = original(*args, **kwargs)
                            if _once({str(flag)!r}):
                                raise RuntimeError('hand-over reply lost after commit')
                            return result
                        module.steer_session_input = steer_session_input
                        settle = module.settle_steer_transfers

                        def settle_steer_transfers(conn, **kwargs):
                            if kwargs.get('outcome') == 'unknown' and kwargs.get('transfers') \\
                                    and _once({str(settlement_flag)!r}):
                                import sqlite3
                                raise sqlite3.OperationalError('database or disk is full')
                            return settle(conn, **kwargs)
                        module.settle_steer_transfers = settle_steer_transfers
                    spec.loader.exec_module = patched
                    return spec

            if not (sys.argv[-2:] == ['-m', 'agent.managed_worker'] or 'agent.managed_worker' in sys.orig_argv):
                sys.meta_path.insert(0, _OwnerHook())
        '''))
    return site, flag


@pytest.mark.platforms("linux")
def test_a_sole_correction_whose_hand_over_was_not_confirmed_runs_again_after_a_loss(tmp_path):
    site, flag = _failing_handover_site(tmp_path)
    with owner(tmp_path, MANAGED, handler=ManagedCorrectionModel, site=site) as (home, peer, desc):
        _peer(peer, tmp_path / 'unused-marker')

        async def probe():
            async with websocket(home, desc) as ws:
                sid = await _managed_session(ws, home, peer)
                await rpc(ws, 'prompt.submit', session_id=sid, input_id='turn', text='BLOCK_FINAL')
                await _blocked(peer, 1)
                generation = await _generation(ws, sid)
                steer = await rpc(ws, 'session.steer', session_id=sid, execution_generation=generation,
                                  text='HANDOVER_STEER')
                assert steer.get('result', {}).get('status') == 'queued', steer
                peer.release.set()
                await async_wait_until(lambda: _turn_row(home)['status'] == 'unknown', timeout=40,
                                       interval=.05, what='the unfinished execution fenced unknown')
                # Its hand-over committed, then raised: not confirmed, so the fence queues it again.
                await async_wait_until(lambda: _kept(home) == [('queued', None, 'HANDOVER_STEER')],
                                       timeout=10, interval=.05, what='the unconfirmed correction queued again')
                assert not flag.exists()  # the hand-over fault fired
                await _resolve(ws, sid, _turn_row(home))
                await rpc(ws, 'prompt.submit', session_id=sid, input_id='later', text='LATER_TURN')
                await settled(home)
            assert _kept(home) == [('terminal', 'completed', 'HANDOVER_STEER')]
            # Reached the model in the lost turn's follow-up and again as its own turn: a duplicate
            # after an uncertain hand-over, never left in transfer; the later message runs too.
            assert len(_prompted(peer, 'HANDOVER_STEER')) == 2, [r['messages'][-1] for r in peer.requests]
            assert len(_prompted(peer, 'LATER_TURN')) == 1, [r['messages'][-1] for r in peer.requests]
        asyncio.run(probe())


@pytest.mark.platforms("linux")
@pytest.mark.parametrize('confirmed', [True, False], ids=['confirmed-consumed', 'hand-over-unconfirmed'])
def test_a_failed_transfer_settlement_after_a_loss_settles_with_the_supported_resolution(tmp_path, confirmed):
    """NC8-OWN-01: the lost claim's own transfer settlement fails once. The correction stays bound to
    that claim (``terminal/steering``) until the supported ``prompt.resolve_unknown`` of the lost
    turn settles it in the same owner: confirmed consumed, it is ``interrupted`` and never runs
    again; unconfirmed (its hand-over committed, then raised), it is queued again and runs again, a
    duplicate. Either way nothing is left in transfer and the turn queued behind it runs."""
    site, flag = _failing_handover_site(tmp_path, handover_fails=not confirmed, settlement_fails=True)
    with owner(tmp_path, MANAGED, handler=ManagedCorrectionModel, site=site) as (home, peer, desc):
        _peer(peer, tmp_path / 'unused-marker')

        async def probe():
            async with websocket(home, desc) as ws:
                sid = await _managed_session(ws, home, peer)
                await rpc(ws, 'prompt.submit', session_id=sid, input_id='turn', text='BLOCK_FINAL')
                await _blocked(peer, 1)
                generation = await _generation(ws, sid)
                steer = await rpc(ws, 'session.steer', session_id=sid, execution_generation=generation,
                                  text='SETTLE_STEER')
                assert steer.get('result', {}).get('status') == 'queued', steer
                await rpc(ws, 'prompt.submit', session_id=sid, input_id='later', text='LATER_TURN')
                peer.release.set()
                await async_wait_until(lambda: _turn_row(home)['status'] == 'unknown', timeout=40,
                                       interval=.05, what='the unfinished execution fenced unknown')
                await async_wait_until(lambda: not (tmp_path / 'fail-settlement').exists(), timeout=10,
                                       interval=.05, what="the lost claim's transfer settlement failed")
                assert _kept(home) == [('terminal', 'steering', 'SETTLE_STEER')]  # still in transfer
                assert confirmed or not flag.exists()  # unconfirmed: its hand-over fault fired
                await _resolve(ws, sid, _turn_row(home))
                await settled(home)
                pending = (await rpc(ws, 'session.resume', session_id=sid))['result']['pending']
                assert 'SETTLE_STEER' not in json.dumps(pending), pending
            assert (_turn_row(home)['status'], _turn_row(home)['outcome']) == ('terminal', 'interrupted')
            assert _kept(home) == [('terminal', 'interrupted' if confirmed else 'completed', 'SETTLE_STEER')]
            # Once in the lost turn's follow-up; unconfirmed, once more as its own turn.
            assert len(_prompted(peer, 'SETTLE_STEER')) == (1 if confirmed else 2), \
                [r['messages'][-1] for r in peer.requests]
            assert len(_prompted(peer, 'LATER_TURN')) == 1, [r['messages'][-1] for r in peer.requests]
        asyncio.run(probe())
