import { beforeEach, expect, test, vi } from 'vitest'

// UI-02 (review t_42a8d5e2) and review22815 UI-R1..R4: a correction whose outcome is unknown survives a
// renderer restart and is delivered again on its own, as the same correction, to its own destination.
// A restart is a fresh load of every module (no pins, notices, retry closures or timers of the previous
// renderer; storage remains). Real CanonicalDesktopProtocol; the owner-routed request is recorded as the
// wire after the protocol's own prepare.
vi.mock('@/i18n', () => ({ translateNow: (key: string) => key }))
vi.mock('@/store/notifications', () => ({ notify: vi.fn(), dismissNotification: vi.fn() }))
vi.mock('@/store/gateway', () => ({ activeGatewayRoute: () => ({ connectionId: null, profile: 'active' }) }))
vi.mock('@/store/session-states', () => ({
  knownOwnerForSession: (id: string) => (id === 'c-session' ? 'work' : id === 'x-session' ? X : undefined)
}))
vi.mock('@/store/session-request-router', () => ({
  isSessionOwnerRoute: (owner: unknown) => Boolean(owner && typeof owner === 'object' && 'connectionId' in owner),
  profileScopeForSessionOwner: (owner: unknown) => owner ?? undefined,
  requestForSessionProfile: vi.fn()
}))

// Registry connection X's "work" (another connection exposes the same bare profile).
const X = { connectionId: 'X', profile: 'work', targetProfile: 'work' }

type Owner = string | { connectionId: string; profile: string } | null

beforeEach(() => {
  window.localStorage.clear()
  vi.resetModules()
})

async function renderer() {
  const protocol = await import('@/api/canonical-protocol')
  const store = await import('./unresolved-corrections')
  const { notify } = await import('@/store/notifications')
  const { requestForSessionProfile } = await import('@/store/session-request-router')
  vi.mocked(notify).mockClear()
  vi.mocked(requestForSessionProfile).mockReset()

  return { protocol, store, notify: vi.mocked(notify), gateway: vi.mocked(requestForSessionProfile) }
}

type Renderer = Awaited<ReturnType<typeof renderer>>

const profileOf = (owner: Owner) => (owner && typeof owner === 'object' ? owner.profile : (owner ?? 'default'))

/** Record each request as the wire: its owner route and the protocol's prepared frame. */
function wireTo(r: Renderer, protocol: InstanceType<Renderer['protocol']['CanonicalDesktopProtocol']>, wire: unknown[], reply = async (_frame: Record<string, unknown>): Promise<unknown> => ({ status: 'queued' })) {
  r.gateway.mockImplementation((async (owner: Owner, _ambient: unknown, method: string, params: Record<string, unknown>) => {
    const frame = { owner, method, ...protocol.prepare(method, { ...params, profile: profileOf(owner) }) }
    delete (frame as Record<string, unknown>).profile
    wire.push(frame)

    return reply(frame)
  }) as never)
}

test('A fails before delivery, the renderer restarts, B sends at once, A is delivered to its own destination', async () => {
  const first = await renderer()
  const before = new first.protocol.CanonicalDesktopProtocol()
  before.result('session.resume', { session_id: 's' }, { session_id: 's', revision: 1, execution_generation: 4 })
  before.prepare('session.steer', { session_id: 's', text: 'A words', correction_id: 'A' })
  expect(first.store.keepUnresolvedCorrection({ id: 'A', mode: 'steer', text: 'A words', sessionId: 's' }, 'A words')).toMatchObject({
    sessionId: 's',
    profile: 'default',
    generation: 4
  })

  vi.resetModules()
  const second = await renderer()
  expect(second.protocol.correctionPin('A')).toBeUndefined() // the restart lost every in-memory pin
  const after = new second.protocol.CanonicalDesktopProtocol()
  after.result('session.resume', { session_id: 's' }, { session_id: 's', revision: 2, execution_generation: 5 })
  const wire: unknown[] = []
  wireTo(second, after, wire)

  second.store.restoreUnresolvedCorrections()
  await second.gateway('default', undefined as never, 'session.steer', { session_id: 's', text: 'B words', correction_id: 'B' })

  const [offer] = second.notify.mock.calls.map(([input]) => input)
  expect(second.notify).toHaveBeenCalledTimes(1)
  expect(offer).toMatchObject({ id: 'unresolved-correction:A', persistent: true, detail: 'A words' })
  offer.action!.onClick() // Send again: now, rather than at its next unattended retry
  await vi.waitFor(() => expect(wire).toHaveLength(2))

  expect(wire).toEqual([
    { owner: 'default', method: 'session.steer', session_id: 's', text: 'B words', execution_generation: 5 },
    { owner: 'default', method: 'session.steer', session_id: 's', text: 'A words', execution_generation: 4 }
  ])
  await vi.waitFor(() => expect(second.store.unresolvedCorrections()).toEqual([]))
})

test('an unconfirmed restored retry keeps the correction; an undelivered refusal leaves its words in a notice', async () => {
  const first = await renderer()
  const protocol = new first.protocol.CanonicalDesktopProtocol()
  protocol.result('session.resume', { session_id: 's' }, { session_id: 's', revision: 1, execution_generation: 4 })
  protocol.prepare('session.redirect', { session_id: 's', text: 'A words', correction_id: 'A' })
  const entry = first.store.keepUnresolvedCorrection({ id: 'A', mode: 'interrupt', text: 'A words', sessionId: 's' }, 'A words')!

  await expect(first.store.resendUnresolvedCorrection(entry, (async () => ({ status: 'rejected' })) as never)).rejects.toThrow(
    /unconfirmed/
  )
  expect(first.store.unresolvedCorrections()).toHaveLength(1)

  first.gateway.mockResolvedValue({ status: 'rejected', delivered: false } as never)
  first.store.offerUnresolvedCorrection(entry)
  first.notify.mock.calls.at(-1)![0].action!.onClick()
  await vi.waitFor(() => expect(first.store.unresolvedCorrections()).toEqual([]))
  expect(first.notify.mock.calls.at(-1)![0]).toMatchObject({ title: 'composer.correctionNotSentTitle', detail: 'A words' })
})

// The at-least-once directive: a correction that failed before any preparation pinned it is kept too,
// with the session it was captured for and that session's owner. After a restart it goes to that session
// through that owner and pins the session's then-current generation; a session shown meanwhile is never
// its destination.
test('C fails before it is pinned, the renderer restarts, B sends at once, C is delivered to its own session', async () => {
  const first = await renderer()
  expect(first.store.keepUnresolvedCorrection({ id: 'C', mode: 'steer', text: 'C words', sessionId: 'c-session' }, 'C words'))
    .toMatchObject({ sessionId: 'c-session', profile: 'work', owner: 'work', generation: null })

  vi.resetModules()
  const second = await renderer()
  const after = new second.protocol.CanonicalDesktopProtocol()
  after.result('session.resume', { session_id: 'c-session', profile: 'work' }, { session_id: 'c-session', revision: 1, execution_generation: 7 })
  after.result('session.resume', { session_id: 'shown', profile: 'active' }, { session_id: 'shown', revision: 1, execution_generation: 2 })
  const wire: unknown[] = []
  wireTo(second, after, wire)

  second.store.restoreUnresolvedCorrections()
  await second.gateway('active', undefined as never, 'session.steer', { session_id: 'shown', text: 'B words', correction_id: 'B' })
  second.notify.mock.calls[0][0].action!.onClick()
  await vi.waitFor(() => expect(wire).toHaveLength(2))

  expect(wire).toEqual([
    { owner: 'active', method: 'session.steer', session_id: 'shown', text: 'B words', execution_generation: 2 },
    { owner: 'work', method: 'session.steer', session_id: 'c-session', text: 'C words', execution_generation: 7 }
  ])
  expect(second.protocol.correctionPin('C')).toEqual({ sessionId: 'c-session', profile: 'work', generation: 7 })
  await vi.waitFor(() => expect(second.store.unresolvedCorrections()).toEqual([]))
})

// review22815 UI-R2/R3/R4/R1 together: journaled before its first request (which never answers), kept
// with X's exact route; after a restart it is delivered with nobody clicking, through X, while B goes at
// once; its delivery pins generation 7 and the pin is kept before any reply; every reply there is lost;
// after another restart (the session now at generation 8) it still names 7, through X.
test('journaled before its first request, delivered unattended through its own route at its first prepared generation', async () => {
  const first = await renderer()
  let answered = false
  const pending = first.store.sendJournaledCorrection(
    { id: 'D', mode: 'steer', text: 'D words', sessionId: 'x-session' },
    X,
    () => new Promise<boolean>(() => {}) // its readiness wait never ends in this renderer
  )
  void pending.then(() => (answered = true))
  expect(first.store.unresolvedCorrections()).toEqual([
    expect.objectContaining({ id: 'D', owner: X, sessionId: 'x-session', generation: null })
  ])
  expect(answered).toBe(false)

  vi.resetModules()
  const second = await renderer()
  const after = new second.protocol.CanonicalDesktopProtocol()
  after.result('session.resume', { session_id: 'x-session', profile: 'work' }, { session_id: 'x-session', revision: 1, execution_generation: 7 })
  const wire: Array<Record<string, unknown>> = []
  let keptAtSend: unknown
  wireTo(second, after, wire, async frame => {
    if (frame.text === 'D words') {
      keptAtSend = second.store.unresolvedCorrections()[0]?.generation
      throw new Error('reply lost')
    }

    return { status: 'queued' }
  })
  second.store.restoreUnresolvedCorrections()
  await second.gateway(X, undefined as never, 'session.steer', { session_id: 'x-session', text: 'B words', correction_id: 'B' })
  await vi.waitFor(() => expect(wire.filter(frame => frame.text === 'D words')).not.toHaveLength(0), { timeout: 5_000 })
  expect(wire[0]).toMatchObject({ text: 'B words' })
  expect(wire[1]).toEqual({ owner: X, method: 'session.steer', session_id: 'x-session', text: 'D words', execution_generation: 7 })
  expect(keptAtSend).toBe(7) // kept when the pin was made, before the request left
  expect(second.store.unresolvedCorrections()).toEqual([expect.objectContaining({ id: 'D', owner: X, generation: 7 })])

  vi.resetModules()
  const third = await renderer()
  const later = new third.protocol.CanonicalDesktopProtocol()
  later.result('session.resume', { session_id: 'x-session', profile: 'work' }, { session_id: 'x-session', revision: 2, execution_generation: 8 })
  const wire3: unknown[] = []
  wireTo(third, later, wire3)
  third.store.restoreUnresolvedCorrections()
  await vi.waitFor(() => expect(wire3).toHaveLength(1), { timeout: 5_000 })
  expect(wire3).toEqual([{ owner: X, method: 'session.steer', session_id: 'x-session', text: 'D words', execution_generation: 7 }])
  await vi.waitFor(() => expect(third.store.unresolvedCorrections()).toEqual([]))
})
