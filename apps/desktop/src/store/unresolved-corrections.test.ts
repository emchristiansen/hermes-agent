import { beforeEach, expect, test, vi } from 'vitest'

// UI-02 (review t_42a8d5e2): a correction whose outcome is unknown survives a renderer restart. A's first
// send was prepared against generation 4 and failed before delivery; the renderer restarts (fresh modules:
// no pins, no notices, no retry closures; storage remains); independent B sends at once against the newer
// generation 5; A is still offered and its Send again reaches A's own session with generation 4, never a
// prompt and never the latest generation. Real CanonicalDesktopProtocol; the gateway is the protocol's
// own prepare, recorded as the wire.
vi.mock('@/i18n', () => ({ translateNow: (key: string) => key }))
vi.mock('@/store/notifications', () => ({ notify: vi.fn() }))
vi.mock('@/store/gateway', () => ({ requestGatewayForProfile: vi.fn() }))

beforeEach(() => {
  window.localStorage.clear()
  vi.resetModules()
})

async function renderer() {
  const protocol = await import('@/api/canonical-protocol')
  const store = await import('./unresolved-corrections')
  const { notify } = await import('@/store/notifications')
  const { requestGatewayForProfile } = await import('@/store/gateway')
  vi.mocked(notify).mockClear()

  return { protocol, store, notify: vi.mocked(notify), gateway: vi.mocked(requestGatewayForProfile) }
}

test('A fails before delivery, the renderer restarts, B sends at once, A stays retryable to its own destination', async () => {
  const first = await renderer()
  const before = new first.protocol.CanonicalDesktopProtocol()
  before.result('session.resume', { session_id: 's' }, { session_id: 's', revision: 1, execution_generation: 4 })
  before.prepare('session.steer', { session_id: 's', text: 'A words', correction_id: 'A' })
  expect(first.store.keepUnresolvedCorrection({ id: 'A', mode: 'steer', text: 'A words' }, 'A words')).toMatchObject({
    sessionId: 's',
    profile: 'default',
    generation: 4
  })

  vi.resetModules()
  const second = await renderer()
  expect(second.protocol.correctionPin('A')).toBeUndefined() // the restart lost every in-memory pin
  const after = new second.protocol.CanonicalDesktopProtocol()
  after.result('session.resume', { session_id: 's' }, { session_id: 's', revision: 2, execution_generation: 5 })
  const wire: Array<Record<string, unknown>> = []
  second.gateway.mockImplementation((async (profile: string, method: string, params: Record<string, unknown>) => {
    wire.push({ profile, method, ...after.prepare(method, params) })

    return { status: 'queued' }
  }) as never)

  second.store.restoreUnresolvedCorrections()
  await second.gateway('default', 'session.steer', { session_id: 's', text: 'B words', correction_id: 'B' })

  const [offer] = second.notify.mock.calls.map(([input]) => input)
  expect(second.notify).toHaveBeenCalledTimes(1)
  expect(offer).toMatchObject({ id: 'unresolved-correction:A', persistent: true, detail: 'A words' })
  offer.action!.onClick()
  await vi.waitFor(() => expect(wire).toHaveLength(2))

  expect(wire).toEqual([
    { profile: 'default', method: 'session.steer', session_id: 's', text: 'B words', execution_generation: 5 },
    { profile: 'default', method: 'session.steer', session_id: 's', text: 'A words', execution_generation: 4 }
  ])
  await vi.waitFor(() => expect(second.store.unresolvedCorrections()).toEqual([]))
})

test('an unconfirmed restored retry keeps the correction; an undelivered refusal leaves its words in a notice', async () => {
  const first = await renderer()
  const protocol = new first.protocol.CanonicalDesktopProtocol()
  protocol.result('session.resume', { session_id: 's' }, { session_id: 's', revision: 1, execution_generation: 4 })
  protocol.prepare('session.redirect', { session_id: 's', text: 'A words', correction_id: 'A' })
  const entry = first.store.keepUnresolvedCorrection({ id: 'A', mode: 'interrupt', text: 'A words' }, 'A words')!

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

test('a correction without a pinned destination is not kept: it would have no generation to present', async () => {
  const { store } = await renderer()
  expect(store.keepUnresolvedCorrection({ id: 'never-prepared', mode: 'steer', text: 'x' }, 'x')).toBeUndefined()
  expect(store.unresolvedCorrections()).toEqual([])
})
