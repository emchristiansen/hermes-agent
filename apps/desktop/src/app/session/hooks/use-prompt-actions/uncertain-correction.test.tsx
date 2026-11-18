import { JsonRpcGatewayError } from '@hermes/shared'
import { useStore } from '@nanostores/react'
import { act, cleanup, render, waitFor } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import { CanonicalDesktopProtocol } from '@/api/canonical-protocol'
import { useComposerSubmit } from '@/app/chat/composer/hooks/use-composer-submit'
import { ComposerScopeProvider, MAIN_COMPOSER_SCOPE } from '@/app/chat/composer/scope'
import { createSessionRpcDispatcher } from '@/app/contrib/session-rpc-dispatcher'
import { sessionRoute } from '@/app/routes'
import { textPart } from '@/lib/chat-messages'
import { $queuedPromptsBySession, serverOwnsComposerQueue } from '@/store/composer-queue'
import { requestGatewayForAgent } from '@/store/gateway'
import { $notifications } from '@/store/notifications'
import {
  $activeSessionId,
  $connection,
  $selectedStoredSessionId,
  setActiveSessionId,
  setActiveSessionStoredIdRotation,
  setAwaitingResponse,
  setBusy,
  setMessages,
  setSelectedStoredSessionId,
  setSessions
} from '@/store/session'
import { $sessionStates, clearAllSessionStates } from '@/store/session-states'
import type { SessionInfo } from '@/types/hermes'

import { useSessionStateCache } from '../use-session-state-cache'

import { clearSingleFlightSessionResumeState } from './single-flight-resume'

import { usePromptActions } from '.'

// NC12-CLIENT-01: a busy desktop correction whose ACK is lost (or that the owner could not keep)
// stays a correction. The normal composer, prompt actions, session cache, ownership router and RPC
// dispatcher are real; only the external gateway edge is substituted. The owner's replies are the
// shapes it returns: the kept ACK of a card session's ended claim (gateway/session_managed_worker
// keep_closed_card_correction), `correction_not_kept`, and the undelivered `stale_generation`.
vi.mock('@/store/gateway', async original => ({
  ...(await original<Record<string, unknown>>()),
  requestGatewayForAgent: vi.fn(),
  requestGatewayForProfile: vi.fn(),
  retainGatewayForSessionTurn: vi.fn(async () => () => undefined)
}))

const KEPT = {
  text: 'TASK_BOUND_RETRY',
  execution_generation: 4,
  authority_epoch: 1,
  status: 'rejected',
  delivered: false,
  kept: 'task_comment'
}

const busyRef = { current: false }
// The composer's own busy belief and busy input mode (a test may render an idle composer).
const composerState: { busy: boolean; mode: 'interrupt' | 'steer' } = { busy: true, mode: 'steer' }
const loadIntoComposer = vi.fn()
let routedStoredId: string | null = 'stored-B'
let handle: { actions: ReturnType<typeof usePromptActions>; cache: ReturnType<typeof useSessionStateCache> }
let composer: ReturnType<typeof useComposerSubmit>
const draft = { current: '' }
const submits = vi.fn(async (text: string, options: never) => handle.actions.submitText(text, options))

function Harness() {
  const activeSessionId = useStore($activeSessionId)
  const selectedStoredSessionId = useStore($selectedStoredSessionId)

  const cache = useSessionStateCache({
    activeSessionId,
    selectedStoredSessionId,
    busyRef,
    setAwaitingResponse,
    setBusy,
    setMessages
  })

  const requestGateway = createSessionRpcDispatcher({
    ...cache,
    ambientRequest: async () => {
      throw new Error('unexpected ambient request')
    }
  })

  const actions = usePromptActions({
    activeSessionId,
    ...cache,
    busyRef,
    branchCurrentSession: async () => false,
    createBackendSessionForSend: async () => {
      throw new Error('unexpected create')
    },
    getRoutedStoredSessionId: () => routedStoredId,
    getRouteToken: () => `${routedStoredId ? sessionRoute(routedStoredId) : '/'}::`,
    handleSkinCommand: () => '',
    openMemoryGraph: () => undefined,
    refreshSessions: async () => undefined,
    requestGateway,
    resumeStoredSession: async () => {
      throw new Error('unexpected foreground resume')
    },
    startFreshSessionDraft: () => undefined,
    sttEnabled: false
  })

  handle = { actions, cache }

  return null
}

function Composer() {
  composer = useComposerSubmit({
    activeQueueSessionKey: 'stored-B',
    activeQueueSessionKeyRef: { current: 'stored-B' },
    attachments: [],
    busy: composerState.busy,
    busyInputMode: composerState.mode,
    clearDraft: () => {
      draft.current = ''
    },
    disabled: false,
    draftScopeRef: { current: 'stored-B' },
    draftRef: draft,
    drainNextQueued: async () => false,
    editorRef: { current: null },
    exitQueuedEdit: () => false,
    focusInput: () => {},
    inputDisabled: false,
    loadIntoComposer,
    onCancel: () => {},
    onSteer: (text: string, mode?: 'interrupt' | 'steer') => handle.actions.redirectPrompt(text, mode),
    onSteerHidden: async () => true,
    onSubmit: submits,
    queueCurrentDraft: () => true,
    queueEdit: null,
    queuedPrompts: [],
    sessionId: 'rt-B',
    setComposerText: () => {},
    stashAt: () => {}
  } as never)

  return null
}

function seed() {
  routedStoredId = 'stored-B'
  busyRef.current = false
  setSessions(
    ['A', 'B'].map(id => ({
      id: `stored-${id}`,
      connection_id: `connection-${id}`,
      profile: 'default',
      source: 'desktop',
      message_count: 1
    })) as SessionInfo[]
  )
  setSelectedStoredSessionId('stored-B')
  setActiveSessionId('rt-B')
  render(<Harness />)
  act(() => {
    for (const id of ['A', 'B']) {
      handle.cache.updateSessionState(
        `rt-${id}`,
        state => ({
          ...state,
          messages: [{ id: `history-${id}`, role: 'assistant', parts: [textPart(`history ${id}`)] }]
        }),
        `stored-${id}`
      )
    }
  })
  $connection.set({ mode: 'local', connectionId: 'connection-B', wsUrl: 'ws://localhost/api/ws?native_dial=unminted' } as never)
  $queuedPromptsBySession.set({})
  $notifications.set([])
  render(
    <ComposerScopeProvider value={MAIN_COMPOSER_SCOPE}>
      <Composer />
    </ComposerScopeProvider>
  )
}

// [method, params] of every call through the substituted gateway edge.
const calls = () =>
  vi.mocked(requestGatewayForAgent).mock.calls.map(call => [call[2], call[3]] as [string, Record<string, unknown>])

const promptSubmits = () => calls().filter(([method]) => method === 'prompt.submit')

const retained = () => $notifications.get().filter(note => note.detail === 'TASK_BOUND_RETRY')

async function send(text: string) {
  draft.current = text
  act(() => composer.submitDraft())
  await act(async () => {
    await new Promise(resolve => setTimeout(resolve, 20))
  })
}

async function sendAgain() {
  const note = retained().at(-1)
  expect(note?.action).toBeDefined()
  await act(async () => {
    note!.action!.onClick()
    await new Promise(resolve => setTimeout(resolve, 20))
  })
}

// The owner's replies, in order; anything else is unexpected.
function owner(...replies: Array<(params: Record<string, unknown>) => unknown>) {
  vi.mocked(requestGatewayForAgent).mockImplementation(async (_connection, _profile, method, params) => {
    if (method === 'prompt.submit') {
      return { status: 'queued', admission_id: 'unbound-admission', accepted: true } as never
    }

    if (method !== 'session.steer') {
      throw new Error(`unexpected ${method}`)
    }

    const reply = replies.shift()

    if (!reply) {
      throw new Error('no reply scripted')
    }

    return reply(params as Record<string, unknown>) as never
  })
}

const lostAck = () => {
  throw new Error('task comment kept but ACK lost')
}

const notKept = () => {
  throw new JsonRpcGatewayError('correction_not_kept', { code: 4001, data: { reason: 'correction_not_kept' } })
}

afterEach(() => {
  composerState.busy = true
  composerState.mode = 'steer'
  cleanup()
  clearAllSessionStates()
  clearSingleFlightSessionResumeState()
  setActiveSessionId(null)
  setActiveSessionStoredIdRotation(null)
  setSelectedStoredSessionId(null)
  setSessions([])
  setBusy(false)
  setAwaitingResponse(false)
  setMessages([])
  $connection.set(null)
  $queuedPromptsBySession.set({})
  $notifications.set([])
  window.localStorage.clear()
  vi.clearAllMocks()
})

it('settles a correction the owner kept on its task without any prompt', async () => {
  owner(() => KEPT)
  seed()
  expect(serverOwnsComposerQueue('rt-B')).toBe(true)
  await send('TASK_BOUND_RETRY')

  expect(calls()).toEqual([['session.steer', { session_id: 'rt-B', text: 'TASK_BOUND_RETRY', correction_id: expect.any(String) }]])
  expect(retained()).toEqual([])
})

it.each([
  ['lost-ack', lostAck],
  ['not-kept', notKept]
])(
  'keeps an uncertain correction (%s) a correction: later input still goes, the explicit retry reaches its own session',
  async (_case, failure) => {
    // A's ACK is lost; B is still sent; the claim A named ends; the user sends A again from the
    // retained notice after switching to another session: the owner keeps it on the card's task.
    owner(failure, () => ({ status: 'queued', text: 'NEXT_B' }), () => KEPT)
    seed()
    await send('TASK_BOUND_RETRY')

    expect(promptSubmits()).toEqual([])
    expect(retained()).toHaveLength(1)
    expect(retained()[0]).toMatchObject({ kind: 'warning', persistent: true })

    await send('NEXT_B')

    expect(calls().map(([, params]) => params.text)).toEqual(['TASK_BOUND_RETRY', 'NEXT_B'])
    expect(retained()).toHaveLength(1)

    routedStoredId = 'stored-A'
    act(() => {
      setSelectedStoredSessionId('stored-A')
      setActiveSessionId('rt-A')
    })
    await sendAgain()

    expect(calls()).toEqual([
      ['session.steer', { session_id: 'rt-B', text: 'TASK_BOUND_RETRY', correction_id: expect.any(String) }],
      ['session.steer', { session_id: 'rt-B', text: 'NEXT_B', correction_id: expect.any(String) }],
      ['session.steer', { session_id: 'rt-B', text: 'TASK_BOUND_RETRY', correction_id: expect.any(String) }]
    ])
    // The retry is the same correction (its generation pin travels with this id); B is its own.
    const [a, b, again] = calls().map(([, params]) => params.correction_id)
    expect(again).toBe(a)
    expect(b).not.toBe(a)
    expect(promptSubmits()).toEqual([])
  }
)

it('keeps a correction whose retry is uncertain again, and lets it be sent once more', async () => {
  owner(lostAck, lostAck, () => KEPT)
  seed()
  await send('TASK_BOUND_RETRY')
  await sendAgain()

  // Still uncertain: the same correction keeps its one notice and stays in delivery (review22815 UI-R4).
  expect(retained()).toHaveLength(1)
  await sendAgain()

  expect(calls().map(([method]) => method)).toEqual(['session.steer', 'session.steer', 'session.steer'])
  expect(promptSubmits()).toEqual([])
})

// Control: the owner confirmed the retried correction undelivered (a session that is not a Kanban
// card, its generation over): the words become the next prompt, as for a first refusal (#141644).
it('re-admits a retried correction the owner refused undelivered as the next prompt', async () => {
  const undelivered = () => {
    throw new JsonRpcGatewayError('stale_generation', { code: 4001, data: { reason: 'stale_generation' } })
  }

  owner(lostAck, undelivered)
  seed()
  await send('TASK_BOUND_RETRY')
  await sendAgain()
  await waitFor(() => expect(promptSubmits()).toHaveLength(1))

  expect(promptSubmits()[0]![1]).toMatchObject({ session_id: 'rt-B', text: 'TASK_BOUND_RETRY', queued: true })
})

// A reply that is neither accepted nor an owner-confirmed undelivered refusal proves nothing about a
// kept copy: a plain 'rejected' can follow delivery, and a missing or unrecognised reply says nothing.
it.each([
  ['missing', () => undefined],
  ['empty', () => ({})],
  ['plain rejected', () => ({ status: 'rejected', text: 'TASK_BOUND_RETRY' })],
  ['unrecognised', () => ({ status: 'mystery', text: 'TASK_BOUND_RETRY' })]
])('keeps a correction whose reply is %s a correction, never a prompt', async (_case, reply) => {
  owner(reply, () => KEPT)
  seed()
  await send('TASK_BOUND_RETRY')

  expect(promptSubmits()).toEqual([])
  expect(retained()).toHaveLength(1)
  await sendAgain()

  expect(calls().map(([method]) => method)).toEqual(['session.steer', 'session.steer'])
  expect(promptSubmits()).toEqual([])
})

// Control: the owner's confirmed undelivered refusal (`delivered: false`, no live agent took it)
// is the one reply that becomes the next prompt.
it('re-admits a correction the owner confirmed undelivered as the next prompt', async () => {
  owner(() => ({ status: 'rejected', delivered: false, text: 'TASK_BOUND_RETRY', execution_generation: 4 }))
  seed()
  await send('TASK_BOUND_RETRY')
  await waitFor(() => expect(promptSubmits()).toHaveLength(1))

  expect(retained()).toEqual([])
})

// NC13: the gateway edge below runs the real canonical decorator (`CanonicalDesktopProtocol`:
// prepare / event / result) and answers as the owner's `session_busy_controls.correct` does: the
// exact running generation takes the correction; any other is closed, so a Kanban card session keeps
// it on its task (the kept ACK) and any other session refuses it `stale_generation` (undelivered).
const STALE = () => new JsonRpcGatewayError('stale_generation', { code: 4009, data: { reason: 'stale_generation' } })

function fencedOwner(state: { running: number | null; card: boolean; lose?: number; gone?: boolean }) {
  const protocol = new CanonicalDesktopProtocol()
  protocol.result('session.resume', { session_id: 'rt-B' }, { session_id: 'rt-B', revision: 1, execution_generation: 4 })
  const wire: Array<[string, Record<string, unknown>]> = []

  vi.mocked(requestGatewayForAgent).mockImplementation(async (_connection, _profile, method, params) => {
    const prepared = protocol.prepare(method, params as Record<string, unknown>)
    wire.push([protocol.wire(method, prepared), prepared])

    if (method === 'prompt.submit') {
      return { status: 'queued', admission_id: 'unbound-admission', accepted: true } as never
    }

    // The stored chat resumed on a NEW runtime whose own execution reuses the number 4.
    if (method === 'session.resume') {
      return protocol.result(method, prepared, { session_id: 'rt-B2', revision: 1, execution_generation: 4 }) as never
    }

    if (method !== 'session.steer' && method !== 'session.redirect') {
      throw new Error(`unexpected ${method}`)
    }

    if (state.gone && prepared.session_id === 'rt-B') {
      throw new Error('session not found')
    }

    if (state.lose) {
      state.lose -= 1
      throw new Error('ACK lost')
    }

    const generation = prepared.execution_generation

    if (generation === state.running) {
      return { status: method === 'session.steer' ? 'queued' : 'redirected', text: prepared.text, execution_generation: generation } as never
    }

    if (state.card) {
      return { ...KEPT, text: prepared.text, execution_generation: generation } as never
    }

    throw STALE()
  })

  // A later execution of rt-B starts: the owner's state event moves the decorator's cache.
  const advance = (generation: number) => {
    state.running = generation
    protocol.event({ type: 'session.info', session_id: 'rt-B', payload: { revision: generation, execution_generation: generation, running: true } })
  }

  return { advance, wire }
}

const steers = (wire: Array<[string, Record<string, unknown>]>) =>
  wire.filter(([method]) => method !== 'prompt.submit').map(([method, params]) => [method, params.text, params.execution_generation])

const switchToA = () => {
  routedStoredId = 'stored-A'
  act(() => {
    setSelectedStoredSessionId('stored-A')
    setActiveSessionId('rt-A')
  })
}

it.each([
  ['steer', 'session.steer'],
  ['interrupt', 'session.redirect']
] as const)(
  'an explicit retry (%s) keeps its first generation through a later execution and a focus switch; later B is sent at its own',
  async (mode, method) => {
    composerState.mode = mode
    const { advance, wire } = fencedOwner({ running: 4, card: true, lose: 1 })
    seed()
    await send('TASK_BOUND_RETRY')
    expect(retained()).toHaveLength(1)

    advance(5)
    await send('NEXT_B')
    switchToA()
    await sendAgain()

    // A again names generation 4, never 5: the owner keeps it for the card task generation 4 ran.
    expect(steers(wire)).toEqual([
      [method, 'TASK_BOUND_RETRY', 4],
      [method, 'NEXT_B', 5],
      [method, 'TASK_BOUND_RETRY', 4]
    ])
    expect(wire.every(([, params]) => params.session_id === 'rt-B' && !('correction_id' in params))).toBe(true)
    expect(promptSubmits()).toEqual([])
    // Its outcome is known (kept for the card task): its notice is gone (review22815 UI-R4).
    expect(retained()).toEqual([])
  }
)

it('an explicit retry another session refuses stale is queued for its own session, not the focused one', async () => {
  const { advance, wire } = fencedOwner({ running: 4, card: false, lose: 1 })
  seed()
  await send('TASK_BOUND_RETRY')
  advance(5)
  switchToA()
  await sendAgain()

  expect(steers(wire)).toEqual([
    ['session.steer', 'TASK_BOUND_RETRY', 4],
    ['session.steer', 'TASK_BOUND_RETRY', 4]
  ])
  // Confirmed undelivered (#141644): the words become rt-B's next prompt, as before NC13.
  expect(promptSubmits()).toHaveLength(1)
  expect(promptSubmits()[0]![1]).toMatchObject({ session_id: 'rt-B', text: 'TASK_BOUND_RETRY', queued: true })
  // Its outcome is known (refused undelivered): its notice is gone (review22815 UI-R4).
  expect(retained()).toEqual([])
})

function endPane() {
  act(() =>
    handle.cache.updateSessionState(
      'rt-B',
      state => ({ ...state, busy: false, awaitingResponse: false, turnLive: false, streamId: null }),
      'stored-B'
    )
  )
  // #105176: the composer still believes the turn busy one effect tick after its pane settled.
  busyRef.current = true
}

const bubbles = (text: string) =>
  ($sessionStates.get()['rt-B']?.messages ?? []).filter(
    message => message.role === 'user' && JSON.stringify(message.parts).includes(text)
  )

it.each([
  ['steer', 'session.steer'],
  ['interrupt', 'session.redirect']
] as const)('an ended card pane keeps a %s correction task-bound: no ordinary prompt and no bubble', async (mode, method) => {
  composerState.mode = mode
  const { wire } = fencedOwner({ running: null, card: true })
  seed()
  endPane()
  await send('TASK_BOUND_RETRY')

  expect(steers(wire)).toEqual([[method, 'TASK_BOUND_RETRY', 4]])
  expect(promptSubmits()).toEqual([])
  expect(bubbles('TASK_BOUND_RETRY')).toEqual([])
  expect(retained()).toEqual([])
})

it('an ended ordinary pane still queues the words as its next prompt once the owner refuses them', async () => {
  const { wire } = fencedOwner({ running: null, card: false })
  seed()
  endPane()
  await send('NOT_A_CARD')

  expect(steers(wire)).toEqual([['session.steer', 'NOT_A_CARD', 4]])
  expect(promptSubmits()).toHaveLength(1)
  expect(promptSubmits()[0]![1]).toMatchObject({ session_id: 'rt-B', text: 'NOT_A_CARD', queued: true })
  expect(bubbles('NOT_A_CARD')).toEqual([])
})

it('an ended pane leaves later input free, and a legitimate idle prompt is an ordinary prompt', async () => {
  const { wire } = fencedOwner({ running: null, card: true })
  seed()
  endPane()
  await send('TASK_BOUND_RETRY')
  await send('NEXT_B')
  expect(steers(wire).map(([, text]) => text)).toEqual(['TASK_BOUND_RETRY', 'NEXT_B'])

  cleanup()
  composerState.busy = false
  seed()
  await send('IDLE_PROMPT')
  expect(promptSubmits().map(([, params]) => params.text)).toEqual(['IDLE_PROMPT'])
})

it('a canonical correction whose target navigation cannot establish returns to the composer, never a prompt', async () => {
  fencedOwner({ running: 4, card: true })
  seed()
  // The route has left the selection: the correction has no provable target (#105176).
  routedStoredId = 'stored-A'
  await send('TASK_BOUND_RETRY')

  expect(calls()).toEqual([])
  // Never handed to the ordinary submit path (whatever that path would then do with it).
  expect(submits).not.toHaveBeenCalled()
  expect(loadIntoComposer).toHaveBeenCalledWith('TASK_BOUND_RETRY', [])
  expect(retained()).toEqual([])
})

it('a retry the resolver would send to a resumed runtime is sent nowhere: the words return to the composer', async () => {
  // A's first send meets "session not found"; the stored chat resumes on rt-B2, whose own execution
  // also numbers 4. The pinned destination is rt-B's: rt-B2 never receives A, and A is no prompt.
  const { wire } = fencedOwner({ running: 4, card: true, gone: true })
  seed()
  await send('TASK_BOUND_RETRY')

  expect(wire.map(([method, params]) => [method, params.session_id])).toEqual([
    ['session.steer', 'rt-B'],
    ['session.resume', 'stored-B']
  ])
  expect(promptSubmits()).toEqual([])
  expect(retained()).toEqual([])
  expect(loadIntoComposer).toHaveBeenCalledWith('TASK_BOUND_RETRY', [])
})
