import { AssistantRuntimeProvider, type ThreadMessageLike, useExternalStoreRuntime } from '@assistant-ui/react'
import { act, cleanup, fireEvent, render, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n/context'
import { UncertainCorrection, UnsentCorrection } from '@/lib/uncertain-correction'
import { clearSessionDraft, mainComposerScope, takeSessionDraft } from '@/store/composer'
import { $queuedPromptsBySession, getQueuedPrompts } from '@/store/composer-queue'
import { $notifications } from '@/store/notifications'
import { $connection, $gatewayState, $sessionsLoading } from '@/store/session'
import { $sessionStates } from '@/store/session-states'

import { requestComposerSubmit } from './focus'
import { ChatBar } from './index'
import { ComposerSurfaceProvider } from './scope'
import type { ChatBarProps, ChatBarState } from './types'

// A canonical correction through every composer entry that steers, on the full mounted ChatBar: its
// input handler (which mirrors the editor into the draft on the next animation frame), draft loader,
// persistence, notices, focus bus and queue are real. Only onSteer / onSubmit and the configuration
// socket are substituted.
vi.mock('@/store/session-request-router', async importOriginal => ({
  ...(await importOriginal<object>()),
  requestForSessionProfile: async (
    _owner: unknown,
    request: (method: string, params: unknown) => unknown,
    method: string,
    params: unknown
  ) => request(method, params)
}))

const gateway = { request: async (method: string) => (method === 'config.get' ? { value: 'steer' } : {}) } as never

const state: ChatBarState = {
  model: { canSwitch: false, model: '', provider: '' },
  tools: { enabled: false, label: '' },
  voice: { enabled: false, active: false }
}

function Harness({ busy = true, onSteer, onSubmit }: Pick<ChatBarProps, 'onSteer' | 'onSubmit'> & { busy?: boolean }) {
  const runtime = useExternalStoreRuntime({
    convertMessage: (message: ThreadMessageLike) => message,
    isRunning: busy,
    messages: [] as ThreadMessageLike[],
    onNew: async () => {}
  })

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <MemoryRouter>
        <I18nProvider configClient={null} initialLocale="en">
          <ComposerSurfaceProvider value="surface">
            <div data-composer-surface-id="surface" data-composer-target="main">
              <ChatBar
                busy={busy}
                disabled={false}
                gateway={gateway}
                onCancel={() => {}}
                onSteer={onSteer}
                onSubmit={onSubmit}
                queueSessionKey="stored-B"
                sessionId="rt-B"
                state={state}
              />
            </div>
          </ComposerSurfaceProvider>
        </I18nProvider>
      </MemoryRouter>
    </AssistantRuntimeProvider>
  )
}

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
  mainComposerScope.clear()
  clearSessionDraft('stored-B')
  $connection.set(null)
  $sessionStates.set({})
  $queuedPromptsBySession.set({})
  $notifications.set([])
  localStorage.clear()
})

function canonicalOwner() {
  $sessionsLoading.set(false)
  $gatewayState.set('open')
  $connection.set({ mode: 'local', wsUrl: 'ws://localhost/api/ws?native_dial=unminted' } as never)
}

const notice = (detail: string, action: boolean) =>
  $notifications.get().find(n => n.detail === detail && Boolean(n.action) === action)

// NC16-CLIENT-01: words typed while a correction is out, but not yet mirrored into the draft, are input.
it.each(['refused retry', 'first failure'] as const)(
  'a correction that was not sent keeps newer, not yet mirrored input (%s)',
  async which => {
    canonicalOwner()
    let refuse!: (error: Error) => void

    const pending = () =>
      new Promise<boolean>((_resolve, reject) => {
        refuse = reject
      })

    const onSteer = vi.fn(
      which === 'refused retry'
        ? async () => {
            throw new UncertainCorrection(new Error('lost ACK'), pending)
          }
        : pending
    )

    const onSubmit = vi.fn(async () => true)
    const view = render(<Harness onSteer={onSteer} onSubmit={onSubmit} />)
    await act(async () => {})
    const editor = view.container.querySelector<HTMLDivElement>('[data-slot="composer-rich-input"]')!
    const frame = vi.spyOn(window, 'requestAnimationFrame').mockImplementation(() => 1)
    await act(async () => {
      editor.textContent = 'ORIGINAL_CORRECTION'
      fireEvent.input(editor)
      fireEvent.submit(view.container.querySelector('form')!)
    })
    await waitFor(() => expect(onSteer).toHaveBeenCalledTimes(1))

    if (which === 'refused retry') {
      await act(async () => notice('ORIGINAL_CORRECTION', true)!.action!.onClick())
    }

    expect(editor.textContent?.trim()).toBe('')
    await act(async () => {
      editor.textContent = 'NEWER_INPUT'
      fireEvent.input(editor)
    })
    expect(frame).toHaveBeenCalled() // the mirror into the draft is still pending
    await act(async () => refuse(new UnsentCorrection('its session changed since it was first sent')))
    act(() => window.dispatchEvent(new Event('pagehide')))
    expect(editor.textContent?.trim()).toBe('NEWER_INPUT')
    expect(takeSessionDraft('stored-B').text.trim()).toBe('NEWER_INPUT')
    expect(notice('ORIGINAL_CORRECTION', false)?.persistent).toBe(true)
    expect(onSubmit).not.toHaveBeenCalled()
  }
)

// NC16-CLIENT-03: an external request (the review pane's ship bar, the plugin SDK) that steers a busy
// turn settles like a typed correction: never an ordinary queued prompt unless the owner confirmed it
// undelivered.
it('an external correction whose reply was lost stays the same correction, delivered again on its own, A and B both', async () => {
  canonicalOwner()
  const retries = { A: vi.fn(async () => true), B: vi.fn(async () => true) }

  const onSteer = vi.fn(async (text: string) => {
    throw new UncertainCorrection(new Error('lost ACK'), retries[text === 'EXTERNAL_A' ? 'A' : 'B'])
  })

  const onSubmit = vi.fn(async () => true)
  const view = render(<Harness onSteer={onSteer} onSubmit={onSubmit} />)
  await act(async () => {})
  await act(async () => void requestComposerSubmit('EXTERNAL_A', { target: 'main' }))
  await act(async () => void requestComposerSubmit('EXTERNAL_B', { target: 'main' }))
  await waitFor(() => expect(onSteer).toHaveBeenCalledTimes(2))
  expect(onSteer.mock.calls.map(([text]) => text)).toEqual(['EXTERNAL_A', 'EXTERNAL_B'])
  expect(notice('EXTERNAL_A', true)).toBeDefined()
  // Nobody clicks: each is sent again on its own (review22815 UI-R4), neither waiting for the other.
  await waitFor(() => expect([retries.A.mock.calls.length, retries.B.mock.calls.length]).toEqual([1, 1]), { timeout: 5_000 })
  await waitFor(() => expect(notice('EXTERNAL_A', true)).toBeUndefined())
  view.rerender(<Harness busy={false} onSteer={onSteer} onSubmit={onSubmit} />)
  await act(async () => {
    await new Promise(resolve => setTimeout(resolve, 20))
  })
  expect(getQueuedPrompts('stored-B')).toEqual([])
  expect(onSubmit).not.toHaveBeenCalled()
})

it.each([
  { outcome: 'not sent', queued: false },
  { outcome: 'refused undelivered', queued: true }
] as const)('an external correction $outcome: queued as a prompt only when the owner confirmed it undelivered', async ({ queued }) => {
  canonicalOwner()

  const onSteer = vi.fn(async () => {
    if (!queued) {
      throw new UnsentCorrection()
    }

    return false
  })

  render(<Harness onSteer={onSteer} onSubmit={vi.fn(async () => true)} />)
  await act(async () => {})
  await act(async () => void requestComposerSubmit('EXTERNAL_CORRECTION', { target: 'main' }))
  await waitFor(() => expect(onSteer).toHaveBeenCalledTimes(1))
  await act(async () => {})
  expect(getQueuedPrompts('stored-B').map(entry => entry.text)).toEqual(queued ? ['EXTERNAL_CORRECTION'] : [])
  expect(Boolean(notice('EXTERNAL_CORRECTION', false))).toBe(!queued)
})
