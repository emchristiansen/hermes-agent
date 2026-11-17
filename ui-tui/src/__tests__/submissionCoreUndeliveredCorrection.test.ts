import { mkdtempSync, rmSync } from 'node:fs'
import { join } from 'node:path'

import { JsonRpcGatewayError } from '@hermes/shared/json-rpc-channel'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { submitPrompt, type SubmitPromptDeps } from '../app/submissionCore.js'
import { captureDestination } from '../app/submissionDestination.js'
import { patchUiState, resetUiState } from '../app/uiStore.js'
import type { GatewayClient } from '../gatewayClient.js'
import type { QueueItem } from '../hooks/useQueue.js'
import { loadPendingInputs } from '../lib/pendingInputs.js'

// #141511: a busy correction the owner refused before any agent saw it (the turn the TUI believed
// busy had ended) stayed in the journal as a control naming that dead generation, so every retry
// was refused again and the words were never delivered. It must be re-admitted as a prompt.

let home: string

beforeEach(() => {
  home = mkdtempSync(join(process.env.TMPDIR ?? '.', 'ink-undelivered-'))
  vi.stubEnv('HERMES_HOME', home)
  resetUiState()
  patchUiState({ sid: 'owner', busy: true, status: 'running…' })
})

afterEach(() => {
  resetUiState()
  vi.unstubAllEnvs()
  rmSync(home, { recursive: true, force: true })
})

function deps(gw: GatewayClient): SubmitPromptDeps {
  return { appendMessage: vi.fn(), enqueue: vi.fn(), expand: t => t, gw, setLastUserMsg: vi.fn(), sys: vi.fn() }
}

function correction(settle: (accepted: boolean) => void): QueueItem {
  return {
    controlMethod: 'session.redirect',
    destination: captureDestination(),
    display: 'next thought',
    executionGeneration: 4,
    settle,
    submissionId: 'sub-1',
    text: 'next thought'
  }
}

const refusals: Array<[string, () => Promise<unknown>]> = [
  ['stale_generation', async () => {
    throw new JsonRpcGatewayError('stale_generation', { code: 4001, data: { reason: 'stale_generation' } })
  }],
  ['execution_not_ready', async () => {
    throw new JsonRpcGatewayError('execution_not_ready', { code: 4001, data: { reason: 'execution_not_ready' } })
  }],
  ['rejected undelivered', async () => ({ status: 'rejected', delivered: false, text: 'next thought', execution_generation: 4 })]
]

it.each(refusals)('re-admits a correction refused undelivered (%s) as a prompt with the same submission_id', async (_name, refuse) => {
  const calls: Array<{ method: string; params: Record<string, unknown> }> = []
  const settle = vi.fn()

  const request = vi.fn(async (method: string, params: Record<string, unknown>) => {
    calls.push({ method, params })

    return method === 'prompt.submit' ? { status: 'queued' } : refuse()
  })

  const item = correction(settle)
  submitPrompt('next thought', deps({ isCanonical: true, request } as unknown as GatewayClient), true, undefined, {
    behindTurn: true,
    queueItem: item,
    skipDetectDrop: true
  })

  await vi.waitFor(() => expect(calls.map(c => c.method)).toEqual(['session.redirect', 'prompt.submit']))
  expect(calls[1]!.params).toMatchObject({ session_id: 'owner', text: 'next thought', submission_id: 'sub-1', queued: true })
  expect(calls[1]!.params).not.toHaveProperty('execution_generation')
  // The journal no longer holds a control a later retry would replay against the dead generation.
  expect(loadPendingInputs(captureDestination()).find(row => row.submissionId === 'sub-1')?.controlMethod).toBeUndefined()
})

it('keeps an ambiguous transport failure as a retained control, never replayed as a prompt', async () => {
  const calls: string[] = []
  const settle = vi.fn()

  const request = vi.fn(async (method: string) => {
    calls.push(method)
    throw new Error('request timed out after 120s: session.redirect')
  })

  submitPrompt('next thought', deps({ isCanonical: true, request } as unknown as GatewayClient), true, undefined, {
    behindTurn: true,
    queueItem: correction(settle),
    skipDetectDrop: true
  })

  await vi.waitFor(() => expect(settle).toHaveBeenCalledWith(false))
  expect(calls).toEqual(['session.redirect'])
})

it('settles a plain rejected correction as refused without re-sending it as a prompt', async () => {
  // A live agent's refusal can follow delivery (Codex turn/steer whose ACK timed out): re-sending
  // the words would run the correction twice.
  const calls: string[] = []
  const settle = vi.fn()

  const request = vi.fn(async (method: string) => {
    calls.push(method)

    return { status: 'rejected', text: 'next thought', execution_generation: 4 }
  })

  submitPrompt('next thought', deps({ isCanonical: true, request } as unknown as GatewayClient), true, undefined, {
    behindTurn: true,
    queueItem: correction(settle),
    skipDetectDrop: true
  })

  await vi.waitFor(() => expect(settle).toHaveBeenCalledWith(false))
  await new Promise(resolve => setTimeout(resolve, 20))
  expect(calls).toEqual(['session.redirect'])
})

// NC8-CONTRACT-01: a managed Kanban worker's correction is kept on the card's task before the child
// sees it; when the running turn could not take it, the owner answers this exact acknowledgement
// (``gateway/session_managed_worker.KeptCorrection.unreached`` merged into the correction ACK). Its
// ``rejected``/``delivered: false`` must not re-admit it as a fresh prompt: that prompt would carry
// no claim binding. The claim may have ended (same generation) or been replaced (another one).
const keptOnTask = (generation: number) => ({
  authority_epoch: 1,
  delivered: false,
  execution_generation: generation,
  kept: 'task_comment',
  status: 'rejected',
  text: 'next thought'
})

it.each([
  ['its claim ended', 4],
  ['its claim was replaced', 5]
])('settles a correction kept on its card task as kept, never re-sent as an unbound prompt (%s)', async (_name, generation) => {
  const calls: Array<{ method: string; params: Record<string, unknown> }> = []
  const settle = vi.fn()
  const sys = vi.fn()

  const request = vi.fn(async (method: string, params: Record<string, unknown>) => {
    calls.push({ method, params })

    return method === 'prompt.submit' ? { status: 'queued' } : keptOnTask(generation)
  })

  submitPrompt('next thought', { ...deps({ isCanonical: true, request } as unknown as GatewayClient), sys }, true, undefined, {
    behindTurn: true,
    queueItem: correction(settle),
    skipDetectDrop: true
  })

  await vi.waitFor(() => expect(settle).toHaveBeenCalledWith(true))
  await new Promise(resolve => setTimeout(resolve, 20))
  expect(calls.map(c => c.method)).toEqual(['session.redirect'])
  expect(sys).toHaveBeenCalledWith('correction kept on its task')
})
