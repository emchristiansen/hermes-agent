import { correctionPin } from '@/api/canonical-protocol'
import { translateNow } from '@/i18n'
import type { UnresolvedCorrectionIdentity } from '@/lib/uncertain-correction'
import { requestGatewayForProfile } from '@/store/gateway'
import { notify } from '@/store/notifications'

// A correction whose outcome is unknown (its reply lost, or a failure after the owner may have taken it)
// stays a correction the user can send again to the destination it was captured for (NC12-CLIENT-01).
// Its words used to live only in a persistent notice and its retry closure, both renderer memory, so a
// renderer restart lost them (review t_42a8d5e2 UI-02). Each one is kept here, keyed by its correction
// id, with the exact destination its first preparation pinned: owner session, profile and execution
// generation. A restarted renderer offers each again with a Send again that presents that same id and
// generation to that same session, through that profile's gateway; the owner refuses or keeps it for
// that execution, never a newer one. It is never re-sent on its own, never becomes an ordinary prompt
// (a prompt would carry no card binding, NC13-CLIENT-02), and never goes to whichever session the
// composer shows now. Other sends are not held behind it: it is only an offer. An entry leaves when its
// outcome is known (accepted, or refused undelivered) or the user discards it; there is no expiry, since
// a correction that silently vanished is the loss this store exists to prevent.
const PREFIX = 'hermes:unresolved-correction:v1:'

export interface UnresolvedCorrection extends UnresolvedCorrectionIdentity {
  /** The words as the notice shows them (a terminal chip's display form). */
  displayText: string
  sessionId: string
  profile: string
  generation: number
  /** When it was first kept (ms): the order restored notices appear in. */
  at: number
}

export type CorrectionRequest = <T>(profile: string, method: string, params: Record<string, unknown>) => Promise<T>

const key = (id: string) => `${PREFIX}${id}`

function read(storageKey: string): UnresolvedCorrection | undefined {
  try {
    const entry = JSON.parse(window.localStorage.getItem(storageKey) || 'null') as UnresolvedCorrection | null

    return entry &&
      typeof entry.id === 'string' &&
      (entry.mode === 'interrupt' || entry.mode === 'steer') &&
      typeof entry.text === 'string' &&
      typeof entry.displayText === 'string' &&
      typeof entry.sessionId === 'string' &&
      typeof entry.profile === 'string' &&
      typeof entry.generation === 'number' &&
      typeof entry.at === 'number'
      ? entry
      : undefined
  } catch {
    return undefined
  }
}

/**
 * Keep an unresolved correction durably. Only a correction whose destination is pinned (its first
 * preparation reached the canonical protocol) can be: without a pin there is no generation to present,
 * and a restored send must never take the latest one. Returns the kept entry (the existing one when
 * already kept), or undefined when it cannot be kept.
 */
export function keepUnresolvedCorrection(
  correction: UnresolvedCorrectionIdentity,
  displayText: string
): UnresolvedCorrection | undefined {
  const existing = read(key(correction.id))

  if (existing) {
    return existing
  }

  const pin = correctionPin(correction.id)

  if (!pin) {
    return undefined
  }

  const entry: UnresolvedCorrection = { ...correction, displayText, ...pin, at: Date.now() }

  try {
    window.localStorage.setItem(key(correction.id), JSON.stringify(entry))
  } catch {
    // Storage full or unavailable: the in-memory notice still holds it, as before.
    return undefined
  }

  return entry
}

/** Its outcome is known, or the user discarded it: no longer offered after a restart. */
export function forgetUnresolvedCorrection(id: string): void {
  try {
    window.localStorage.removeItem(key(id))
  } catch {
    // Best-effort, like keeping it.
  }
}

/** Every kept correction, oldest first. */
export function unresolvedCorrections(): UnresolvedCorrection[] {
  const entries: UnresolvedCorrection[] = []

  try {
    for (let index = 0; index < window.localStorage.length; index++) {
      const storageKey = window.localStorage.key(index)
      const entry = storageKey?.startsWith(PREFIX) ? read(storageKey) : undefined

      if (entry) {
        entries.push(entry)
      }
    }
  } catch {
    return entries
  }

  return entries.sort((a, b) => a.at - b.at)
}

/**
 * Send a kept correction again, as the same correction (its id and pinned generation) to the session
 * it was captured for. Resolves true when the owner took or kept it, false when the owner confirms it
 * undelivered; any other reply is still uncertain and rejects.
 */
export async function resendUnresolvedCorrection(
  entry: UnresolvedCorrection,
  request: CorrectionRequest = requestGatewayForProfile
): Promise<boolean> {
  let result: { delivered?: boolean; kept?: string | null; status?: string; voice_stopped?: boolean } | null

  try {
    result = await request(entry.profile, entry.mode === 'steer' ? 'session.steer' : 'session.redirect', {
      session_id: entry.sessionId,
      text: entry.text,
      correction_id: entry.id,
      execution_generation: entry.generation
    })
  } catch (err) {
    // The owner's typed refusal of a correction no agent saw (the run it named has ended).
    if ((err as { data?: { delivered?: unknown } })?.data?.delivered === false) {
      return false
    }

    throw err
  }

  if (
    result?.voice_stopped === true ||
    result?.status === 'redirected' ||
    result?.status === 'queued' ||
    result?.kept === 'task_comment'
  ) {
    return true
  }

  if (result?.status === 'rejected' && result.delivered === false) {
    return false
  }

  throw new Error(`Correction outcome unconfirmed: ${JSON.stringify(result ?? null)}`)
}

export const unresolvedCorrectionNoticeId = (id: string) => `unresolved-correction:${id}`

/**
 * The persistent notice for a kept correction: Send again retries it to its own destination (the
 * in-page retry while this renderer holds one, otherwise the kept destination), and Dismiss drops it.
 * Not `onDismiss`: the notice is dismissed when Send again is clicked too, and the entry must outlive
 * a retry in flight (a restart during it offers the correction again).
 */
export function offerUnresolvedCorrection(
  entry: UnresolvedCorrection,
  retry: () => Promise<boolean> = () => resendUnresolvedCorrection(entry)
): void {
  notify({
    id: unresolvedCorrectionNoticeId(entry.id),
    kind: 'warning',
    persistent: true,
    durationMs: 0,
    title: translateNow('composer.correctionUnconfirmedTitle'),
    message: translateNow('composer.correctionUnconfirmedBody'),
    detail: entry.displayText,
    action: {
      label: translateNow('composer.correctionSendAgain'),
      onClick: () => void settleRestored(entry, retry)
    },
    secondaryAction: {
      label: translateNow('statusStack.dismiss'),
      onClick: () => forgetUnresolvedCorrection(entry.id)
    }
  })
}

function settleRestored(entry: UnresolvedCorrection, retry: () => Promise<boolean>): Promise<void> {
  return retry().then(
    accepted => {
      forgetUnresolvedCorrection(entry.id)

      if (!accepted) {
        // Refused undelivered after a restart: no draft it was typed in is loaded to give it back to,
        // so its words stay in a persistent notice, never a prompt.
        notify({
          kind: 'warning',
          persistent: true,
          title: translateNow('composer.correctionNotSentTitle'),
          message: translateNow('composer.correctionNotSentBody'),
          detail: entry.displayText
        })
      }
    },
    () => offerUnresolvedCorrection(entry, retry)
  )
}

let restored = false

/** Once per page: offer every correction a previous renderer left unresolved. */
export function restoreUnresolvedCorrections(): void {
  if (restored) {
    return
  }

  restored = true

  for (const entry of unresolvedCorrections()) {
    offerUnresolvedCorrection(entry)
  }
}
