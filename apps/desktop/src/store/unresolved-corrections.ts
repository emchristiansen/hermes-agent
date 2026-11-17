import { correctionPin, onCorrectionPinned } from '@/api/canonical-protocol'
import { translateNow } from '@/i18n'
import { UncertainCorrection, type UnresolvedCorrectionIdentity, UnsentCorrection } from '@/lib/uncertain-correction'
import { activeGatewayRoute } from '@/store/gateway'
import { dismissNotification, notify } from '@/store/notifications'
import {
  isSessionOwnerRoute,
  profileScopeForSessionOwner,
  requestForSessionProfile,
  type SessionOwnerRoute
} from '@/store/session-request-router'
import { knownOwnerForSession } from '@/store/session-states'

// A correction whose outcome is unknown (its reply lost, or a failure after the owner may have taken it)
// stays a correction, delivered again to the destination it was captured for (NC12-CLIENT-01). Each one
// is kept here, keyed by its correction id, from BEFORE its first request leaves (review22815 UI-R2: a
// renderer restart while that first request or its readiness wait is still pending must not lose the
// words), with the exact destination: the owner route its send used (registry connection, Desktop
// profile, backend target profile; UI-R3), the owner session, and the execution generation its first
// preparation pinned, recorded the moment that pin is made, before the request that carries it leaves
// (UI-R1: a crash during a restored retry must not lose the generation that retry was prepared for).
//
// Delivery is unattended (review22815 UI-R4, Eric's at-least-once directive): every unresolved
// correction, this page's own and each one a previous renderer left, is sent again on its own, with
// backoff, as the same correction (its id and pinned generation) to that same destination, until the
// owner takes it, keeps it, or refuses it undelivered; the owner refuses or keeps it for that execution,
// never a newer one. A persistent notice shows it meanwhile, with Send again (now) and Dismiss. It is never
// re-admitted as an ordinary prompt (a prompt would carry no card binding, NC13-CLIENT-02) and never goes
// to whichever session the composer shows now. Nothing waits for it: each retry runs on its own timer,
// so no other send is held behind it (IL465010: order is a minor concern and never blocks a send). An
// entry leaves when its outcome is known or the user discards it; there is no expiry, since a correction
// that silently vanished is the loss this store exists to prevent.
const PREFIX = 'hermes:unresolved-correction:v1:'

/** Where a kept correction goes: the exact owner route its send used, else the bare owner profile, else
 * null (its send went through the ambient route; a retry then uses the kept profile's own route). */
export type CorrectionOwner = SessionOwnerRoute | string | null

export interface UnresolvedCorrection extends UnresolvedCorrectionIdentity {
  /** The words as the notice shows them (a terminal chip's display form). */
  displayText: string
  sessionId: string
  profile: string
  /** The owner route of the send (UI-R3). Absent on records kept before it was recorded (null). */
  owner: CorrectionOwner
  /** The pinned execution generation; null while no preparation has pinned one: a retry then pins the
   * session's current generation, for this session only, and that pin is recorded at once. */
  generation: number | null
  /** When it was first kept (ms): the order restored notices appear in. */
  at: number
}

export type CorrectionRequest = <T>(owner: CorrectionOwner, method: string, params: Record<string, unknown>) => Promise<T>

const key = (id: string) => `${PREFIX}${id}`

const optionalString = (value: unknown) => value === undefined || typeof value === 'string'

function validOwner(owner: unknown): owner is CorrectionOwner | undefined {
  if (owner === undefined || owner === null || typeof owner === 'string') {
    return true
  }

  const route = owner as Partial<SessionOwnerRoute>

  return (
    typeof owner === 'object' &&
    typeof route.connectionId === 'string' &&
    typeof route.profile === 'string' &&
    optionalString(route.targetProfile) &&
    (route.mode === undefined || route.mode === 'local' || route.mode === 'remote')
  )
}

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
      validOwner(entry.owner) &&
      (typeof entry.generation === 'number' || entry.generation === null) &&
      typeof entry.at === 'number'
      ? { ...entry, owner: entry.owner ?? null }
      : undefined
  } catch {
    return undefined
  }
}

function write(entry: UnresolvedCorrection): UnresolvedCorrection | undefined {
  try {
    window.localStorage.setItem(key(entry.id), JSON.stringify(entry))
  } catch {
    // Storage full or unavailable: this page still delivers it (its notice and retry are in memory).
    return undefined
  }

  return entry
}

/** The session's known owner: its exact route when one is known, else its owner profile. */
function knownOwner(sessionId: string): CorrectionOwner {
  const owner = knownOwnerForSession(sessionId)

  return isSessionOwnerRoute(owner) ? owner : typeof owner === 'string' && owner.trim() ? owner : null
}

/** The profile that owns a session (its exact owner route when known, else the active route's). */
function ownerProfile(sessionId: string, owner: CorrectionOwner): string {
  const scope = profileScopeForSessionOwner(owner ?? knownOwnerForSession(sessionId))

  return (typeof scope === 'string' ? scope : scope?.profile) || activeGatewayRoute().profile
}

// Corrections this page sends or delivers itself: a restore never adopts them a second time.
const ownedHere = new Set<string>()

/**
 * Journal a correction durably BEFORE its first request leaves (UI-R2), with the owner route that send
 * uses (UI-R3). Its generation is recorded when its first preparation pins one (below). Synchronous, so
 * a caller that journals before its first await cannot lose the words to a restart. Returns the entry,
 * or undefined when storage refuses it.
 */
export function journalCorrection(
  correction: UnresolvedCorrectionIdentity,
  owner: CorrectionOwner | undefined,
  displayText: string = correction.text
): UnresolvedCorrection | undefined {
  ownedHere.add(correction.id)
  const existing = read(key(correction.id))

  if (existing) {
    return existing
  }

  const route = owner ?? knownOwner(correction.sessionId)
  const pin = correctionPin(correction.id)

  return write({
    ...correction,
    displayText,
    owner: route,
    ...(pin ?? { sessionId: correction.sessionId, profile: ownerProfile(correction.sessionId, route), generation: null }),
    at: Date.now()
  })
}

// The pin is recorded the moment the canonical protocol makes it, before the request carrying it leaves.
onCorrectionPinned((id, pin) => {
  const existing = read(key(id))

  if (existing && existing.generation === null) {
    write({ ...existing, sessionId: pin.sessionId, profile: pin.profile, generation: pin.generation })
  }
})

/**
 * Keep an unresolved correction durably (its journal entry when it has one, with the words as shown).
 * A pinned one keeps the pinned session, profile and generation; an unpinned one keeps the session it was
 * captured for and that session's owner, with no generation. A pin made by a later attempt replaces an
 * unpinned record. Returns the kept entry, or undefined when storage refuses it.
 */
export function keepUnresolvedCorrection(
  correction: UnresolvedCorrectionIdentity,
  displayText: string
): UnresolvedCorrection | undefined {
  const existing = read(key(correction.id)) ?? journalCorrection(correction, undefined, displayText)

  if (!existing) {
    return undefined
  }

  const pin = existing.generation === null ? correctionPin(correction.id) : undefined

  return existing.displayText === displayText && !pin ? existing : write({ ...existing, ...pin, displayText })
}

/**
 * A correction's first send on the canonical owner, journaled before it starts (UI-R2) with the owner
 * route it uses (UI-R3). A known outcome, or a send that never left (anything but an
 * UncertainCorrection), forgets it; an uncertain one stays kept for its unattended delivery.
 */
export function sendJournaledCorrection(
  correction: UnresolvedCorrectionIdentity,
  owner: CorrectionOwner | undefined,
  attempt: () => Promise<boolean>
): Promise<boolean> {
  journalCorrection(correction, owner)

  return attempt().then(
    accepted => {
      forgetUnresolvedCorrection(correction.id)

      return accepted
    },
    (err: unknown) => {
      if (!(err instanceof UncertainCorrection)) {
        forgetUnresolvedCorrection(correction.id)
      }

      throw err
    }
  )
}

/** Its outcome is known, or the user discarded it: no longer delivered or offered, here or after a restart. */
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

const routedRequest: CorrectionRequest = <T>(owner: CorrectionOwner, method: string, params: Record<string, unknown>) =>
  requestForSessionProfile<T>(
    owner,
    () => Promise.reject(new Error('Correction owner unknown; not sent through the shown session')),
    method,
    params
  )

/**
 * Send a kept correction again, as the same correction (its id, and its pinned generation when it has
 * one) to the session it was captured for, through its own owner route (UI-R3). Resolves true when the
 * owner took or kept it, false when the owner confirms it undelivered; any other reply is still
 * uncertain and rejects.
 */
export async function resendUnresolvedCorrection(
  entry: UnresolvedCorrection,
  request: CorrectionRequest = routedRequest
): Promise<boolean> {
  // The kept generation, read again: a pin recorded since this entry was loaded is the one to present.
  const generation = read(key(entry.id))?.generation ?? entry.generation
  let result: { delivered?: boolean; kept?: string | null; status?: string; voice_stopped?: boolean } | null

  try {
    result = await request(entry.owner ?? entry.profile, entry.mode === 'steer' ? 'session.steer' : 'session.redirect', {
      session_id: entry.sessionId,
      text: entry.text,
      correction_id: entry.id,
      ...(generation === null ? {} : { execution_generation: generation })
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

/** What a delivered correction's known outcome does where it was typed. */
export interface CorrectionOutcome {
  /** Refused undelivered: its words go back (a queue, the draft, or a notice). */
  refused: () => void
  /** Never sent there (its session changed under it): its words go back to the composer. */
  unsent: () => void
}

/** First unattended retry after this long, doubling up to the cap. */
export const CORRECTION_RETRY_FIRST_MS = 1_000
export const CORRECTION_RETRY_MAX_MS = 60_000

interface Delivery {
  attempt: () => Promise<boolean>
  delay: number
  inFlight: boolean
  outcome: CorrectionOutcome
  stopped: boolean
  timer?: ReturnType<typeof setTimeout>
}

const deliveries = new Map<string, Delivery>()

function finish(id: string): void {
  const delivery = deliveries.get(id)

  if (delivery) {
    delivery.stopped = true
    clearTimeout(delivery.timer)
    deliveries.delete(id)
  }

  forgetUnresolvedCorrection(id)
  dismissNotification(unresolvedCorrectionNoticeId(id))
}

function schedule(id: string, delivery: Delivery): void {
  clearTimeout(delivery.timer)
  delivery.timer = setTimeout(() => void deliver(id), delivery.delay)
  delivery.delay = Math.min(delivery.delay * 2, CORRECTION_RETRY_MAX_MS)
}

/** One delivery attempt; an uncertain one schedules the next. Never awaited by any other send. */
async function deliver(id: string): Promise<void> {
  const delivery = deliveries.get(id)

  if (!delivery || delivery.stopped || delivery.inFlight) {
    return
  }

  delivery.inFlight = true
  clearTimeout(delivery.timer)

  let accepted: boolean

  try {
    accepted = await delivery.attempt()
  } catch (err) {
    delivery.inFlight = false

    if (delivery.stopped) {
      return
    }

    if (err instanceof UnsentCorrection) {
      finish(id)
      delivery.outcome.unsent()
    } else {
      schedule(id, delivery)
    }

    return
  }

  delivery.inFlight = false

  if (!delivery.stopped) {
    finish(id)

    if (!accepted) {
      delivery.outcome.refused()
    }
  }
}

/**
 * Deliver an unresolved correction unattended (UI-R4): retried on its own with backoff until its outcome
 * is known, with a persistent notice meanwhile (Send again: now; Dismiss: drop it). `attempt` sends the
 * same correction to its own destination and resolves like the first send (true taken or kept, false
 * refused undelivered); an UnsentCorrection ends it as unsent, any other failure is still uncertain.
 */
export function deliverUnresolvedCorrection(
  entry: Pick<UnresolvedCorrection, 'displayText' | 'id'>,
  attempt: () => Promise<boolean>,
  outcome: CorrectionOutcome
): void {
  ownedHere.add(entry.id)
  const previous = deliveries.get(entry.id)

  if (previous) {
    previous.stopped = true
    clearTimeout(previous.timer)
  }

  const delivery: Delivery = { attempt, delay: CORRECTION_RETRY_FIRST_MS, inFlight: false, outcome, stopped: false }
  deliveries.set(entry.id, delivery)

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
      onClick: () => void deliver(entry.id)
    },
    secondaryAction: {
      label: translateNow('statusStack.dismiss'),
      onClick: () => finish(entry.id)
    }
  })

  schedule(entry.id, delivery)
}

let transient = 0

/**
 * The composer's uncertain correction: kept durably with its destination (its journal entry, the words
 * as shown) and delivered unattended through this page's own retry, which settles where it was typed.
 */
export function deliverUncertainCorrection(err: UncertainCorrection, displayText: string, outcome: CorrectionOutcome): void {
  const kept = err.correction ? keepUnresolvedCorrection(err.correction, displayText) : undefined
  const id = kept?.id ?? err.correction?.id ?? `transient:${++transient}`

  deliverUnresolvedCorrection({ id, displayText }, err.retry, outcome)
}

/** Refused undelivered after a restart: no draft it was typed in is loaded, so its words stay in a notice. */
function notifyRestoredNotSent(entry: UnresolvedCorrection): void {
  notify({
    kind: 'warning',
    persistent: true,
    title: translateNow('composer.correctionNotSentTitle'),
    message: translateNow('composer.correctionNotSentBody'),
    detail: entry.displayText
  })
}

/** A kept correction from a previous renderer, delivered unattended to its own kept destination. */
export function offerUnresolvedCorrection(
  entry: UnresolvedCorrection,
  retry: () => Promise<boolean> = () => resendUnresolvedCorrection(entry)
): void {
  deliverUnresolvedCorrection(entry, retry, {
    refused: () => notifyRestoredNotSent(entry),
    unsent: () => notifyRestoredNotSent(entry)
  })
}

let restored = false

/** Once per page: deliver every correction a previous renderer left unresolved (not this page's own). */
export function restoreUnresolvedCorrections(): void {
  if (restored) {
    return
  }

  restored = true

  for (const entry of unresolvedCorrections()) {
    if (!ownedHere.has(entry.id)) {
      offerUnresolvedCorrection(entry)
    }
  }
}
