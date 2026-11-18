/**
 * A busy correction (steer/redirect) whose fate is unknown: its reply was lost, or the call failed
 * after the owner may already have delivered it, or kept it on a Kanban card's task. It stays a
 * correction. `retry` sends the same words again, as the same kind of correction, to the session and
 * execution generation it was captured for (the owner refuses or keeps it for that execution, never a
 * later one), and resolves like the first attempt (true accepted, false refused undelivered).
 * A duplicate correction is acceptable; re-admitting the words as an ordinary prompt is not, since
 * that prompt carries no card binding (NC12-CLIENT-01).
 */
export class UncertainCorrection extends Error {
  readonly retry: () => Promise<boolean>
  /** Which correction this is: its id (the key of its pinned destination), kind and words. With it the
   * composer keeps the unresolved correction durably, so a renderer restart cannot lose it (UI-02). */
  readonly correction?: UnresolvedCorrectionIdentity

  constructor(cause: unknown, retry: () => Promise<boolean>, correction?: UnresolvedCorrectionIdentity) {
    super(cause instanceof Error ? cause.message : String(cause), { cause })
    this.name = 'UncertainCorrection'
    this.retry = retry
    this.correction = correction
  }
}

export interface UnresolvedCorrectionIdentity {
  id: string
  mode: 'interrupt' | 'steer'
  text: string
  /** The session its first send was captured for: its destination when no preparation pinned one. */
  sessionId: string
}

/**
 * A correction that was never sent: its target could not be established on the canonical owner, or a
 * retry would have reached another session than the one its first send named. Handing it back as the
 * next prompt would make it a task-unbound prompt (NC13-CLIENT-02), so the caller returns the words
 * to the composer.
 */
export class UnsentCorrection extends Error {
  constructor(reason = 'its session could not be established') {
    super(`Correction not sent: ${reason}`)
    this.name = 'UnsentCorrection'
  }
}

/**
 * The canonical protocol's refusal, before the wire, of a correction retry that names another owner
 * session (another runtime, profile or stored chat) than its first send: its pinned execution
 * generation means nothing there, so it is sent nowhere (NC13-CLIENT-01).
 */
export class CorrectionTargetChanged extends Error {
  constructor() {
    super('Correction not sent: its session is no longer the one its first send named')
    this.name = 'CorrectionTargetChanged'
  }
}
