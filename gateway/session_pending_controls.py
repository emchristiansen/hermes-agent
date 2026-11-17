"""Generation-bound projection/control of the existing blocking approval queue.

Ordinary approvals and clarify prompts share the existing native waiters.
Secret/sudo prompts remain native; no response is published or journaled here.
"""
from copy import deepcopy

from hermes_state_runtime import RuntimeStoreError
from tools.approval import list_gateway_approvals, resolve_gateway_approval


class PendingControls:
    def __init__(self, events):
        self.events = events
        self.pending = {}
        self.remote_responders = {}
        self._executions = {}

    def _register(self, session_id, route, prompt):
        # Registration runs under the owner's generation fence and stream lock.
        # A later snapshot/response must not borrow a successor's execution stamp.
        generation = prompt['execution_generation']
        execution = dict(self.events.execution)
        if execution and execution.get('execution_generation') != generation:
            raise RuntimeStoreError('stale_generation')
        # Non-admission users can have an unstamped stream. Never invent their
        # admission/epoch; completion may supply its committed row's identity.
        execution['execution_generation'] = generation
        prompt_id = prompt['prompt_id']
        self.pending[prompt_id] = (route, prompt)
        self._executions[prompt_id] = execution
        self.events.publish(session_id, prompt, event_type=prompt['kind'] + '.request', execution=execution)

    def settle(self, session_id, prompt_id, generation, *, execution=None):
        """Remove/publish one owning projection; a stale callback cannot consume its successor."""
        with self.events.lock:
            saved = self.pending.get(prompt_id)
            if saved is None or saved[1]['execution_generation'] != generation:
                return False
            _route, prompt = self.pending.pop(prompt_id)
            owning = self._executions.pop(prompt_id)
            self.remote_responders.pop(prompt_id, None)
            if 'admission_id' not in owning and execution is not None:
                owning = execution
            self.events.publish(session_id, {'prompt_id': prompt_id, 'execution_generation': generation},
                                event_type=prompt['kind'] + '.settled', execution=owning)
            return True

    def register(self, session_id, route, generation, data, *, timeout_s=None):
        from gateway.run import _redact_approval_command
        with self.events.lock:
            prompt_id = data['request_id']
            choices = ['once', 'deny']
            if data.get('allow_session', True) and not data.get('smart_denied', False):
                choices.append('session')
            if data.get('allow_permanent', True) and not data.get('smart_denied', False):
                choices.append('always')
            prompt = {'kind': 'approval', 'prompt_id': prompt_id,
                      'execution_generation': generation,
                      'command': _redact_approval_command(data.get('command', '')),
                      'description': _redact_approval_command(data.get('description', '')),
                      'choices': choices}
            if 'edit' in data:
                prompt['edit'] = deepcopy(data['edit'])
                prompt['choices'] = ['once', 'deny']
            if timeout_s is not None:
                prompt['timeout_s'] = timeout_s
            self._register(session_id, route, prompt)

    def register_clarify(self, session_id, generation, entry, *, timeout_s=None):
        from gateway.run import _redact_approval_command
        with self.events.lock:
            prompt = {"kind": "clarify", "prompt_id": entry.clarify_id,
                      "execution_generation": generation,
                      "question": _redact_approval_command(entry.question),
                      "choices": [_redact_approval_command(c) for c in entry.choices or []],
                      "multi_select": entry.multi_select}
            if timeout_s is not None:
                prompt["timeout_s"] = timeout_s
            self._register(session_id, entry, prompt)

    def complete_execution(self, session_id, execution):
        """Settle only the completed claim's projections, not a later running turn's.

        The same stamp addresses cleanup frames and completion. Publication of an
        older result must not alter the stream's currently executing claim.
        """
        with self.events.lock:
            generation = execution['execution_generation']
            if generation is None:  # an unclaimed row owns no execution controls
                return
            for prompt_id, (_route, prompt) in list(self.pending.items()):
                if prompt['execution_generation'] == generation:
                    self.settle(session_id, prompt_id, generation, execution=execution)

    def snapshot(self, session_id, generation):
        with self.events.lock:
            for prompt_id, (route, prompt) in list(self.pending.items()):
                active = (prompt_id in self.remote_responders or
                          (not route.event.is_set() if prompt['kind'] == 'clarify' else
                           any(p['request_id'] == prompt_id for p in list_gateway_approvals(route))))
                if prompt['execution_generation'] != generation or not active:
                    self.settle(session_id, prompt_id, prompt['execution_generation'])
            return tuple(deepcopy(prompt) for _, prompt in self.pending.values())

    def respond(self, session_id, generation, prompt_id, response, *, kind="approval"):
        with self.events.lock:
            self.snapshot(session_id, generation)
            entry = self.pending.get(prompt_id)
            if entry is None:
                return {'status': 'already_resolved', 'prompt_id': prompt_id}
            route, prompt = entry
            if prompt['kind'] != kind:
                raise RuntimeStoreError('invalid_params')
            if prompt_id in self.remote_responders:
                field = 'answer' if kind == 'clarify' else 'choice'
                if (not isinstance(response, dict) or set(response) != {field}
                        or not isinstance(response[field], str) or len(response[field]) > 16384
                        or (kind == 'approval' and response[field] not in prompt['choices'])):
                    raise RuntimeStoreError('invalid_params')
                self.remote_responders[prompt_id](kind, prompt_id, response[field])
                self.settle(session_id, prompt_id, generation)
                return {'status': 'resolved', 'prompt_id': prompt_id}
            if kind == 'clarify':
                from tools.clarify_gateway import SKIPPED, resolve_gateway_clarify
                if (not isinstance(response, dict) or set(response) != {'answer'}
                        or not isinstance(response['answer'], str)):
                    raise RuntimeStoreError('invalid_params')
                # An empty answer is the documented skip; the waiter reads a bare "" as a timeout.
                resolved = resolve_gateway_clarify(prompt_id, response['answer'] if response['answer'].strip() else SKIPPED)
                self.snapshot(session_id, generation)
                return {'status': 'resolved' if resolved else 'already_resolved', 'prompt_id': prompt_id}
            if not isinstance(response, dict) or set(response) != {'choice'} or response['choice'] not in prompt['choices']:
                raise RuntimeStoreError('invalid_params')
            # The tools lock arbitrates native taps versus attached responders;
            # an exact request ID can never consume the next FIFO approval.
            resolved = resolve_gateway_approval(route, response['choice'], request_id=prompt_id)
            self.snapshot(session_id, generation)
            return {'status': 'resolved' if resolved else 'already_resolved', 'prompt_id': prompt_id}
