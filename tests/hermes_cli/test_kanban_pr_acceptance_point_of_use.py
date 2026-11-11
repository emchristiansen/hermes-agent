"""PR-contract completion with no repository-required CI, and the assignee's point-of-use gh token.

Real SQLite through the shared ``complete_task`` boundary; a real ``gh`` child (a shim on PATH)
that accepts only the expected token, and a real token-command child configured in the
ASSIGNEE profile's own ``config.yaml``. A local HTTP server plays GitHub."""
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli.kanban_db_connect import connect

PR = "https://github.com/acme/repo/pull/7"
HEAD = "a" * 40
TOKEN = "point-of-use-token"
DROP = object()  # remove a key from the GraphQL baseRef: missing evidence, not an explicit null


def _graphql(state):
    if "graphql" in state:
        return state["graphql"]
    base_ref = {"branchProtectionRule": None, "refUpdateRule": None}
    for key, value in state.get("base_ref", {}).items():
        if value is DROP:
            base_ref.pop(key)
        else:
            base_ref[key] = value
    pr = {"headRefOid": HEAD, "baseRefName": "main", "state": "MERGED", "baseRef": base_ref}
    if state.get("base_ref_value", pr) is not pr:
        pr["baseRef"] = state["base_ref_value"]
    return {"data": {"repository": {"pullRequest": pr}}}


def _pull(state):
    if state.get("on_reread"):
        state["on_reread"]()
    return {"head": {"sha": state.get("head_after", HEAD)}, "base": {"ref": state.get("base_after", "main")},
            "state": state.get("pr_state", "closed"), "merged": state.get("merged", True)}


def _routes(state):
    runs = state.get("runs", [])
    return [
        ("/graphql", lambda: _graphql(state)),
        ("/rules/branches/", lambda: state.get("rules", [[]])),
        ("/check-runs", lambda: [{"total_count": len(runs), "check_runs": runs}]),
        ("/statuses", lambda: [[]]),
        ("/pulls/", lambda: _pull(state)),
    ]


@pytest.fixture
def gh(tmp_path, monkeypatch):
    """GitHub server + gh shim + an assignee profile ``reviewer`` with a token command."""
    state = {"requests": []}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            state["requests"].append(self.path)
            for marker, respond in _routes(state):
                if marker in self.path:
                    value = respond()
                    break
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps(value).encode())

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    gh_log, helper_log, control = tmp_path / "gh.jsonl", tmp_path / "helper.jsonl", tmp_path / "helper.json"
    shim = tmp_path / "bin"
    shim.mkdir()
    gh = shim / "gh"
    # Real gh semantics that matter here: no token -> exit 4 ("not logged in"); a token GitHub
    # rejects -> "HTTP 401" on stderr; an unreadable endpoint -> "HTTP 404".
    gh.write_text(
        f"#!{sys.executable}\nimport json, os, sys, urllib.error, urllib.request\n"
        f"with open({str(gh_log)!r}, 'a') as f:\n"
        "    f.write(json.dumps({'argv': sys.argv[1:], 'cwd': os.getcwd(),\n"
        "        'env': {k: os.environ.get(k) for k in ('GH_TOKEN', 'GITHUB_TOKEN', 'GH_CONFIG_DIR')}}) + '\\n')\n"
        "token = os.environ.get('GH_TOKEN')\n"
        "if not token:\n    sys.exit(4)\n"
        f"if token != {TOKEN!r}:\n    print('gh: Bad credentials (HTTP 401)', file=sys.stderr); sys.exit(1)\n"
        "try:\n"
        f"    print(urllib.request.urlopen('http://127.0.0.1:{server.server_port}/' + sys.argv[2]).read().decode())\n"
        "except urllib.error.HTTPError as exc:\n"
        "    print(f'gh: Not Found (HTTP {exc.code})', file=sys.stderr); sys.exit(1)\n")
    gh.chmod(0o755)
    helper = tmp_path / "token_helper.py"
    helper.write_text(
        "import json, os, sys\n"
        f"with open({str(helper_log)!r}, 'a') as f:\n"
        "    f.write(json.dumps({'owner': os.environ.get('GH_ACCEPTANCE_OWNER'), 'cwd': os.getcwd(),\n"
        "        'GH_TOKEN': os.environ.get('GH_TOKEN'), 'HERMES_HOME': os.environ.get('HERMES_HOME')}) + '\\n')\n"
        f"control = json.load(open({str(control)!r}))\n"
        "sys.stdout.write(control['stdout'])\nsys.exit(control['exit'])\n")
    control.write_text(json.dumps({"stdout": TOKEN + "\n", "exit": 0}))

    launch_home = tmp_path / "home"
    reviewer = launch_home / "profiles" / "reviewer"
    reviewer.mkdir(parents=True)
    (reviewer / ".env").write_text("", encoding="utf-8")
    (reviewer / "config.yaml").write_text(json.dumps(
        {"kanban": {"pr_acceptance": {"gh_token_command": [sys.executable, str(helper)]}}}), encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(launch_home))
    monkeypatch.setenv("PATH", str(shim) + os.pathsep + os.environ["PATH"])
    # Ambient launch login: must never decide the read, and never reach the helper.
    monkeypatch.setenv("GH_TOKEN", TOKEN)
    monkeypatch.setenv("GITHUB_TOKEN", TOKEN)
    kb.init_db()

    def read(path):
        return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []

    state.update(home=launch_home, reviewer=reviewer, control=control,
                 gh_calls=lambda: read(gh_log), helper_calls=lambda: read(helper_log))
    try:
        yield state
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def _complete(conn, assignee="reviewer"):
    tid = kb.create_task(conn, title="publish", completion_contract="acme/repo", assignee=assignee)
    ok = kb.complete_task(conn, tid, result="done", metadata={"published_pr": PR})
    receipts = [json.loads(r[0]) for r in conn.execute(
        "SELECT payload FROM task_events WHERE task_id=? AND kind='pr_acceptance'", (tid,))]
    return tid, ok, receipts


NO_REQUIRED_SHAPES = {
    "no protection rule": {},
    "rule without required checks": {"branchProtectionRule": {"requiresStatusChecks": False,
                                                              "requiredStatusChecks": []}},
    "rule not requiring checks reports no list": {"branchProtectionRule": {"requiresStatusChecks": False,
                                                                           "requiredStatusChecks": None}},
    "non-admin view without contexts": {"refUpdateRule": {"requiredStatusCheckContexts": []}},
}


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("base_ref", NO_REQUIRED_SHAPES.values(), ids=NO_REQUIRED_SHAPES.keys())
def test_no_required_checks_completes_through_the_assignee_point_of_use_token(gh, base_ref):
    gh["base_ref"] = base_ref
    with connect() as conn:
        tid, ok, receipts = _complete(conn)
        task = kb.get_task(conn, tid)
    assert ok is True and task.status == "done"
    assert receipts[-1]["ok"] is True and receipts[-1]["classification"] == "success"
    assert receipts[-1]["required"] == [] and receipts[-1]["head_sha"] == HEAD
    assert "require no checks" in receipts[-1]["detail"]
    # Final PR reread still happens with zero requirements; no check evidence is fabricated.
    assert any("/pulls/7" in p for p in gh["requests"])
    assert not any("/check-runs" in p for p in gh["requests"])
    # One point-of-use fetch per acceptance, as the assignee, with the owner as data.
    (helper,) = gh["helper_calls"]()
    assert helper["owner"] == "acme" and helper["GH_TOKEN"] is None
    assert helper["HERMES_HOME"] == str(gh["reviewer"])
    calls = gh["gh_calls"]()
    assert [c["argv"][1] for c in calls] == [
        "graphql", "repos/acme/repo/rules/branches/main?per_page=100", "repos/acme/repo/pulls/7"]
    assert all(c["env"]["GH_TOKEN"] == TOKEN and c["env"]["GITHUB_TOKEN"] is None for c in calls)
    assert TOKEN not in json.dumps(receipts)


INCOMPLETE_EVIDENCE = {
    "baseRef missing": {"base_ref": {"branchProtectionRule": DROP, "refUpdateRule": DROP}, "base_ref_value": DROP},
    "baseRef null": {"base_ref_value": None},
    "branchProtectionRule missing": {"base_ref": {"branchProtectionRule": DROP}},
    "refUpdateRule missing": {"base_ref": {"refUpdateRule": DROP}},
    "required list missing": {"base_ref": {"branchProtectionRule": {"requiresStatusChecks": True}}},
    "required list null while required": {"base_ref": {"branchProtectionRule": {
        "requiresStatusChecks": True, "requiredStatusChecks": None}}},
    "requiresStatusChecks malformed": {"base_ref": {"branchProtectionRule": {
        "requiresStatusChecks": None, "requiredStatusChecks": []}}},
    "non-admin contexts null": {"base_ref": {"refUpdateRule": {"requiredStatusCheckContexts": None}}},
    "ruleset slurp empty": {"rules": []},
    "ruleset slurp malformed": {"rules": {"message": "x"}},
    "ruleset page malformed": {"rules": [{"type": "required_status_checks"}]},
    "graphql errors": {"graphql": {"errors": [{"message": "x"}], "data": None}},
}


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("fault", INCOMPLETE_EVIDENCE.values(), ids=INCOMPLETE_EVIDENCE.keys())
def test_missing_or_malformed_requirement_evidence_never_reads_as_none_required(gh, fault):
    for key, value in fault.items():
        if key == "base_ref_value" and value is DROP:
            gh["graphql"] = {"data": {"repository": {"pullRequest": {
                "headRefOid": HEAD, "baseRefName": "main", "state": "MERGED"}}}}
        else:
            gh[key] = value
    with connect() as conn:
        tid, ok, receipts = _complete(conn)
        assert not ok and kb.get_task(conn, tid).status != "done"
    assert receipts[-1]["ok"] is False and receipts[-1]["classification"] == "infra"


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("rule", ["workflows", "code_scanning", "code_quality", "code_coverage",
                                  "license_compliance_scanning", "required_deployments"])
def test_ruleset_ci_the_gate_cannot_evaluate_refuses(gh, rule):
    gh["rules"] = [[{"type": rule, "parameters": {}}]]
    with connect() as conn:
        tid, ok, receipts = _complete(conn)
        assert not ok and kb.get_task(conn, tid).status != "done"
    assert receipts[-1]["classification"] == "missing" and rule in receipts[-1]["detail"]


def _run(**fields):
    return {"id": 5, "name": "ci", "head_sha": HEAD, "app": {"id": 9}, "status": "completed",
            "conclusion": "success", "html_url": "https://github.com/acme/repo/actions/runs/5", **fields}


REQUIRED_OUTCOMES = {
    "missing": ([], "missing"),
    "pending": ([_run(status="in_progress", conclusion=None)], "pending"),
    "failed": ([_run(conclusion="failure")], "failure"),
    "old head": ([_run(head_sha="b" * 40)], "stale"),
    "success": ([_run()], "success"),
}


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("source", ["non-admin view", "ruleset"])
@pytest.mark.parametrize("runs,expected", REQUIRED_OUTCOMES.values(), ids=REQUIRED_OUTCOMES.keys())
def test_required_checks_from_any_source_still_demand_current_success(gh, source, runs, expected):
    if source == "ruleset":
        gh["rules"] = [[{"type": "required_status_checks",
                         "parameters": {"required_status_checks": [{"context": "ci"}]}}]]
    else:
        gh["base_ref"] = {"refUpdateRule": {"requiredStatusCheckContexts": ["ci"]}}
    gh["runs"] = runs
    with connect() as conn:
        tid, ok, receipts = _complete(conn)
        assert (kb.get_task(conn, tid).status == "done") is ok
    assert ok is (expected == "success")
    assert receipts[-1]["classification"] == expected
    assert receipts[-1]["required"] == [{"context": "ci", "app_id": None}]


STALE = {"head moved": {"head_after": "c" * 40}, "base moved": {"base_after": "release"},
         "closed unmerged": {"pr_state": "closed", "merged": False}}


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("change", STALE.values(), ids=STALE.keys())
def test_no_required_checks_still_refuses_a_stale_pr(gh, change):
    gh.update(change)
    with connect() as conn:
        tid, ok, receipts = _complete(conn)
        assert not ok and kb.get_task(conn, tid).status != "done"
    assert receipts[-1]["classification"] == "stale"


IDENTITY_FAULTS = {
    "token command fails": ({"stdout": "", "exit": 3}, "exited 3"),
    "token command prints nothing": ({"stdout": "\n", "exit": 0}, "no usable token"),
    "token command prints noise": ({"stdout": "two words\n", "exit": 0}, "no usable token"),
    "token of another identity": ({"stdout": "someone-else\n", "exit": 0}, "HTTP 401"),
}


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("control,detail", IDENTITY_FAULTS.values(), ids=IDENTITY_FAULTS.keys())
def test_point_of_use_identity_failures_are_auth(gh, control, detail):
    gh["control"].write_text(json.dumps(control))
    with connect() as conn:
        tid, ok, receipts = _complete(conn)
        assert not ok and kb.get_task(conn, tid).status != "done"
    assert receipts[-1]["classification"] == "auth" and detail in receipts[-1]["detail"]
    assert "'reviewer'" in receipts[-1]["detail"]
    assert "someone-else" not in json.dumps(receipts)
    if detail != "HTTP 401":
        assert gh["gh_calls"]() == []  # gh never runs without the assignee's own token


@pytest.mark.platforms("posix")
def test_malformed_token_command_is_auth_not_ambient(gh):
    (gh["reviewer"] / "config.yaml").write_text(json.dumps(
        {"kanban": {"pr_acceptance": {"gh_token_command": "gh auth token"}}}), encoding="utf-8")
    with connect() as conn:
        _, ok, receipts = _complete(conn)
    assert not ok and receipts[-1]["classification"] == "auth"
    assert "must be a list" in receipts[-1]["detail"] and gh["gh_calls"]() == []


@pytest.mark.platforms("posix")
def test_token_command_of_another_profile_is_never_used(gh):
    """Only the assignee's own config names its route: the launch profile's command and the
    ambient GH_TOKEN both stay out, so an assignee without a login fails as itself."""
    (gh["home"] / "config.yaml").write_text((gh["reviewer"] / "config.yaml").read_text(), encoding="utf-8")
    other = gh["home"] / "profiles" / "other"
    other.mkdir()
    (other / ".env").write_text("", encoding="utf-8")
    with connect() as conn:
        _, ok, receipts = _complete(conn, assignee="other")
    assert not ok and receipts[-1]["classification"] == "auth" and "'other'" in receipts[-1]["detail"]
    assert gh["helper_calls"]() == []
    (call,) = gh["gh_calls"]()
    assert call["env"]["GH_TOKEN"] is None and call["env"]["GH_CONFIG_DIR"] == str(other / "gh")


@pytest.mark.platforms("posix")
def test_inaccessible_repository_is_auth(gh):
    gh["graphql"] = {"data": {"repository": None}}
    with connect() as conn:
        _, ok, receipts = _complete(conn)
    assert not ok and receipts[-1]["classification"] == "auth" and "acme/repo" in receipts[-1]["detail"]


@pytest.mark.platforms("posix")
def test_unresolved_assignee_profile_is_auth(gh):
    with connect() as conn:
        _, ok, receipts = _complete(conn, assignee="ghost")
    assert not ok and receipts[-1]["classification"] == "auth" and "cannot be resolved" in receipts[-1]["detail"]
    assert gh["helper_calls"]() == [] and gh["gh_calls"]() == []


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("change", ["reassigned", "blocked", "contract changed"])
def test_card_changed_during_collection_gets_no_receipt_or_completion(gh, change):
    """Evidence read as one profile, for one run/status/contract, cannot complete the card after it
    moved. No supported command rewrites a bound contract, so a concurrent writer on this temp
    board stands in for one."""
    with connect() as conn:
        tid = kb.create_task(conn, title="publish", completion_contract="acme/repo", assignee="reviewer")

        def move():
            with connect() as rival:
                if change == "reassigned":
                    assert kb.assign_task(rival, tid, "other")
                elif change == "blocked":
                    assert kb.block_task(rival, tid, reason="human input")
                else:
                    with kb.write_txn(rival):
                        rival.execute("UPDATE tasks SET completion_contract=? WHERE id=?",
                                      ("https://github.com/acme/repo/pull/8", tid))
        gh["on_reread"] = move
        assert not kb.complete_task(conn, tid, result="done", metadata={"published_pr": PR})
        task = kb.get_task(conn, tid)
        assert task.status != "done" and any("/pulls/7" in p for p in gh["requests"])
        assert conn.execute("SELECT count(*) FROM task_events WHERE task_id=? AND kind IN ('pr_acceptance', 'completed')",
                            (tid,)).fetchone()[0] == 0


@pytest.mark.platforms("posix")
def test_non_admin_view_of_a_pinned_check_keeps_its_app_pin(gh):
    """The admin rule pins ``ci`` to app 9; the non-admin view repeats ``ci`` without a pin.
    Another app's failing ``ci`` run must not decide a requirement GitHub pins to app 9."""
    gh["base_ref"] = {"branchProtectionRule": {"requiresStatusChecks": True, "requiredStatusChecks": [
                          {"context": "ci", "app": {"databaseId": 9}}]},
                      "refUpdateRule": {"requiredStatusCheckContexts": ["ci"]}}
    gh["runs"] = [_run(), _run(id=6, app={"id": 2}, conclusion="failure")]
    with connect() as conn:
        tid, ok, receipts = _complete(conn)
        assert ok and kb.get_task(conn, tid).status == "done"
    assert receipts[-1]["required"] == [{"context": "ci", "app_id": 9}]


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("pull", [{"pr_state": "draft"}, {"pr_state": None}, {"pr_state": "closed", "merged": None}],
                         ids=["unknown state", "null state", "closed merge unknown"])
def test_final_pr_state_must_be_known(gh, pull):
    gh.update(pull)
    with connect() as conn:
        tid, ok, receipts = _complete(conn)
        assert not ok and kb.get_task(conn, tid).status != "done"
    assert receipts[-1]["classification"] in {"infra", "stale"}


# --- Ruleset and protection evidence parsed against GitHub's published shapes (REST `repository-rule-detailed`,
# GraphQL `RequiredStatusCheckDescription`): an active analysis rule, a malformed container or an unknown rule kind
# must never read as "no checks required".

def _status_rule(checks):
    return {"type": "required_status_checks", "parameters": {"required_status_checks": checks}}


@pytest.mark.platforms("posix")
def test_active_code_quality_rule_without_status_checks_is_not_no_ci(gh):
    gh["rules"] = [[{"type": "code_quality", "parameters": {"severity": "errors"}}]]
    with connect() as conn:
        tid, ok, receipts = _complete(conn)
        assert not ok and kb.get_task(conn, tid).status != "done"
    assert receipts[-1]["classification"] == "missing" and "code_quality" in receipts[-1]["detail"]
    assert not any("/pulls/7" in p for p in gh["requests"])


MALFORMED_RULES = {
    "status checks as an object": [[_status_rule({})]],
    "status checks as a string": [[_status_rule("")]],
    "status checks null": [[_status_rule(None)]],
    "status checks missing": [[{"type": "required_status_checks", "parameters": {}}]],
    "parameters missing": [[{"type": "required_status_checks"}]],
    "check without context": [[_status_rule([{"integration_id": 9}])]],
    "integration id as a string": [[_status_rule([{"context": "ci", "integration_id": "9"}])]],
    "rule type null": [[{"type": None}]],
    "rule type missing": [[{"parameters": {}}]],
    "rule type not a string": [[{"type": 5}]],
    "unrecognized rule type": [[{"type": "future_required_analysis", "parameters": {}}]],
    "rule not an object": [["required_status_checks"]],
}


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("rules", MALFORMED_RULES.values(), ids=MALFORMED_RULES.keys())
def test_malformed_or_unknown_ruleset_evidence_is_never_no_ci(gh, rules):
    gh["rules"] = rules
    with connect() as conn:
        tid, ok, receipts = _complete(conn)
        assert not ok and kb.get_task(conn, tid).status != "done"
    assert receipts[-1]["ok"] is False and receipts[-1]["classification"] == "infra"


VALID_NO_CI_RULES = {
    "exact empty status-check list": [[_status_rule([])]],
    "review and push-policy rules only": [[
        {"type": "pull_request", "parameters": {"required_approving_review_count": 1}},
        {"type": "deletion"}, {"type": "non_fast_forward"}, {"type": "required_linear_history"},
        {"type": "copilot_code_review", "parameters": {"review_on_push": True}}]],
    "several pages": [[{"type": "creation"}], [{"type": "required_signatures"}]],
}


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("rules", VALID_NO_CI_RULES.values(), ids=VALID_NO_CI_RULES.keys())
def test_valid_rules_without_ci_still_complete_after_the_final_reread(gh, rules):
    gh["rules"] = rules
    with connect() as conn:
        tid, ok, receipts = _complete(conn)
        assert ok and kb.get_task(conn, tid).status == "done"
    assert receipts[-1]["required"] == [] and any("/pulls/7" in p for p in gh["requests"])


@pytest.mark.platforms("posix")
def test_check_pinned_to_an_app_with_unknown_id_accepts_no_other_app(gh):
    """``app: {databaseId: null}`` is a restriction GitHub could not identify, not ``app: null``."""
    gh["base_ref"] = {"branchProtectionRule": {"requiresStatusChecks": True, "requiredStatusChecks": [
        {"context": "ci", "app": {"databaseId": None}}]}}
    gh["runs"] = [_run(app={"id": 123})]
    with connect() as conn:
        tid, ok, receipts = _complete(conn)
        assert not ok and kb.get_task(conn, tid).status != "done"
    assert receipts[-1]["classification"] == "missing" and "'ci'" in receipts[-1]["detail"]


@pytest.mark.platforms("posix")
def test_check_without_app_restriction_accepts_any_source(gh):
    gh["base_ref"] = {"branchProtectionRule": {"requiresStatusChecks": True, "requiredStatusChecks": [
        {"context": "ci", "app": None}]}}
    gh["runs"] = [_run(app={"id": 123})]
    with connect() as conn:
        tid, ok, receipts = _complete(conn)
        assert ok and kb.get_task(conn, tid).status == "done"
    assert receipts[-1]["required"] == [{"context": "ci", "app_id": None}]


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("ruleset_requires_ci", [False, True], ids=["alone", "with active ruleset"])
def test_disabled_protection_rule_requires_nothing_from_its_retained_list(gh, ruleset_requires_ci):
    """``requiresStatusChecks: false`` decides for THAT rule; an active ruleset still requires its checks."""
    gh["base_ref"] = {"branchProtectionRule": {"requiresStatusChecks": False, "requiredStatusChecks": [
        {"context": "legacy", "app": None}]}}
    if ruleset_requires_ci:
        gh["rules"] = [[_status_rule([{"context": "ci"}])]]
    with connect() as conn:
        tid, ok, receipts = _complete(conn)
        assert (kb.get_task(conn, tid).status == "done") is ok
    if ruleset_requires_ci:
        assert not ok and receipts[-1]["classification"] == "missing"
        assert receipts[-1]["required"] == [{"context": "ci", "app_id": None}]
    else:
        assert ok and receipts[-1]["required"] == []
