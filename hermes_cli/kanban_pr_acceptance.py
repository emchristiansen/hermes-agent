"""Exact-head GitHub acceptance for explicitly declared PR tasks.

Network work happens outside SQLite transactions. The lifecycle owner persists
receipts only after rechecking the captured run/status/contract under its lock.

``gh`` runs as the card's assignee profile (``profile_home``), not the ambient
login: :func:`_gh_env` resolves that profile's own credentials/config for the
subprocess — a multi-profile host's default ``gh`` login cannot read another
org's private repos (#122689). A profile whose token is fetched at the point of
use (a secrets-manager wrapper) declares ``kanban.pr_acceptance.gh_token_command``
in its OWN config; :func:`_acceptance_env` runs it for this child only.
"""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from urllib.parse import quote

_REPO = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+")
_PR = re.compile(r"https://github\.com/([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)/pull/([1-9][0-9]*)")
# One printable, whitespace-free line: anything else is helper noise, not a token.
_TOKEN = re.compile(r"[!-~]+")
_TOKEN_COMMAND_TIMEOUT_SECONDS = 60
# Every rule kind the branch-rules endpoint returns (REST `repository-rule-detailed`), sorted into exactly one of:
# required_status_checks, parsed below; rules that gate a merge on CI or analysis results this gate cannot read from
# check-runs/statuses; and rules that never wait on CI. A kind in none of them is refused, so a rule GitHub adds later
# cannot read as "nothing required".
_UNEVALUABLE_CI_RULES = frozenset({
    "workflows",                    # named workflows must pass
    "code_scanning",                # code scanning tools must provide results
    "code_quality",                 # a code quality analysis must be done before merging
    "code_coverage",                # uploaded coverage below its threshold blocks merging
    "license_compliance_scanning",  # license compliance scanning results
    "required_deployments",         # environments must be successfully deployed to
})
_NOT_CI_RULES = frozenset({
    "creation", "update", "deletion", "non_fast_forward",          # who may create/update/delete/force-push refs
    "required_linear_history", "required_signatures",             # commit graph and signature policy
    "pull_request",                                                # review requirements
    "copilot_code_review",                                         # requests a review; does not gate on a result
    "merge_queue",                                                 # how merges happen; its checks are status checks
    "commit_message_pattern", "commit_author_email_pattern", "committer_email_pattern",
    "branch_name_pattern", "tag_name_pattern",                     # metadata patterns checked on push
    "file_path_restriction", "max_file_path_length", "file_extension_restriction", "max_file_size",  # push content
})


def validate_contract(value: str | None) -> str:
    if value is None or value == "local-only":
        return "local-only"
    if not isinstance(value, str) or not (_REPO.fullmatch(value) or _PR.fullmatch(value)):
        raise ValueError("completion_contract must be local-only, OWNER/REPO, or an exact GitHub PR URL")
    return value


def _api(endpoint: str, *, env: dict[str, str] | None, query: str | None = None,
         paginate: bool = False):
    command = ["gh", "api", endpoint, "--hostname", "github.com"]
    if query is not None:
        command += ["-f", "query=" + query]
    if paginate:
        command += ["--paginate", "--slurp"]
    try:
        result = subprocess.run(command, stdin=subprocess.DEVNULL, capture_output=True,
                                text=True, encoding="utf-8", errors="replace", timeout=30,
                                check=True, env=env)
    except subprocess.CalledProcessError as exc:
        # 401/403/404 = the login cannot see this repository (wrong profile identity
        # or missing grant), not a transient API failure. Persist only the status
        # code + endpoint, never gh's stderr (credentials/host details).
        denied = re.search(r"HTTP (40[134])", exc.stderr or "")
        if denied:
            raise _GateAuthError(f"HTTP {denied[1]} on {endpoint.split('?')[0]}") from None
        if exc.returncode == 4:  # gh's authentication-required exit: this profile has no login
            raise _GateAuthError(f"gh has no login for {endpoint.split('?')[0]}") from None
        raise
    value = json.loads(result.stdout)
    if isinstance(value, dict) and value.get("errors"):
        raise ValueError("GitHub returned incomplete GraphQL evidence")
    return value


class _GateAuthError(RuntimeError):
    """gh was refused at HTTP 401/403/404 (or GraphQL returned no repository):
    this profile's login cannot see the repo — an identity problem to fix, not
    an infrastructure blip to retry."""


class _UnevaluableRequirement(RuntimeError):
    """The repository requires CI this gate cannot evaluate: never read as "nothing required"."""


def _gh_env(profile_home: str | None) -> dict[str, str] | None:
    """Child env for ``gh``: the card's profile identity when one is resolvable.

    The completion boundary runs in the worker (assignee), the CLI, or a
    reviewer/dispatcher turn, so an ambient ``gh`` login is whichever process
    happened to call it (#122689). ``served_profile_child_env(inherit_credentials=True)``
    is the seam for "this child acts for that profile": it scrubs the launch
    profile's credential residue and overlays the target profile's own
    ``GH_TOKEN``/``GH_CONFIG_DIR`` (its ``.env`` + external secret sources).
    ``None`` keeps the ambient env — unassigned cards behave exactly as before.
    """
    if not profile_home:
        return None
    from tools.environments.local import _is_routed_home, hermes_subprocess_env, served_profile_child_env
    base = hermes_subprocess_env(inherit_credentials=True)
    routed = _is_routed_home(profile_home)
    if routed:
        # gh's config dir decides which login `gh api` uses, yet it is a path, not a
        # credential, so no scrub list sees it; the target's own value is overlaid from its .env.
        base.pop("GH_CONFIG_DIR", None)
    env = served_profile_child_env(base=base, target_home=profile_home, inherit_credentials=True)
    if routed and not (env.keys() & {"GH_TOKEN", "GITHUB_TOKEN", "GH_CONFIG_DIR"}):
        # HOME/XDG_CONFIG_HOME are still the launch process's: without a login of its own the
        # child would fall through to ~/.config/gh/hosts.yml — the ambient login. Pin gh's config
        # to a profile-owned dir so it fails "not logged in" (exit 4 -> auth) instead.
        env["GH_CONFIG_DIR"] = str(Path(profile_home) / "gh")
    return env


def _gh_token_command(profile_home: str) -> list[str] | None:
    """The assignee profile's own ``kanban.pr_acceptance.gh_token_command`` argv, or None.

    Read from THAT profile's config — never the completing process's — so only the identity
    that owns the card can name how its login is fetched. A managed worker serving its own
    profile reads its frozen session config, like every other config reader in that process.
    A configured but unreadable or malformed route is an identity failure, not "none"."""
    from agent.safe_worker_policy import worker_config_snapshot
    from tools.environments.local import _is_routed_home
    from utils import load_yaml_file_readonly

    snapshot = worker_config_snapshot()
    try:
        if snapshot is not None and not _is_routed_home(profile_home):
            config = snapshot
        else:
            path = Path(profile_home) / "config.yaml"
            config = load_yaml_file_readonly(path) if path.exists() else {}
    except Exception:  # health: allow BLE001 -- any parse/read failure is reported as this profile's auth fault
        raise _GateAuthError("assignee profile config.yaml is unreadable") from None
    section = (config or {}).get("kanban") if isinstance(config, dict) else None
    section = section.get("pr_acceptance") if isinstance(section, dict) else None
    command = section.get("gh_token_command") if isinstance(section, dict) else None
    if command in (None, []):
        return None
    if not isinstance(command, list) or not all(isinstance(a, str) and a for a in command):
        raise _GateAuthError("kanban.pr_acceptance.gh_token_command must be a list of non-empty strings")
    return command


def _point_of_use_token(command: list[str], env: dict[str, str], owner: str) -> str:
    """Run the profile's token command for *owner*; the token lives only in the gh child's env.

    The owner travels as data (``GH_ACCEPTANCE_OWNER``), never interpolated into argv. Helper
    stdout/stderr are never persisted: failures name only the exit status."""
    try:
        result = subprocess.run(command, text=True, encoding="utf-8", errors="replace",
                                stdin=subprocess.DEVNULL, capture_output=True, check=False,
                                timeout=_TOKEN_COMMAND_TIMEOUT_SECONDS,
                                env={**env, "GH_ACCEPTANCE_OWNER": owner})
    except (OSError, subprocess.SubprocessError):
        raise _GateAuthError("gh_token_command could not run") from None
    if result.returncode != 0:
        raise _GateAuthError(f"gh_token_command exited {result.returncode}")
    token = result.stdout.strip()
    if not _TOKEN.fullmatch(token):
        raise _GateAuthError("gh_token_command printed no usable token")
    return token


def _acceptance_env(profile_home: str | None, owner: str) -> dict[str, str] | None:
    """The env every ``gh`` read of one acceptance uses: :func:`_gh_env`, plus a point-of-use
    token when the assignee profile declares a token command. Fetched once per collection."""
    env = _gh_env(profile_home)
    command = _gh_token_command(profile_home) if profile_home else None
    if command is None:
        return env
    token = _point_of_use_token(command, env, owner)
    env = {k: v for k, v in env.items() if k != "GITHUB_TOKEN"}
    env["GH_TOKEN"] = token
    return env


def _assignee_profile_home(assignee: str | None) -> str | None:
    """Home whose ``gh`` login must read the contract repo — the assignee's, resolved
    exactly as the dispatcher resolves the worker's home — or None (unassigned) so the
    ambient login is used. An assigned card whose profile cannot be resolved is an
    identity failure (``auth``), never a silent fall-through to the ambient login."""
    if not assignee:
        return None
    from hermes_cli.profiles import normalize_profile_name, resolve_profile_env
    try:
        return resolve_profile_env(normalize_profile_name(assignee))
    except (FileNotFoundError, ValueError):
        raise _GateAuthError(f"assignee profile {assignee!r} cannot be resolved") from None


def _protection_requirements(base_ref) -> tuple[set[tuple[str, int | None]], set[str]]:
    """Branch-protection required checks from the PR's base ref: ``(pinned, contexts)``.

    ``branchProtectionRule`` is the admin view and carries each check's app pin; its
    ``requiresStatusChecks`` decides whether that rule requires checks at all, so a list it keeps while
    disabled requires nothing. ``refUpdateRule`` is the rule as enforced on a non-admin viewer and carries
    contexts only, so a login that cannot see one still sees the other. An absent base ref or a rule
    without its check list is incomplete evidence, never "nothing required"."""
    if not isinstance(base_ref, dict):
        raise ValueError("PR base branch is unavailable")
    pinned: set[tuple[str, int | None]] = set()
    rule = base_ref["branchProtectionRule"]
    if rule is not None:
        requires, checks = rule["requiresStatusChecks"], rule["requiredStatusChecks"]
        if type(requires) is not bool or not (isinstance(checks, list) or (checks is None and not requires)):
            raise ValueError("Incomplete branch protection evidence")
        if requires:
            pinned.update(_protection_check(r) for r in checks)
    contexts: set[str] = set()
    update_rule = base_ref["refUpdateRule"]
    if update_rule is not None:
        listed = update_rule["requiredStatusCheckContexts"]
        if not isinstance(listed, list) or not all(isinstance(c, str) for c in listed):
            raise ValueError("Incomplete branch protection evidence")
        contexts.update(listed)
    return pinned, contexts


def _protection_check(check) -> tuple[str, int | None]:
    """One ``RequiredStatusCheckDescription``: ``app: null`` accepts any source; an app whose database
    id is unknown still restricts the check to that app, so it can be neither matched nor waived."""
    if not isinstance(check, dict) or not isinstance(check["context"], str):
        raise ValueError("Incomplete branch protection evidence")
    app = check["app"]
    if app is None:
        return check["context"], None
    if not isinstance(app, dict):
        raise ValueError("Incomplete branch protection evidence")
    app_id = app["databaseId"]
    if app_id is None:
        raise _UnevaluableRequirement(f"check {check['context']!r} is restricted to an app whose id GitHub did not return")
    if type(app_id) is not int:
        raise ValueError("Incomplete branch protection evidence")
    return check["context"], app_id


def _ruleset_requirements(rules) -> set[tuple[str, int | None]]:
    """Required checks from every ruleset page; an empty or malformed page set is not "none"."""
    if not isinstance(rules, list) or not rules or not all(isinstance(page, list) for page in rules):
        raise ValueError("Incomplete branch ruleset evidence")
    required: set[tuple[str, int | None]] = set()
    for page in rules:
        for rule in page:
            kind = rule["type"] if isinstance(rule, dict) else None
            if kind == "required_status_checks":
                required.update(_ruleset_checks(rule))
            elif kind in _UNEVALUABLE_CI_RULES:
                raise _UnevaluableRequirement(kind)
            elif kind not in _NOT_CI_RULES:
                raise ValueError("Unrecognized branch rule evidence")
    return required


def _ruleset_checks(rule: dict) -> set[tuple[str, int | None]]:
    """``parameters.required_status_checks``: an array of ``{context, integration_id?}``. Any other
    shape, an empty container of the wrong type included, is incomplete evidence."""
    params = rule["parameters"]
    checks = params["required_status_checks"] if isinstance(params, dict) else None
    if not isinstance(checks, list):
        raise ValueError("Incomplete branch ruleset evidence")
    required = set()
    for check in checks:
        if not isinstance(check, dict) or not isinstance(check["context"], str):
            raise ValueError("Incomplete branch ruleset evidence")
        app_id = check.get("integration_id")
        if app_id is not None and type(app_id) is not int:
            raise ValueError("Incomplete branch ruleset evidence")
        required.add((check["context"], app_id))
    return required


def _collect_required_outcomes(repo: str, sha: str, required, receipt: dict, env) -> list[str]:
    pages = _api(f"repos/{repo}/commits/{sha}/check-runs?per_page=100&filter=latest",
                 paginate=True, env=env)
    runs = [run for page in pages for run in page["check_runs"]]
    if len({r["id"] for r in runs}) != pages[0]["total_count"]:
        raise ValueError("Incomplete check-run pagination")
    statuses = [{**s, "sha": sha} for page in _api(f"repos/{repo}/commits/{sha}/statuses?per_page=100",
                                                   paginate=True, env=env) for s in page]
    outcomes = []
    for context, app_id in sorted(required, key=str):
        matching = [r for r in runs if r["name"] == context and
                    (app_id in (None, -1) or r["app"]["id"] == app_id)]
        # A legacy status can satisfy an unpinned context, but never a check pinned to an app.
        legacy = [s for s in statuses if s["context"] == context] if app_id in (None, -1) else []
        selected = matching + ([max(legacy, key=lambda s: s["id"])] if legacy else [])
        if not selected:
            outcomes.append("missing")
            receipt["checks"].append({"name": context, "classification": "missing", "head_sha": sha})
        for check in selected:
            is_run = "conclusion" in check
            outcome = check.get("conclusion") if is_run else check["state"]
            classification = _classify(check, sha, outcome, is_run)
            outcomes.append(classification)
            receipt["checks"].append({"name": context, "id": check["id"],
                "url": check.get("html_url") or check.get("target_url"),
                "head_sha": check.get("head_sha", check.get("sha")),
                "classification": classification, "conclusion": outcome})
    return outcomes


def collect_acceptance(contract: str, published_pr: str | None,
                       assignee: str | None = None) -> dict:
    receipt = {"ok": False, "classification": "missing", "head_sha": None,
               "pr_url": published_pr, "checks": [],
               "recovery": "Fix required failures, rerun infrastructure checks or wait, then retry completion. "
                           "Use kanban_block if human input is needed; receipts remain on the task event log."}
    try:
        profile_home = _assignee_profile_home(assignee)
        declared = _PR.fullmatch(contract)
        url = contract if declared else published_pr
        match = _PR.fullmatch(url or "")
        if not match or (not declared and match[1] != contract) or (declared and published_pr and published_pr != contract):
            receipt["detail"] = "Supply metadata.published_pr matching the persisted completion contract."
            return receipt
        repo, number = match[1], int(match[2])
        receipt["pr_url"] = url
        owner, name = repo.split("/")
        env = _acceptance_env(profile_home, owner)
        query = (f'{{repository(owner:{json.dumps(owner)},name:{json.dumps(name)}){{pullRequest(number:{number:d}){{headRefOid baseRefName state\n'
                 '            baseRef{branchProtectionRule{requiresStatusChecks requiredStatusChecks{context app{databaseId}}}\n'
                 '            refUpdateRule{requiredStatusCheckContexts}}}}}')
        repository = _api("graphql", query=query, env=env)["data"]["repository"]
        if repository is None:
            # A private repo the login cannot read resolves to null, not an error.
            raise _GateAuthError(f"HTTP 404 on graphql {repo}")
        pr = repository["pullRequest"]
        sha, branch = pr["headRefOid"], pr["baseRefName"]
        receipt["head_sha"] = sha
        if not re.fullmatch(r"[0-9a-f]{40}", sha) or pr["state"] not in {"OPEN", "MERGED"}:
            raise ValueError("PR is closed or current head is unavailable")
        required, contexts = _protection_requirements(pr["baseRef"])
        required |= _ruleset_requirements(_api(f"repos/{repo}/rules/branches/{quote(branch, safe='')}?per_page=100",
                                               paginate=True, env=env))
        # A context the non-admin view repeats must keep its app pin, not gain an unpinned twin.
        required |= {(c, None) for c in contexts - {c for c, _ in required}}
        receipt["required"] = [{"context": c, "app_id": a} for c, a in sorted(required, key=str)]
        outcomes = _collect_required_outcomes(repo, sha, required, receipt, env) if required else []
        # Re-read after all pages: old-head successes are never transferable, and a PR with no
        # required checks is still accepted only for the head/base it was evaluated at.
        current = _api(f"repos/{repo}/pulls/{number}", env=env)
        if current["state"] not in {"open", "closed"}:
            raise ValueError("PR state is unavailable")
        if current["head"]["sha"] != sha or current["base"]["ref"] != branch or (current["state"] == "closed" and current.get("merged") is not True):
            receipt.update(classification="stale", detail="PR head/base changed while collecting evidence; retry.")
            return receipt
        if not required:
            receipt["detail"] = "Branch protection and rulesets require no checks on this base branch."
        receipt["classification"] = next((x for x in outcomes if x != "success"), "success")
        receipt["ok"] = receipt["classification"] == "success"
        return receipt
    except _GateAuthError as exc:
        login = f"assignee profile {assignee!r}'s gh login" if assignee else "the ambient gh login"
        receipt.update(classification="auth",
                       detail=f"GitHub refused the acceptance read ({exc}) as {login}; "
                              "fix that profile's GitHub credentials/access to the repository, then retry completion.")
        return receipt
    except _UnevaluableRequirement as exc:
        receipt.update(classification="missing",
                       detail=f"The base branch requires CI this gate cannot evaluate ({exc}); "
                              "complete after confirming it on GitHub through a supported route.")
        return receipt
    except (OSError, subprocess.SubprocessError, ValueError, KeyError, TypeError, IndexError):
        # Never persist gh stderr (credentials/host details); the failed phase is actionable.
        receipt.update(classification="infra", detail="GitHub acceptance evidence unavailable or incomplete; check gh authentication/API access and retry.")
        return receipt


def _classify(check: dict, sha: str, outcome: str | None, is_run: bool) -> str:
    if check.get("head_sha", check.get("sha")) != sha:
        return "stale"
    if is_run and check.get("status") != "completed":
        return "pending"
    return {"success": "success", "failure": "failure", "error": "infra", "pending": "pending"}.get(outcome, "infra")
