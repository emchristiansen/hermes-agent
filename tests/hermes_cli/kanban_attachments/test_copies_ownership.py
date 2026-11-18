"""The completion-artifact copies are owned by hermes_cli/kanban_attachments/copies.py (review22981 KB-EXTRACT-01).

Their consumers import them from there, and kanban_db reads them there instead of keeping its own
re-exported bindings, so a patch of the owning module is the one every caller sees and a patch of a
stale kanban_db name fails loudly rather than silently missing.
"""
from hermes_cli.kanban_attachments import copies

MOVED = ("KANBAN_ATTACHMENT_MAX_BYTES", "ArtifactPreservationError", "VettedArtifact", "_copy_capped",
         "_copy_vetted_artifact", "_discard_staged_copies", "_unique_attachment_path")


def test_kanban_db_keeps_no_re_exported_binding_of_a_moved_name():
    from hermes_cli import kanban_db

    assert [name for name in MOVED if hasattr(kanban_db, name)] == []
    assert kanban_db._attachments is copies


def test_consumers_bind_the_owning_module():
    from hermes_cli import kanban, kanban_db_settle
    from tools import kanban_tools

    assert kanban.ArtifactPreservationError is copies.ArtifactPreservationError
    assert kanban_db_settle.ArtifactPreservationError is copies.ArtifactPreservationError
    assert kanban_db_settle._discard_staged_copies is copies._discard_staged_copies
    assert kanban_tools.ArtifactPreservationError is copies.ArtifactPreservationError
    assert kanban_tools.VettedArtifact is copies.VettedArtifact
    assert kanban_tools.KANBAN_ATTACHMENT_MAX_BYTES == copies.KANBAN_ATTACHMENT_MAX_BYTES


def test_a_patch_of_the_owning_module_reaches_the_completion_copy(tmp_path, monkeypatch):
    """The completion path copies through ``copies._copy_capped`` as looked up at call time."""
    from hermes_cli import kanban_db

    seen = []
    monkeypatch.setattr(copies, "_copy_capped", lambda src, dest, artifact: seen.append(artifact))
    monkeypatch.setattr(kanban_db, "_scratch_workspace", lambda conn, task_id: tmp_path)
    monkeypatch.setattr(kanban_db, "_managed_scratch_path_info", lambda path: (True, None))
    monkeypatch.setattr(kanban_db, "_is_managed_scratch_path", lambda path: True)
    monkeypatch.setattr(kanban_db, "task_attachments_dir", lambda task_id, board=None: tmp_path / "attachments")
    report = tmp_path / "REPORT.md"
    report.write_text("report\n")

    class Conn:
        def execute(self, *_args):
            return []

    metadata = {"artifacts": [str(report)]}
    kanban_db._persist_scratch_completion_artifacts(Conn(), "t_1", metadata)
    assert seen == [str(report)]
