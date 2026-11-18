"""Kanban task attachments.

- ``copies``: the attachment size cap and the capped, vetted copy of a declared completion artifact
  into a task's attachments (with its preservation error and destination naming).
"""

from hermes_cli.kanban_attachments.copies import (
    KANBAN_ATTACHMENT_MAX_BYTES,
    ArtifactPreservationError,
    VettedArtifact,
)

__all__ = ["KANBAN_ATTACHMENT_MAX_BYTES", "ArtifactPreservationError", "VettedArtifact"]
