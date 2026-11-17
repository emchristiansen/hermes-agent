# Retained original13f reservation producer

`retained_worker_reservation_13f.py` is an exact executable historical fixture, not current production code. Its SHA256 is `e1fde2526f53c3a73a117713848c1b6851588378de038e64d40751d61cc07d68`.

Source: the complete original13f `gateway/session_worker_reservation.py` preserved at `/nix/store/ja33scr6af4072kxmm0j963swj2hd22p-frozen-mr27-30/preimages-13f/gateway/session_worker_reservation.py`. The fixture bytes are copied unchanged; that review-capsule path is provenance only, NOT a runtime dependency.

The owning regression `tests/hermes_state/test_retained_worker_child_lineage.py` loads this repository-relative file to produce real retained accepted opaque IDs, including a root whose ID spells another root's child. Using the current allocator would erase the historical condition. The test exercises current child/reservation/receipt/settlement consumers against those retained roots; it does not fabricate worker/admission rows or substitute retirement behavior.

Keep this exact fixture and its provenance in the declared source/test closure. Do not replace it with the current producer, skip the retention cases, or restore a dependency on this workstation's prior review artifacts.
