"""The summary-model bench (fallback streak >= 2) and its persisted probe window."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent.context_compressor import ContextCompressor
from agent.conversation_compression import recover_rotated_compression_session
from tests.fixtures.storage_seed import seed_messages, seed_session
from tests.gateway.fixtures.storage_handles import close_storage_handles, storage_handle


def _compressor():
    with patch("agent.context_compressor.get_model_context_length", return_value=100000):
        c = ContextCompressor(model="test/model", threshold_percent=0.85, protect_first_n=2, protect_last_n=2, quiet_mode=True)
        _ = c.context_length
        return c


def _messages(n_pairs=10):
    msgs = [{"role": "system", "content": "system prompt"}]
    for i in range(n_pairs):
        msgs += [{"role": "user", "content": f"question {i}"}, {"role": "assistant", "content": "short reply"}]
    return msgs


def test_bench_probe_deadline_survives_a_compressor_rebuild(tmp_path):
    """The gateway rebuilds the agent on cache eviction: a fresh compressor bound to the same
    session must resume the persisted bench window, not restart it (else the probe starves)."""
    db = storage_handle(tmp_path)
    seed_session(db, "s1", "cli")
    db.lineage.set_compression_fallback_streak("s1", 2)
    compressor = _compressor()
    compressor.bind_session_state(db, "s1", persistence=db)
    window = compressor._ANTI_THRASH_RECOVERY_SECONDS

    with patch("agent.context_compressor.time.time", return_value=1000.0), \
            patch.object(compressor, "_generate_summary", return_value="LLM summary") as mock_gen:
        compressor.compress(_messages(), force=False)  # benched, arms the window
    mock_gen.assert_not_called()

    rebuilt = _compressor()
    rebuilt.bind_session_state(db, "s1", persistence=db)
    with patch("agent.context_compressor.time.time", return_value=1000.0 + window + 1), \
            patch.object(rebuilt, "_generate_summary", return_value="LLM summary") as mock_gen:
        rebuilt.compress(_messages(), force=False)
    mock_gen.assert_called_once()

    # Rotating compaction (in_place: false) binds a fresh child row: the deadline must follow it.
    with patch("agent.context_compressor.time.time", return_value=2000.0), \
            patch.object(rebuilt, "_generate_summary", return_value="LLM summary"):
        rebuilt.compress(_messages(), force=False)  # benched again, re-arms the window
    seed_session(db, "s2", "cli", parent_session_id="s1")
    rebuilt.on_session_start("s2", boundary_reason="compression", old_session_id="s1", session_db=db, persistence=db)
    with patch("agent.context_compressor.time.time", return_value=2000.0 + window + 1), \
            patch.object(rebuilt, "_generate_summary", return_value="LLM summary") as mock_gen:
        rebuilt.compress(_messages(), force=False)
    mock_gen.assert_called_once()


@pytest.mark.parametrize("child_recovered", [False, True], ids=["deadline-armed", "already-recovered"])
def test_adopting_a_live_child_resumes_its_summary_probe(tmp_path, child_recovered):
    db = storage_handle(tmp_path)
    # The old agent either never saw the child's deadline or still remembers a bench the child has cleared.
    deadline = 1000.0 + ContextCompressor._ANTI_THRASH_RECOVERY_SECONDS
    seed_session(db, "parent", "cli", model_config={"_fallback_probe_at": deadline} if child_recovered else {})
    db.lineage.set_compression_fallback_streak("parent", 2)
    contender = _compressor()
    contender.bind_session_state(db, "parent", persistence=db)
    agent = SimpleNamespace(
        session_id="parent", _session_db=db, persistence=db, context_compressor=contender, _memory_manager=None,
    )

    db.sessions.end("parent", "compression")
    seed_session(db, "child", "cli", parent_session_id="parent",
                 model_config={} if child_recovered else {"_fallback_probe_at": deadline})
    seed_messages(db, "child", {"role": "user", "content": "compacted history"})
    if not child_recovered:
        db.lineage.set_compression_fallback_streak("child", 2)

    assert recover_rotated_compression_session(agent)
    assert agent.session_id == "child"
    expected_deadline = None if child_recovered else deadline
    assert db.sessions.get_model_config_value("child", "_fallback_probe_at") == expected_deadline
    with patch("agent.context_compressor.time.time", return_value=deadline - 1), \
            patch.object(contender, "_generate_summary", return_value="LLM summary") as summarize:
        contender.compress(_messages())
    assert summarize.call_count == int(child_recovered)

    rebuilt = _compressor()
    rebuilt.bind_session_state(db, "child", persistence=db)
    with patch("agent.context_compressor.time.time", return_value=deadline), \
            patch.object(rebuilt, "_generate_summary", return_value="LLM summary") as summarize:
        rebuilt.compress(_messages())
    summarize.assert_called_once()
