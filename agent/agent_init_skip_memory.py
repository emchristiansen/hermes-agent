"""A ``--skip-memory`` session at agent init: drop the built-in memory store and tool the toolset would load."""


def apply_session_skip_memory(agent) -> None:
    """A ``--skip-memory`` session's agent (``SessionPolicy.skip_memory_requested``, the CLI flag): no memory at all.

    ``skip_memory`` at construction skips only the external provider; a requested or inherited ``memory``
    toolset still gets the built-in store in ``_init_memory`` (#65429), which loads MEMORY.md/USER.md into the
    prompt. A session that opted out drops that store and the memory tool that would run against it, before the
    first turn builds the system prompt. Delegate children inherit the opt-out (``tools/delegate_tool.py``), and
    compaction takes no memory checkpoint."""
    agent._session_skip_memory = True
    from tools.skills_tool import suppress_memory_provider_loads_for
    suppress_memory_provider_loads_for(getattr(agent, "session_id", None), getattr(agent, "task_id", None))
    agent.compression_checkpoint_required = False
    agent._memory_store = None
    agent._memory_enabled = agent._user_profile_enabled = False
    tools = getattr(agent, "tools", None)
    if tools:
        agent.tools = [tool for tool in tools if (tool.get("function") or {}).get("name") != "memory"]
        agent.valid_tool_names = {tool["function"]["name"] for tool in agent.tools}
    agent._cached_system_prompt = agent._cached_system_prompt_static = None


def carry_session_skip_memory(agent) -> None:
    """After ``agent.session_id`` changed (rotating compaction, live-child adoption): an explicitly memoryless agent's
    opt-out binds its new id too, so a dispatch naming only that id stays suppressed (SKM-06)."""
    if getattr(agent, "_session_skip_memory", False) is True:
        from tools.skills_tool import suppress_memory_provider_loads_for
        suppress_memory_provider_loads_for(getattr(agent, "session_id", None))
