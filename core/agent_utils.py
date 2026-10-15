from typing import Any

from langchain.agents import create_agent

from core.audit import AuditLogger
from core.observability import AgentTrace


def invoke_agent(
    agent_name: str,
    model: Any,
    tools: list[Any],
    system_prompt: str,
    task: str,
    run_id: str,
    inputs: dict[str, Any],
) -> dict[str, Any]:
    trace = AgentTrace(agent_name, run_id).set_input(**inputs)
    audit = AuditLogger(run_id)
    audit.log(agent_name, "start", inputs=inputs)

    try:
        agent = create_agent(model=model, tools=tools, system_prompt=system_prompt)
        result = agent.invoke({"messages": [{"role": "user", "content": task}]})
        messages = result.get("messages", [])
        trace.extract_from_messages(messages)
        final_message = getattr(messages[-1], "content", "") if messages else ""
        trace.set_output(final_message=final_message, message_count=len(messages)).complete()
        audit.log(agent_name, "complete", message_count=len(messages))
        return result
    except Exception as error:
        trace.fail(error)
        audit.log(agent_name, "fail", error=str(error))
        raise