import datetime
import sys
from pathlib import Path

# Add project root to Python path to support both Flask and ADK web command
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from google.adk.agents import Agent, LoopAgent
from google.adk.agents.llm_agent import ToolContext
from google.adk.models.lite_llm import LiteLlm

MODEL = LiteLlm(model=f"lm_studio/qwen3-vl-30b-a3b-instruct", api_key="sk-0", base_url="http://localhost:1234/v1")

async def exit_loop(tool_context: ToolContext):
    """Exit the loop by setting both escalate flag and end_invocation."""
    # Set escalate on the event actions
    tool_context.actions.escalate = True
    return {"status": "exiting loop - escalate flags set"}

manager_agent = Agent(
    name="manager",
    model=MODEL,
    description="Manages user requests by answering directly or delegating tasks to available agents and tools.",
    instruction="""You are a capable AI agent that accomplishes tasks through reasoning, tool usage, and sub-agent orchestration.""",
    sub_agents=[],
    tools=[exit_loop])

root_agent = LoopAgent(
    name="root",
    max_iterations=12,
    sub_agents=[manager_agent])