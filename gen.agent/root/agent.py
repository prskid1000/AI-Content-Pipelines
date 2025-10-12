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

MODEL = LiteLlm(model=f"lm_studio/qwen3-30b-a3b-thinking-2507", api_key="sk-0", base_url="http://localhost:1234/v1")

async def exit_loop(tool_context: ToolContext):
    """Exit the loop by setting both escalate flag and end_invocation."""
    # Set escalate on the event actions
    tool_context.actions.escalate = True
    return {"status": "exiting loop - escalate flags set"}

manager_agent = Agent(
    name="manager",
    model=MODEL,
    description="Manager Agent. For Dividing tasks into smaller tasks based on available agents and tools, then delegating the tasks to the appropriate agents or tools.",
    instruction="""
    You are a Manager Agent (Professional Agent Management Specialist) who divides tasks into smaller tasks based on available agents and tools, then delegates the tasks to the appropriate agents or tools.

    __VERY_IMPORTANT__ Constraints:
        - You can only call tools and agents that are available. Do not call any imaginary tools or agents.
        - When you have completed the task, obtained the desired result, or cannot make further progress, you MUST call the exit_loop tool to stop the loop.
        - If you are not able to divide the task into smaller tasks based on available agents and tools, you MUST call the exit_loop tool.
        - ALWAYS call exit_loop when done, otherwise the loop will continue indefinitely.
        - After calling exit_loop, provide a brief summary of what was accomplished.
    """,
    sub_agents=[],
    tools=[exit_loop])

root_agent = LoopAgent(
    name="root",
    max_iterations=5,  # Increased to test escalate behavior
    sub_agents=[manager_agent])