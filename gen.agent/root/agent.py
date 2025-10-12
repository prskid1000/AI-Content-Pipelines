import datetime
import sys
from pathlib import Path

# Add project root to Python path to support both Flask and ADK web command
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from google.adk.agents import Agent, LoopAgent, SequentialAgent
from google.adk.agents.llm_agent import ToolContext

from google.adk.models.lite_llm import LiteLlm

MODEL = LiteLlm(model=f"lm_studio/qwen3-30b-a3b-thinking-2507", api_key="sk-0", base_url="http://localhost:1234/v1")

def exit_loop(tool_context: ToolContext):
    tool_context.actions.escalate = True
    return {}

manager_agent = Agent(
    name="manager",
    model=MODEL,
    description="Manager Agent. For Dividing tasks into smaller tasks based on available agents and tools, then delegating the tasks to the appropriate agents or tools.",
    instruction="""
    You are a Manager Agent (Professional Agent Management Specialist) who divides tasks into smaller tasks based on available agents and tools, then delegates the tasks to the appropriate agents or tools.

    __VERY_IMPORTANT__ Constraints:
        - You can only call tools and agents that are available. Do not call any imaginary tools or agents.
        - If we have got the desired result or not possible to compute the result further or too many repetitions then you need to exit the loop.
        - If you are not able to divide the task into smaller tasks based on available agents and tools then you need to exit the loop.
    """,
    sub_agents=[],
    tools=[exit_loop])

loop_agent = LoopAgent(
    name="loop",
    max_iterations=3,
    sub_agents=[manager_agent])

# Create sequence_agent with the dynamic result_agent
root_agent = SequentialAgent(
    name="root",
    sub_agents=[loop_agent]
)