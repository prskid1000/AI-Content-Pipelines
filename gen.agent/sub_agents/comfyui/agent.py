import datetime
import sys
from pathlib import Path

from google.adk.models.lite_llm import LiteLlm
# Add project root to Python path to support both Flask and ADK web command
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from google.adk.agents import Agent

MODEL = LiteLlm(model=f"lm_studio/nvidia_nemotron-3-nano-30b-a3b", api_key="sk-0", base_url="http://localhost:1234/v1")

def build_agent():
    return Agent(
        name="manager",
        model=MODEL,
        include_contents = 'none',
        description="Manager Agent. For Dividing tasks into smaller tasks based on available agents and tools, then delegating the tasks to the appropriate agents or tools.",
        instruction="""
        You are a Manager Agent (Professional Agent Management Specialist) who divides tasks into smaller tasks based on available agents and tools, then delegates the tasks to the appropriate agents or tools.

        __VERY_IMPORTANT__ Constraints:
            - You can only call tools and agents that are available. Do not call any imaginary tools or agents.
            - If we have got the desired result or not possible to compute the result further or too many repetitions then you need to exit the loop.
            - If you are not able to divide the task into smaller tasks based on available agents and tools then you need to exit the loop.
        """,
        sub_agents=[],
        tools=[])
