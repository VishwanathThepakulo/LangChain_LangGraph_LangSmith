import os
import sys
import subprocess
from pathlib import Path
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("API Key not found")


@tool
def read_file(file_input: str) -> str:
    """Read and return the contents of an existing text file."""
    file_path = Path(file_input)
    if not file_path.is_file():
        return f"Error: '{file_input}' is not a file."
    return file_path.read_text(encoding="utf-8")


@tool
def write_file(file_input: str, content: str) -> str:
    """Write modified code to an existing file."""
    file_path = Path(file_input)
    file_path.write_text(content, encoding="utf-8")
    return "File written successfully"


@tool
def run_code(script_path: str) -> str:
    """Execute a Python script and return its output."""
    command = [sys.executable, script_path]
    if Path(script_path).name == "lg_prac.py":
        command.append("--test")
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=20
        )
    except subprocess.TimeoutExpired:
        return "Error: script timed out after 20 seconds"
    return (
        f"Exit code: {result.returncode}\n"
        f"Output: {result.stdout}\n"
        f"Errors: {result.stderr}"
    )


SYSTEM_PROMPT = """
You are a self-editing Python coding agent.

Project files:
- lg_tools.py: Contains tools and model configuration.

Rules:
1. Always inspect the existing code before editing.
2. When modifying tools, read lg_tools.py.
3. When modifying the graph, read lg_prac.py.
4. Never pass "." or a directory to read_file.
5. Always provide the exact file path.
6. If the file path is unknown, ask the user.
7. Edit only the requested file.
8. Run the modified code and check the output.
"""


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


tools = [read_file, write_file, run_code]

model = init_chat_model(
    "openai/gpt-oss-120b",
    model_provider="groq",
    api_key=api_key,
)
model_with_tools = model.bind_tools(tools)


# --- Graph nodes ---
def agent_node(state: AgentState):
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    response = model_with_tools.invoke(messages)
    return {"messages": [response]}


# --- Build graph ---
builder = StateGraph(AgentState)
builder.add_node("agent", agent_node)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tools_condition)  # -> "tools" or END
builder.add_edge("tools", "agent")                       # loop back after tool use

graph = builder.compile()



if __name__ == "__main__":
    if "--test" in sys.argv:
        print("Test mode: graph compiled OK")
        sys.exit(0)

    print("Agent ready. Type 'exit' to quit.")
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            break
        result = graph.invoke(
            {"messages": [HumanMessage(content=user_input)]},
            {"recursion_limit": 25},
        )
        print("\nAgent:", result["messages"][-1].content)
