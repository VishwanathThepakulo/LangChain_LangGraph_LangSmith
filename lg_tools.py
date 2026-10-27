import sys
import os 
from typing import Annotated, Literal
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from pathlib import Path
from langgraph.checkpoint.memory import InMemorySaver
import subprocess

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
     raise ValueError("API KEY not found")

def read_file(file_input: str) -> str:
    """Read and return the contents of an existing text file."""
    file_path = Path(file_input)

    if not file_path.exists():
        raise ValueError(
            f"Error: The file at '{file_path}' does not exist."
        )

    if file_path.is_dir():
        raise ValueError(
            f"'{file_path}' is a directory. "
            "Please provide a specific file path, "
            "such as lg_tools.py."
        )

    if not file_path.is_file():
        raise ValueError("Path is not a valid file.")

    return file_path.read_text(encoding="utf-8")
    
def write_file(file_input: str, content: str) -> str:
    """Write modified code to an existing file."""
    file_path = Path(file_input)

    if not file_path.exists():
        raise ValueError("File does not exist")

    if not file_path.is_file():
        raise ValueError("Path is not a file")

    file_path.write_text(content, encoding="utf-8")
    return "File updated successfully"

def run_code(script_path):
    """Execute a Python script and return its output."""
    command = [sys.executable, script_path]
    if Path(script_path).name == "lg_tools.py":
        command.append("--test")
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=20
    )
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
3. When modifying the graph, read lg_tools.py.
4. Never pass "." or a directory to read_file.
5. Always provide the exact file path.
6. If the file path is unknown, ask the user.
7. Edit only the requested file.
8. Run the modified code and check the output.
"""

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

def build_graph():
    model = init_chat_model(
        "openai/gpt-oss-120b",
        model_provider="groq",
        api_key=api_key
    )

    from langchain.tools import tool

    @tool
    def read_file_tool(file_path: str) -> str:
        """Read the contents of an existing Python file."""
        return read_file(file_path)

    @tool
    def write_file_tool(file_path: str, content: str) -> str:
        """Write modified code to an existing file."""
        return write_file(file_path, content)

    @tool
    def run_code_tool(script_path: str) -> str:
        """Run a Python script and return its output."""
        return run_code(script_path)

    # Removed addition_tool as per request.

    tools = [
        read_file_tool,
        write_file_tool,
        run_code_tool,
        # addition_tool,  # Removed
    ]

    model_with_tools = model.bind_tools(tools)

    def agent_node(state: AgentState):
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            *state["messages"],
        ]
        response = model_with_tools.invoke(messages)
        return {"messages": [response]}

    def should_continue(state: AgentState) -> Literal["tools", END]:
        last_message = state["messages"][-1]
        if last_message.tool_calls:
            return "tools"
        return END

    graph_builder = StateGraph(AgentState)
    graph_builder.add_node("agent", agent_node)
    graph_builder.add_node("tools", ToolNode(tools))

    graph_builder.add_edge(START, "agent")
    graph_builder.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            END: END,
        },
    )
    graph_builder.add_edge("tools", "agent")

    return graph_builder.compile(checkpointer=InMemorySaver())

def main() -> None:
    agent = build_graph()
    config = {"configurable": {"thread_id": "self-editing-agent"}}

    print("Self-Editing Agent is ready!")
    print("Type 'exit' to stop.\n")

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() in {"exit", "quit"}:
            print("Agent stopped.")
            break

        if not user_input:
            continue

        result = agent.invoke(
            {
                "messages": [
                    HumanMessage(content=user_input)
                ]
            },
            config=config,
        )

        print("\nAgent:", result["messages"][-1].content)
        print()

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        build_graph()
        print("OK: lg_tools.py loaded and graph built, no errors")
    else:
        main()
