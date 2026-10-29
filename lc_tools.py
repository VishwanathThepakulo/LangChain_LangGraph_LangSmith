import os 
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.tools import tool 
from pathlib import Path
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent
import subprocess
import sys

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
     raise ValueError("API KEY not found")

@tool
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
            "such as lc_tools.py or lg_tools.py."
        )

    if not file_path.is_file():
        raise ValueError("Path is not a valid file.")

    return file_path.read_text(encoding="utf-8")
    
    
@tool
def write_file(file_input: str, content: str) -> str:
    """Write modified code to an existing file."""
    file_path = Path(file_input)

    if not file_path.exists():
        raise ValueError("File does not exist")

    if not file_path.is_file():
        raise ValueError("Path is not a file")

    file_path.write_text(content, encoding="utf-8")
    return "File updated successfully"

@tool
def run_code(script_path):
    """Execute a Python script and return its output."""
    command = [sys.executable, script_path]
    if Path(script_path).name == "lc_tools.py":
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
- lc_tools.py: Contains tools and model configuration.

Rules:
1. Always inspect the existing code before editing.
2. When modifying tools, read lc_tools.py.
3. When modifying the graph, read lg_tools.py.
4. Never pass "." or a directory to read_file.
5. Always provide the exact file path.
6. If the file path is unknown, ask the user.
7. Edit only the requested file.
8. Run the modified code and check the output.
"""

def build_agent():
    model = init_chat_model(
    'openai/gpt-oss-120b',
    model_provider='groq',
    api_key=api_key,
    temperature=0.7
    )
    checkpointer = InMemorySaver()
    return create_agent(
        model=model,
        tools=[read_file, write_file, run_code],
        system_prompt=SYSTEM_PROMPT,
        checkpointer=checkpointer
    )


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        build_agent()
        print("OK: lc_tools.py loaded and agent built, no errors")
        return
    agent = build_agent()
    config = {"configurable": {"thread_id": "lc-demo-1"}}
    user_text = input("Enter a prompt : ")
    result = agent.invoke(
        {"messages": [{"role": "user", "content": user_text}]},
        config=config,
    )
    print(result["messages"][-1].content)

if __name__ == "__main__":
    main()
