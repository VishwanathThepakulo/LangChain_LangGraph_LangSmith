# LangChain, LangGraph & LangSmith

Notes and examples for learning the LangChain ecosystem, including direct LLM calls, LangChain components, and LangGraph workflows.

## Overview

This repository is a learning workspace for exploring:

- **Raw LLM calls** using the Groq Python SDK.
- **LangChain** for model initialization and reusable LLM application components.
- **LangGraph** for building stateful workflows with nodes, edges, conditional routing, and checkpointing.
- **LangSmith** concepts for tracing, debugging, and evaluating LLM applications.

## Projects and examples

The repository may contain examples such as:

- `raw_llm/` — calling a Groq-hosted model directly.
- `langchain/` — calling models using LangChain.
- `langgraph/` — building a self-editing coding agent with tools.



## Requirements

Python 3.10+ is recommended. Install the required packages from the included requirements file:

```bash
pip install -r requirements.txt
```

## Environment setup

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

Keep your API key private. Do not commit `.env` or hard-code secrets in source files.

## Running the examples

Activate your virtual environment, install dependencies, and run the relevant Python script. For example:

```bash
python path/to/your_script.py
```

For Jupyter notebooks, open the `.ipynb` file in Jupyter Notebook or VS Code and run cells from top to bottom.

## Core concepts

### Raw LLM

Use the Groq SDK directly to send a prompt and receive a response. This approach gives direct access to the provider's API.

### LangChain

Use `init_chat_model()` and `model.invoke()` to call a chat model through LangChain's common interface. LangChain also provides prompts, output parsers, tools, and agent components.

### LangGraph

Use `StateGraph` to define a workflow:

```text
START → agent → [tool calls?]
                  ├─ Yes → tools → agent
                  └─ No  → END
```

Key concepts include:

- `StateGraph` — defines nodes and transitions.
- `START` and `END` — graph entry and exit points.
- `ToolNode` — executes tool calls requested by the model.
- `add_messages` — merges message updates into conversation state.
- `InMemorySaver` — stores checkpoints in memory while the process is running.

### LangSmith

LangSmith can help trace model calls and agent workflows, debug issues, and evaluate application quality. Configure it only when you need tracing or evaluation, and follow the current LangSmith documentation for environment variables and setup.

## Safety notes

- Review code before running it.
- A self-editing agent can overwrite files; keep backups and test on sample files first.
- Running arbitrary Python scripts can be risky. Avoid running untrusted code.
- Use path restrictions and sandboxing for production agents.

## Learning path

1. Call a model directly with the Groq SDK.
2. Repeat the same task with LangChain.
3. Learn how to define tools and connect them to a model.
4. Build a LangGraph workflow with conditional edges and a tool loop.
5. Explore LangSmith tracing and evaluation.


