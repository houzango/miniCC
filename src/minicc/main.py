import os
import subprocess
from pathlib import Path
from typing import cast

from anthropic import Anthropic
from anthropic.types import ToolParam, ToolUseBlock
from dotenv import load_dotenv

try:
    import readline

    # #143 UTF-8 backspace fix for macOS libedit
    readline.parse_and_bind("set bind-tty-special-chars off")
    readline.parse_and_bind("set input-meta on")
    readline.parse_and_bind("set output-meta on")
    readline.parse_and_bind("set convert-meta off")
except ImportError:
    pass

# -- Load environment variables --
load_dotenv(override=True)

# -- force the third-party proxy to use api_key instead of auth_token --
if os.getenv("ANTHROPIC_BASE_URL"):
    os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)

# -- initialize the Anthropic client --
client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))

MODEL = os.environ["MODEL_ID"]
SYSTEM = f"You are a coding agent at {os.getcwd()}. Use bash to solve tasks. Act, don't explain."

# -- Tool definition --
TOOLS: list[ToolParam] = [
    {
        "name": "bash",
        "description": "Run a shell command.",
        "input_schema": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    },
    # {
    # "name": "read",
    # "description": "Read the contents of a file.",
    # "input_schema": {
    #     "type": "object",
    #     "properties": {"file_path": {"type": "string"}},
    #     "required": ["file_path"],
    #     },
    # },
    # {
    # "name": "write",
    # "description": "Write content to a file.",
    # "input_schema": {
    #     "type": "object",
    #     "properties": {"file_path": {"type": "string"}, "content": {"type": "string"}},
    #     "required": ["file_path", "content"],
    #     },
    # },
]


# -- Tool execution --
def run_bash(command: str) -> str:
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=os.getcwd(),
            capture_output=True,
            text=True,
            timeout=120,
        )
        result_formatted = []
        if result.stdout:
            result_formatted.append(f"      stdout:{result.stdout.strip()}")
        if result.stderr:
            result_formatted.append(f"      stderr:{result.stderr.strip()}")
        result_formatted = (
            "\n".join(result_formatted) if result_formatted else "      no output"
        )
        return result_formatted[:50000]
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"
    except (FileNotFoundError, OSError) as e:
        return f"Error: {e}"


def run_read(file_path: str) -> str:
    try:
        with open(file_path, "r") as file:
            return file.read()
    except FileNotFoundError:
        return f"Error: File not found: {file_path}"
    except OSError as e:
        return f"Error: {e}"


def run_write(file_path: str, content: str) -> str:
    try:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w") as file:
            file.write(content)
        return "Successfully wrote to file"
    except OSError as e:
        return f"Error: {e}"


# -- Print --
def print_response(response_content):
    """
    Args:
        response_content: list of content_block_objects from LLM response
    """
    print("\033[33mLLM RESPONSE: \033[0m")

    for block in response_content:
        block_type = getattr(block, "type", None)

        if block_type == "text":
            print(f"[text] {block.text}")

        elif block_type == "thinking":
            print(f"[thinking] {block.thinking}")

        elif block_type == "tool_use":
            print("[tool_call]")
            print(f"{block.name} (id: {block.id})")
            print(f"input: {block.input}")

    print()


# -- The core pattern: a while loop that calls tools --
def agent_loop(messages: list):

    while True:
        response = client.messages.create(
            model=MODEL,
            system=SYSTEM,
            messages=messages,
            tools=TOOLS,
            max_tokens=8000,
        )

        # response.content is a list of content_block_objects
        print_response(response.content)
        # Add the assistant's response to the messages
        messages.append({"role": "assistant", "content": response.content})

        tool_calls: list[ToolUseBlock] = [
            block for block in response.content if block.type == "tool_use"
        ]

        # If there are no tool calls, loop ends
        if not tool_calls:
            return

        # If there are tool calls, execute them, collect results
        results = []
        print("\033[33mTOOL EXECUTION: \033[0m")
        for tool_block in tool_calls:
            print(f"tool_id: {tool_block.id})")
            print(f"[bash]command: {tool_block.input['command']}")
            result = run_bash(cast(str, tool_block.input["command"]))
            print(f"[bash]result: \n{result}")
            results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": tool_block.id,
                    "content": result.strip(),
                }
            )
        print()

        # Add the tool results to the messages, loop continues
        messages.append({"role": "user", "content": results})


def main() -> None:
    print("\n\033[36mInput a prompt, press Enter to send. Type q to quit.\033[0m\n")

    messages = []
    while True:
        try:
            query = input("\033[36minput >> \033[0m")
        except EOFError, KeyboardInterrupt:
            return
        if query.strip().lower() in ("q", "quit", "exit", ""):
            return
        messages.append({"role": "user", "content": query})
        agent_loop(messages)


# -- Entry point --
if __name__ == "__main__":
    main()
