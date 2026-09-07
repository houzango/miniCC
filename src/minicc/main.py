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

# -- Load .env file --
load_dotenv(override=True)

# -- initialize the Anthropic client --
client = Anthropic(
    base_url=os.getenv("ANTHROPIC_BASE_URL"),
    api_key=os.environ["ANTHROPIC_API_KEY"],
)

MODEL = os.environ["MODEL_ID"]
WORKDIR = Path.cwd()
SYSTEM = f"You are a coding agent at {WORKDIR}. Use tools to solve tasks. Act, don't explain."

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
    {
        "name": "read_file",
        "description": "Read the contents of a file.",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string"},
                # limit == max lines to return
                "limit": {"type": "integer"},
            },
            "required": ["file_path"],
        },
    },
    {
        "name": "write_file",
        "description": "Write content to a file.",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["file_path", "content"],
        },
    },
    {
        "name": "edit_file",
        "description": "Replace only the first exact occurrence of old_text with new_text in a file.",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string"},
                "old_text": {"type": "string"},
                "new_text": {"type": "string"},
            },
            "required": ["file_path", "old_text", "new_text"],
        },
    },
    {
        "name": "glob",
        "description": "Find files matching a glob pattern.",
        "input_schema": {
            "type": "object",
            "properties": {"pattern": {"type": "string"}},
            "required": ["pattern"],
        },
    },
]


# -- Tool execution --
def run_bash(command: str) -> str:
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=WORKDIR,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,
        )
        result_formatted = []
        if result.stdout:
            result_formatted.append(f"stdout:{result.stdout.strip()}")
        if result.stderr:
            result_formatted.append(f"stderr:{result.stderr.strip()}")
        result_formatted = (
            "\n".join(result_formatted) if result_formatted else "(No output.)"
        )
        return result_formatted[:50000]
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"
    except Exception as e:
        return f"Error: {e}"


def run_read(file_path: str, limit: int | None = None) -> str:
    try:
        lines = (WORKDIR / file_path).resolve().read_text().splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more lines)"]
        return "\n".join(lines)
    except Exception as e:
        return f"Error: {e}"


def run_write(file_path: str, content: str) -> str:
    try:
        path = (WORKDIR / file_path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        return f"Successfully wrote {len(content.splitlines())} lines to {file_path}"
    except Exception as e:
        return f"Error: {e}"


def run_edit(file_path: str, old_text: str, new_text: str) -> str:
    try:
        path = (WORKDIR / file_path).resolve()
        text = path.read_text()
        if old_text not in text:
            return f"Error: text not found in {file_path}"
        path.write_text(text.replace(old_text, new_text, 1))
        return f"{file_path}: File edited."
    except Exception as e:
        return f"Error: {e}"


# glob is a module that finds files matching a pattern in the working directory
def run_glob(pattern: str) -> str:
    import glob as g

    try:
        results = []
        for match in g.glob(pattern, root_dir=WORKDIR, recursive=True):
            # pattern may contain ../ or match symlinks pointing outside the working directory
            if (WORKDIR / match).resolve().is_relative_to(WORKDIR):
                results.append(match)
        return "\n".join(results) if results else "(No matches found.)"
    except Exception as e:
        return f"Error: {e}"


TOOL_HANDLERS = {
    "bash": run_bash,
    "read_file": run_read,
    "write_file": run_write,
    "edit_file": run_edit,
    "glob": run_glob,
}

# -- Hook system --
HOOKS = {
    "PostUserSubmit": [],
    "PostModelResponse": [],
    "PreToolUse": [],
    "PostToolUse": [],
    "PreLoopEnd": [],
}


def register_hooks(event: str, *callbacks):
    HOOKS[event].extend(callbacks)


def trigger_hooks(event: str, *args):
    for callback in HOOKS[event]:
        result = callback(*args)
        if result is not None:
            return result
    # None means don't block and let it through
    return None


DENY_LIST = ["rm -rf /", "sudo", "shutdown", "reboot", "mkfs", "dd if="]
DESTRUCTIVE = ["rm ", "> /etc/", "chmod 777"]


def tool_permission_hook(block):
    """PreToolUse: permission pipeline."""
    if block.name == "bash":
        for pattern in DENY_LIST:
            if pattern in block.input.get("command", ""):
                print(
                    f"\033[31m[{block.name}]result: '{pattern}' is blocked by deny list\033[0m"
                )
                return f"'{pattern}' is blocked by deny list"
        for kw in DESTRUCTIVE:
            if kw in block.input.get("command", ""):
                print(
                    f"\033[33m[permission]Potentially destructive command: {kw}\033[0m"
                )
                choice = input("   Allow? [y/N] ").strip().lower()
                if choice not in ("y", "yes"):
                    print(f"[{block.name}]result: Permission denied by user")
                    return f"Permission denied by user for destructive command: '{kw}'"
    if block.name in ("read_file", "write_file", "edit_file"):
        ture_path = (WORKDIR / block.input.get("file_path", "")).resolve()
        if not ture_path.is_relative_to(WORKDIR):
            print(f"\033[33m[permission]Path escapes workspace: {ture_path}\033[0m")
            choice = input("   Allow? [y/N] ").strip().lower()
            if choice not in ("y", "yes"):
                print(f"[{block.name}]result: Permission denied by user")
                return "Permission denied by user for path escapes workspace"
    return None


def tool_log_hook(block):
    """PreToolUse: log every tool call."""
    print(f"tool_id: {block.id})")
    print(f"[{block.name}]input: {block.input}")
    return None


def tool_result_print_hook(block, result):
    """PostToolUse: print the tool result."""
    print(f"[{block.name}]result: \n{result}")
    return None


def cwd_display_hook(query: str):
    """PostUserSubmit: display the current working directory."""
    print()
    print(f"\033[90m[WORKDIR]: {WORKDIR}\033[0m")
    print()
    return None


def model_response_print_hook(response_content):
    """
    PostModelResponse: print the response content

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
    return None


def tool_summary_hook(messages: list):
    """PreLoopEnd: print a summary of the session."""
    tool_count = 0
    for m in messages:
        content = m.get("content")
        if isinstance(content, list):
            for r in content:
                if isinstance(r, dict) and r.get("type") == "tool_result":
                    tool_count += 1
    print(f"\033[90m[tool_summary]: session used {tool_count} tool calls\033[0m")
    print()
    return None


register_hooks("PostUserSubmit", cwd_display_hook)
register_hooks("PostModelResponse", model_response_print_hook)
register_hooks("PreToolUse", tool_log_hook, tool_permission_hook)
register_hooks("PostToolUse", tool_result_print_hook)
register_hooks("PreLoopEnd", tool_summary_hook)


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
        trigger_hooks("PostModelResponse", response.content)
        # Add the assistant's response to the messages
        messages.append({"role": "assistant", "content": response.content})

        tool_calls: list[ToolUseBlock] = [
            block for block in response.content if block.type == "tool_use"
        ]

        # If there are no tool calls, loop ends
        if not tool_calls:
            loop_continue = trigger_hooks("PreLoopEnd", messages)
            if loop_continue:
                messages.append({"role": "user", "content": loop_continue})
                continue
            return

        # If there are tool calls, execute them, collect results
        results = []
        print("\033[33mTOOL EXECUTION: \033[0m")
        for tool_block in tool_calls:
            denied = trigger_hooks("PreToolUse", tool_block)
            if denied:
                results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_block.id,
                        "content": str(denied),
                    }
                )
                continue

            handler = TOOL_HANDLERS.get(tool_block.name)
            result = (
                handler(**tool_block.input)
                if handler
                else f"Unknown tool: {tool_block.name}"
            )

            trigger_hooks("PostToolUse", tool_block, result)
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
            query = input("\033[33mUSER >> \033[0m")
        except EOFError, KeyboardInterrupt:
            return
        if query.strip().lower() in ("q", "quit", "exit", ""):
            return
        trigger_hooks("PostUserSubmit", query)
        messages.append({"role": "user", "content": query})
        agent_loop(messages)


# -- Entry point --
if __name__ == "__main__":
    main()
