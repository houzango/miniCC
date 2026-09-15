import ast
import json
import os
import subprocess
from pathlib import Path

import yaml
from anthropic import Anthropic
from anthropic.types import MessageParam, ToolParam, ToolUseBlock
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
SKILLS_DIR = WORKDIR / "skills"


# Skill catalog scan (used by build_system below)
def _parse_frontmatter(text: str) -> tuple[dict, str]:
    """Parse YAML frontmatter from SKILL.md. Returns (meta, body)."""
    if not text.startswith("---"):
        return {}, text
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}, text
    try:
        meta = yaml.safe_load(parts[1]) or {}
    except yaml.YAMLError:
        meta = {}
    return meta, parts[2].strip()


# Build skill registry at startup (used for safe lookup in load_skill)
SKILL_REGISTRY: dict[str, dict] = {}


def _register_skills():
    """Scan skills/ dir, populate SKILL_REGISTRY with name/description/content."""
    if not SKILLS_DIR.exists():
        return
    for d in sorted(SKILLS_DIR.iterdir()):
        if not d.is_dir():
            continue
        manifest = d / "SKILL.md"
        if manifest.exists():
            raw = manifest.read_text()
            meta, body = _parse_frontmatter(raw)
            name = meta.get("name", d.name)
            description = meta.get(
                "description", body.split("\n")[0].lstrip("#").strip()
            )
            SKILL_REGISTRY[name] = {
                "directory": str(d),
                "name": name,
                "description": description,
                "content": raw,
            }


_register_skills()


def list_skills() -> str:
    """List all skills (name + one-line description)."""
    if not SKILL_REGISTRY:
        return "(no skills found)"
    return "\n".join(
        f"- **{s['name']}**: {s['description']}" for s in SKILL_REGISTRY.values()
    )


def build_system() -> str:
    """Build SYSTEM prompt with skill catalog injected at startup."""
    skills_catalog = list_skills()
    return (
        f"You are a coding agent at {WORKDIR}. Use tools to solve tasks. Act, don't explain. "
        "Before starting any multi-step task, use manage_todo tool to plan your steps. "
        "Update status as you go."
        f"Skills available:\n{skills_catalog}\n"
        "Use load_skill to get full skill details when needed."
    )


def build_sub_system() -> str:
    """Build SUB_SYSTEM prompt with skill catalog injected at startup."""
    skills_catalog = list_skills()
    return (
        f"You are a coding agent at {WORKDIR}. "
        "Complete the given subtask, then return a concise final conclusion. "
        "Do not delegate further."
        f"Skills available:\n{skills_catalog}\n"
        "Use load_skill to get full skill details when needed."
    )


SYSTEM = build_system()

SUB_SYSTEM = build_sub_system()

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
    {
        "name": "manage_todo",
        "description": (
            "Create a list of steps for a multi-step task, and update their status "
            "throughout the session. Pass the full todos list on every call. "
            "Exactly one todo may be in_progress at a time. When moving to the "
            "next step, mark the current one completed and the next one in_progress "
            "in the same call."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "todos": {
                    "type": "array",
                    "maxItems": 20,
                    "items": {
                        "type": "object",
                        "properties": {
                            "content": {"type": "string", "minLength": 1},
                            "status": {
                                "type": "string",
                                "enum": ["pending", "in_progress", "completed"],
                            },
                        },
                        "required": ["content", "status"],
                    },
                }
            },
            "required": ["todos"],
        },
    },
    {
        "name": "task",
        "description": "Launch a subagent to handle a complex subtask. Returns only the final conclusion.",
        "input_schema": {
            "type": "object",
            "properties": {"description": {"type": "string"}},
            "required": ["description"],
        },
    },
    {
        "name": "load_skill",
        "description": "Load the full content of a skill by name.",
        "input_schema": {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        },
    },
]

SUB_TOOLS: list[ToolParam] = [
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
    {
        "name": "load_skill",
        "description": "Load the full content of a skill by name.",
        "input_schema": {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
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


def load_skill(name: str) -> str:
    """Load full skill content. Lookup skill by name via registry — no path traversal."""
    skill = SKILL_REGISTRY.get(name)
    if not skill:
        return f"Skill not found: {name}"
    return f"(Skill directory: {skill['directory']})\n\n{skill['content']}"


class TodoManager:
    def __init__(self):
        self.todos: list[dict] = []

    def update(self, todos: list | str) -> str:
        if isinstance(todos, str):
            try:
                todos = json.loads(todos)
            except json.JSONDecodeError:
                try:
                    todos = ast.literal_eval(todos)
                except (SyntaxError, ValueError) as e:
                    raise ValueError(
                        "todos must be a list or a JSON array string."
                    ) from e

        if not isinstance(todos, list):
            raise ValueError("todos must be a list or a JSON array string.")
        if len(todos) > 20:
            raise ValueError("A maximum of 20 todos is allowed")

        normalized_todos = []
        in_progress_count = 0
        for index, todo in enumerate(todos):
            if not isinstance(todo, dict):
                raise ValueError(f"todos[{index}] must be a JSON object")
            content = str(todo.get("content", "")).strip()
            status = str(todo.get("status", "pending")).lower()
            if not content:
                raise ValueError(f"todos[{index}] requires content")
            if status not in ("pending", "in_progress", "completed"):
                raise ValueError(f"todos[{index}] has invalid status '{status}'")
            if status == "in_progress":
                in_progress_count += 1
            normalized_todos.append({"content": content, "status": status})

        if in_progress_count > 1:
            raise ValueError("Only one todo can be in_progress at a time")

        self.todos = normalized_todos
        return self.render()

    def render(self) -> str:
        if not self.todos:
            return "The list of todos is empty."

        lines = []
        for todo in self.todos:
            marker = {
                "pending": "[ ]",
                "in_progress": "[>]",
                "completed": "[✅]",
            }[todo["status"]]
            lines.append(f"{marker} {todo['content']}")

        done = sum(todo["status"] == "completed" for todo in self.todos)
        lines.append(f"\n({done}/{len(self.todos)} completed)")
        return "\n".join(lines)


TODO_MANAGER = TodoManager()


def run_manage_todo(todos: list | str) -> str:
    result = TODO_MANAGER.update(todos)
    print(f"\n\033[33m## Current Todos\033[0m\n{result}")
    return result


def extract_text(content) -> str:
    """Extract text from LLM response content blocks."""
    if not isinstance(content, list):
        return str(content)
    return "\n".join(
        getattr(b, "text", "") for b in content if getattr(b, "type", None) == "text"
    )


def spawn_subagent(description: str) -> str:
    """Spawn a subagent with fresh messages[], return summary only."""
    print("\n\033[35m[Subagent spawned]\033[0m")
    messages: list[MessageParam] = [{"role": "user", "content": description}]

    for _ in range(30):  # safety limit
        response = client.messages.create(
            model=MODEL,
            system=SUB_SYSTEM,
            messages=messages,
            tools=SUB_TOOLS,
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
            print("\n\033[35m[Subagent completed]\033[0m")
            return f"Subagent completed: {extract_text(response.content)}"

        # If there are tool calls, execute them, collect results
        results = []
        print("\033[33mTOOL EXECUTION: \033[0m")
        for tool_block in tool_calls:
            result_dict = execute_tool(tool_block, SUB_TOOL_HANDLERS)
            results.append(result_dict)

        print()

        # Add the tool results to the messages, loop continues
        messages.append({"role": "user", "content": results})

    messages.append(
        {
            "role": "user",
            "content": "You're out of turns. Stop using tools and give your best final answer now, based on what you've done so far.",
        }
    )
    response = client.messages.create(
        model=MODEL,
        system=SUB_SYSTEM,
        messages=messages,
        tools=[],
        max_tokens=8000,
    )
    trigger_hooks("PostModelResponse", response.content)
    print("\n\033[35m[Subagent hit turn limit]\033[0m")
    return f"Subagent hit turn limit before finishing. Last response: {extract_text(response.content)}"


TOOL_HANDLERS = {
    "bash": run_bash,
    "read_file": run_read,
    "write_file": run_write,
    "edit_file": run_edit,
    "glob": run_glob,
    "load_skill": load_skill,
    "manage_todo": run_manage_todo,
    "task": spawn_subagent,
}

SUB_TOOL_HANDLERS = {
    "bash": run_bash,
    "read_file": run_read,
    "write_file": run_write,
    "edit_file": run_edit,
    "glob": run_glob,
    "load_skill": load_skill,
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


def execute_tool(tool_block: ToolUseBlock, handlers: dict) -> dict:
    denied = trigger_hooks("PreToolUse", tool_block)
    if denied:
        return {
            "type": "tool_result",
            "tool_use_id": tool_block.id,
            "content": str(denied),
        }

    handler = handlers.get(tool_block.name)
    try:
        result = (
            handler(**tool_block.input)
            if handler
            else f"Unknown tool: {tool_block.name}"
        )
    except Exception as e:
        result = f"Tool error: {e}"

    trigger_hooks("PostToolUse", tool_block, result)

    return {
        "type": "tool_result",
        "tool_use_id": tool_block.id,
        "content": str(result).strip(),
    }


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
        manage_todo_used = False
        print("\033[33mTOOL EXECUTION: \033[0m")
        for tool_block in tool_calls:
            result_dict = execute_tool(tool_block, TOOL_HANDLERS)
            results.append(result_dict)

            if tool_block.name == "manage_todo":
                manage_todo_used = True

        # If there is incomplete work in the todo list and manage_todo was not used, add a reminder
        has_incomplete_work = any(
            t["status"] != "completed" for t in TODO_MANAGER.todos
        )
        if has_incomplete_work and not manage_todo_used:
            results.append(
                {
                    "type": "text",
                    "text": "<reminder>Update your todos if needed.</reminder>",
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
