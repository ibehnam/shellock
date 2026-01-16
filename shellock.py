#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.13"
# dependencies = []
# ///
"""Shellock - Real-time CLI flag explainer.

Parses command-line flags and retrieves their descriptions from --help or man pages.
Designed to integrate with fish shell for real-time explanations as you type.

Storage model:
- Durable command metadata lives under `~/.config/fish/shellock/data/`.
- One JSON file per command (e.g. `git.json`) with nested subcommand data.
- `SHELLOCK_HOME` environment variable overrides the storage location.

Usage:
    shellock parse "jq -r -s file.json"           # Parse flags from command
    shellock lookup jq -r                          # Look up description for -r flag of jq
    shellock explain "jq -r -s file.json"         # Parse and explain all flags
    shellock -r tree                               # Refresh `tree` (command + subcommands)
    shellock -r -a                                 # Refresh all known commands
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import signal
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Final, Protocol, TypedDict

# Constants
PROTOCOL_VERSION: Final[int] = 1

SubcommandPath = tuple[str, ...]


class Sources(TypedDict):
    help: bool
    man: bool


class SubcommandData(TypedDict):
    generated_at: str
    sources: Sources
    flags: dict[str, str]
    subcommands: dict[str, "SubcommandData"]


class CommandData(TypedDict):
    protocol_version: int
    command: str
    generated_at: str
    sources: Sources
    flags: dict[str, str]
    subcommands: dict[str, SubcommandData]


def _subcommand_schema_ref() -> dict:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "generated_at": {"type": "string", "minLength": 1},
            "sources": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "help": {"type": "boolean"},
                    "man": {"type": "boolean"},
                },
                "required": ["help", "man"],
            },
            "flags": {
                "type": "object",
                "additionalProperties": {"type": "string"},
            },
            "subcommands": {
                "type": "object",
                "additionalProperties": {"$ref": "#/$defs/subcommand"},
            },
        },
        "required": ["generated_at", "sources", "flags", "subcommands"],
    }


def command_data_json_schema() -> dict:
    """JSON Schema for the durable command metadata stored under `.../data/<cmd>.json`."""
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "protocol_version": {"type": "integer"},
            "command": {"type": "string", "minLength": 1},
            "generated_at": {"type": "string", "minLength": 1},
            "sources": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "help": {"type": "boolean"},
                    "man": {"type": "boolean"},
                },
                "required": ["help", "man"],
            },
            "flags": {
                "type": "object",
                "additionalProperties": {"type": "string"},
            },
            "subcommands": {
                "type": "object",
                "additionalProperties": {"$ref": "#/$defs/subcommand"},
            },
        },
        "$defs": {"subcommand": _subcommand_schema_ref()},
        "required": [
            "protocol_version",
            "command",
            "generated_at",
            "sources",
            "flags",
            "subcommands",
        ],
    }


def subcommand_data_json_schema() -> dict:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "generated_at": {"type": "string", "minLength": 1},
            "sources": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "help": {"type": "boolean"},
                    "man": {"type": "boolean"},
                },
                "required": ["help", "man"],
            },
            "flags": {
                "type": "object",
                "additionalProperties": {"type": "string"},
            },
            "subcommands": {
                "type": "object",
                "additionalProperties": {"$ref": "#"},
            },
        },
        "required": ["generated_at", "sources", "flags", "subcommands"],
    }


def llm_extraction_json_schema() -> dict:
    """JSON Schema for LLM extraction - returns flags and discovered subcommand names.

    The LLM extracts from help/man output:
    - flags: mapping of flag tokens to their descriptions
    - subcommands: list of subcommand names discovered in the documentation
    """
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "flags": {
                "type": "object",
                "additionalProperties": {"type": "string"},
                "description": (
                    "Map of CLI flags to their descriptions. "
                    "Keys are flag tokens (e.g., '-v', '--verbose', '-rf'). "
                    "Values are concise descriptions (max 160 chars). "
                    "Include both short (-x) and long (--example) forms as separate entries. "
                    "Omit argument placeholders from keys (use '--output' not '--output=FILE')."
                ),
            },
            "subcommands": {
                "type": "array",
                "items": {"type": "string", "minLength": 1},
                "description": (
                    "List of subcommand names found in the documentation. "
                    "Only include actual subcommands (e.g., 'commit', 'push' for git). "
                    "Exclude flags, arguments, and general help text. "
                    "Return an empty array if no subcommands are documented."
                ),
            },
        },
        "required": ["flags", "subcommands"],
    }


@dataclass(frozen=True, slots=True)
class LlmExtractionResult:
    """Result from LLM extraction of a single help page."""

    flags: dict[str, str]
    subcommands: list[str]


def validate_llm_extraction(*, payload: object) -> LlmExtractionResult:
    """Validate and normalize LLM extraction output."""
    if not isinstance(payload, dict):
        raise ValueError("LLM extraction payload must be an object")

    raw_flags = payload.get("flags")
    if not isinstance(raw_flags, dict):
        raw_flags = {}
    flags: dict[str, str] = {}
    for k, v in raw_flags.items():
        if isinstance(k, str) and k and isinstance(v, str):
            flags[k.strip()] = v.strip()

    raw_subcommands = payload.get("subcommands")
    if not isinstance(raw_subcommands, list):
        raw_subcommands = []
    subcommands: list[str] = []
    for item in raw_subcommands:
        if isinstance(item, str) and item.strip():
            subcommands.append(item.strip())

    return LlmExtractionResult(flags=flags, subcommands=subcommands)


class LlmAgentError(RuntimeError):
    pass


class StructuredJsonAgent(Protocol):
    def run_json(
        self, *, prompt: str, json_schema: dict, timeout_s: int | None = None
    ) -> object: ...


def _parse_first_json_value(*, text: str) -> object:
    decoder = json.JSONDecoder()
    for idx, ch in enumerate(text):
        if ch not in "{[":
            continue
        try:
            value, _ = decoder.raw_decode(text[idx:])
        except json.JSONDecodeError:
            continue
        return value
    raise ValueError("No JSON value found in agent output")


@dataclass(frozen=True, slots=True)
class ClaudeCodeAgent:
    model: str = "sonnet"
    # executable: str = "claude"
    executable: str = "ccs glm"

    def run_json(
        self, *, prompt: str, json_schema: dict, timeout_s: int | None = None
    ) -> object:
        schema_arg = json.dumps(json_schema, separators=(",", ":"), sort_keys=True)
        base_cmd = shlex.split(self.executable)
        if not base_cmd:
            raise LlmAgentError("Agent executable is empty")
        verbosity = _llm_verbose_level()
        if verbosity:
            print(
                "[shellock] LLM agent start",
                file=sys.stderr,
            )
            print(
                f"[shellock] executable: {shlex.join(base_cmd)}",
                file=sys.stderr,
            )
            print(
                f"[shellock] model: {self.model}",
                file=sys.stderr,
            )
            print(
                f"[shellock] prompt chars: {len(prompt)}",
                file=sys.stderr,
            )
            print(
                f"[shellock] schema chars: {len(schema_arg)}",
                file=sys.stderr,
            )
            print(
                f"[shellock] timeout_s: {timeout_s or 120}",
                file=sys.stderr,
            )
            if verbosity > 1:
                print("[shellock] prompt:", file=sys.stderr)
                print(prompt, file=sys.stderr)
                print("[shellock] json schema:", file=sys.stderr)
                print(schema_arg, file=sys.stderr)
        cmd = [
            *base_cmd,
            "--print",
            "--output-format",
            "json",
            "--no-session-persistence",
            "--no-chrome",
            "--disable-slash-commands",
            "--tools",
            "",
            "--model",
            self.model,
            "--json-schema",
            schema_arg,
            prompt,
        ]
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout_s or 120,
            )
        except FileNotFoundError as e:
            raise LlmAgentError(f"Agent executable not found: {self.executable}") from e
        except subprocess.TimeoutExpired as e:
            raise LlmAgentError("Agent call timed out") from e

        if verbosity:
            print(
                f"[shellock] LLM agent exit: {result.returncode}",
                file=sys.stderr,
            )
        if result.returncode != 0:
            stderr = (result.stderr or "").strip()
            raise LlmAgentError(
                f"Agent failed (exit {result.returncode}): {stderr or 'no stderr'}"
            )

        raw = (result.stdout or "").strip()
        if verbosity:
            stdout_len = len(result.stdout or "")
            stderr_len = len(result.stderr or "")
            print(
                f"[shellock] LLM agent stdout chars: {stdout_len}",
                file=sys.stderr,
            )
            if stderr_len:
                print(
                    f"[shellock] LLM agent stderr chars: {stderr_len}",
                    file=sys.stderr,
                )
            if verbosity > 1 and result.stderr:
                print("[shellock] LLM agent stderr:", file=sys.stderr)
                print(result.stderr, file=sys.stderr)
        if not raw:
            raise LlmAgentError("Agent returned empty output")
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            try:
                parsed = _parse_first_json_value(text=raw)
            except ValueError as e:
                raise LlmAgentError(
                    f"Agent output was not valid JSON: {raw[:4000]}"
                ) from e

        # Handle ccs/claude --output-format json which returns a JSONL array of events.
        # The structured output is in the last "result" event under "structured_output".
        if isinstance(parsed, list):
            for event in reversed(parsed):
                if isinstance(event, dict) and event.get("type") == "result":
                    structured = event.get("structured_output")
                    if structured is not None:
                        return structured
            raise LlmAgentError("No structured_output found in agent result events")

        return parsed


def _validate_sources(*, sources: object) -> Sources:
    if not isinstance(sources, dict):
        raise ValueError("sources must be an object")
    help_ok = sources.get("help")
    man_ok = sources.get("man")
    if not isinstance(help_ok, bool) or not isinstance(man_ok, bool):
        raise ValueError("sources.help and sources.man must be booleans")
    return {"help": help_ok, "man": man_ok}


def _validate_flags(*, flags: object) -> dict[str, str]:
    if not isinstance(flags, dict):
        raise ValueError("flags must be an object")
    out: dict[str, str] = {}
    for k, v in flags.items():
        if not isinstance(k, str) or not isinstance(v, str):
            raise ValueError("flags must map strings to strings")
        if not k:
            continue
        out[k] = v.strip()
    return out


def _validate_subcommand_payload(*, payload: object) -> SubcommandData:
    if not isinstance(payload, dict):
        raise ValueError("subcommand payload must be an object")
    generated_at = payload.get("generated_at")
    if not isinstance(generated_at, str) or not generated_at:
        raise ValueError("subcommand generated_at must be a non-empty string")
    sources = _validate_sources(sources=payload.get("sources"))
    flags = _validate_flags(flags=payload.get("flags"))
    subcommands_raw = payload.get("subcommands")
    if subcommands_raw is None:
        subcommands_raw = {}
    if not isinstance(subcommands_raw, dict):
        raise ValueError("subcommand subcommands must be an object")
    subcommands: dict[str, SubcommandData] = {}
    for sub_name, sub_payload in subcommands_raw.items():
        if not isinstance(sub_name, str) or not sub_name:
            raise ValueError("subcommand names must be non-empty strings")
        subcommands[sub_name] = _validate_subcommand_payload(payload=sub_payload)
    return {
        "generated_at": generated_at,
        "sources": sources,
        "flags": flags,
        "subcommands": subcommands,
    }


def validate_command_data(*, payload: object, command: str) -> CommandData:
    if not isinstance(payload, dict):
        raise ValueError("payload must be an object")
    if payload.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("protocol_version mismatch")
    if payload.get("command") != command:
        raise ValueError("command mismatch")
    generated_at = payload.get("generated_at")
    if not isinstance(generated_at, str) or not generated_at:
        raise ValueError("generated_at must be a non-empty string")

    sources = _validate_sources(sources=payload.get("sources"))
    flags = _validate_flags(flags=payload.get("flags"))

    subcommands_raw = payload.get("subcommands")
    if not isinstance(subcommands_raw, dict):
        raise ValueError("subcommands must be an object")
    subcommands: dict[str, SubcommandData] = {}
    for sub_name, sub_payload in subcommands_raw.items():
        if not isinstance(sub_name, str) or not sub_name:
            raise ValueError("subcommand names must be non-empty strings")
        subcommands[sub_name] = _validate_subcommand_payload(payload=sub_payload)

    return {
        "protocol_version": PROTOCOL_VERSION,
        "command": command,
        "generated_at": generated_at,
        "sources": sources,
        "flags": flags,
        "subcommands": subcommands,
    }


def validate_subcommand_data(*, payload: object) -> SubcommandData:
    return _validate_subcommand_payload(payload=payload)


# ANSI colors for terminal output
DIM: Final[str] = "\033[2m"
RESET: Final[str] = "\033[0m"
CYAN: Final[str] = "\033[36m"

# Command-specific help overrides
# Some commands need special help invocations to show all flags
HELP_SOURCE_OVERRIDES: Final[dict[str, object]] = {
    "curl": ["curl", "--help", "all"],
    "git": lambda subcmds: (["git", *subcmds, "-h"] if subcmds else ["git", "-h"]),
    "docker": lambda subcmds: (
        ["docker", *subcmds, "--help"] if subcmds else ["docker", "--help"]
    ),
    "kubectl": lambda subcmds: (
        ["kubectl", *subcmds, "--help"] if subcmds else ["kubectl", "--help"]
    ),
    "npm": lambda subcmds: (
        ["npm", *subcmds, "--help"] if subcmds else ["npm", "--help"]
    ),
}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def shellock_home() -> Path:
    """Return Shellock's home directory.

    Defaults to `~/.config/fish/shellock`, overridable via `SHELLOCK_HOME`.
    """
    raw = os.environ.get("SHELLOCK_HOME")
    if raw:
        return Path(raw).expanduser()
    return Path.home() / ".config" / "fish" / "shellock"


def shellock_data_dir() -> Path:
    return shellock_home() / "data"


def shellock_config_path() -> Path:
    return shellock_home() / "config.json"


def _load_config() -> dict:
    path = shellock_config_path()
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}
    if isinstance(payload, dict):
        return payload
    return {}


def _config_get(*, key: str) -> object | None:
    # Environment variables override config.json for quick experimentation.
    # Example: `SHELLOCK_SCAN_BACKEND=llm`, `SHELLOCK_LLM_MODEL=sonnet`.
    env_key = f"SHELLOCK_{key.upper()}"
    env_val = os.environ.get(env_key)
    if env_val is not None and env_val.strip() != "":
        return env_val
    return _load_config().get(key)


def _llm_verbose_level() -> int:
    raw = _config_get(key="llm_verbose")
    if raw is None:
        return 0
    if isinstance(raw, bool):
        return 1 if raw else 0
    if isinstance(raw, int):
        if raw <= 0:
            return 0
        return 2 if raw > 1 else 1
    if isinstance(raw, str):
        value = raw.strip().lower()
        if value in {"", "0", "false", "no", "off"}:
            return 0
        if value in {"1", "true", "yes", "on", "basic"}:
            return 1
        return 2
    return 0


def _command_file_name(command: str) -> str:
    """Use the command token as the name, sanitized for filesystem."""
    safe = command.replace("/", "_").replace("\\", "_")
    safe = safe.replace(":", "_")
    safe = re.sub(r"\s+", "_", safe)
    return safe


def _command_exists(*, command: str) -> bool:
    if not command:
        return False
    if "/" in command or "\\" in command:
        path = Path(command).expanduser()
        return path.is_file() and os.access(path, os.X_OK)
    return shutil.which(command) is not None


def command_data_path(*, command: str) -> Path:
    return shellock_data_dir() / f"{_command_file_name(command)}.json"


class HelpProvider(Protocol):
    def get_help_text(
        self, *, command: str, subcommand: SubcommandPath | None
    ) -> str | None: ...

    def get_man_text(
        self, *, command: str, subcommand: SubcommandPath | None
    ) -> str | None: ...


class DefaultHelpProvider:
    def get_help_text(
        self, *, command: str, subcommand: SubcommandPath | None
    ) -> str | None:
        return get_help_text(command=command, subcommand=subcommand)

    def get_man_text(
        self, *, command: str, subcommand: SubcommandPath | None
    ) -> str | None:
        return get_man_text(command=command, subcommand=subcommand)


def _truncate_for_prompt(*, text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n\n[... truncated ...]\n"


def _subcommand_path_label(subcommand: SubcommandPath | None) -> str:
    if not subcommand:
        return ""
    return " ".join(subcommand)


def _llm_extract_flags_and_subcommands(
    *,
    agent: StructuredJsonAgent,
    command: str,
    subcommand_path: SubcommandPath | None,
    help_text: str,
    max_doc_chars: int,
    timeout_s: int,
) -> LlmExtractionResult:
    """Use LLM to extract flags and subcommand names from help text.

    This is the core extraction function - it asks the LLM to parse a single
    help page and return both:
    - flags: dict of flag -> description
    - subcommands: list of subcommand names (for recursive drilling)
    """
    target = (
        command
        if not subcommand_path
        else f"{command} {_subcommand_path_label(subcommand_path)}"
    )
    header = "CLI command" if not subcommand_path else "CLI subcommand"

    prompt = (
        "\n".join(
            [
                f"You are given documentation text for a {header}.",
                "Produce a single JSON value that matches the provided JSON Schema exactly.",
                "",
                "Extraction rules:",
                "- flags: Extract all CLI flags/options (tokens starting with '-' or '--').",
                "  - Use the flag token only (omit argument placeholders like '<path>' or '=VALUE').",
                "  - Include both short (-x) and long (--example) forms as separate entries if documented.",
                "  - Keep descriptions concise (max 160 chars).",
                "  - Do not invent flags not present in the documentation.",
                "",
                "- subcommands: Extract names of subcommands if this command has them.",
                "  - Subcommands are typically listed in a 'Commands:', 'Subcommands:', or similar section.",
                "  - Only include actual command names (e.g., 'commit', 'push', 'run').",
                "  - Do NOT include flags, arguments, or example values.",
                "  - Return empty array if the command has no subcommands.",
                "",
                f"Target: {target}",
                "",
                f"== {target} (help) ==",
                _truncate_for_prompt(text=help_text, max_chars=max_doc_chars),
            ]
        ).strip()
        + "\n"
    )

    candidate = agent.run_json(
        prompt=prompt,
        json_schema=llm_extraction_json_schema(),
        timeout_s=timeout_s,
    )

    return validate_llm_extraction(payload=candidate)


def _scan_subcommand_parallel(
    *,
    help_provider: HelpProvider,
    agent: StructuredJsonAgent,
    command: str,
    subcommand_path: SubcommandPath,
    depth_remaining: int,
    max_subcommands: int | None,
    max_doc_chars: int,
    timeout_s: int,
    executor: concurrent.futures.ThreadPoolExecutor,
) -> SubcommandData:
    """Recursively scan a subcommand node, processing children in parallel.

    1. Get help text for command [subcommand_path...] --help
    2. Ask LLM to extract flags + subcommand names
    3. If depth > 0, recurse on each discovered subcommand in parallel
    4. Return SubcommandData with nested structure
    """
    # Get help text for this node
    help_text = help_provider.get_help_text(
        command=command, subcommand=subcommand_path
    )

    if not help_text:
        # No help available at this level - return empty node
        return {
            "generated_at": _utc_now_iso(),
            "sources": {"help": False, "man": False},
            "flags": {},
            "subcommands": {},
        }

    # Extract flags and subcommand names via LLM
    try:
        extraction = _llm_extract_flags_and_subcommands(
            agent=agent,
            command=command,
            subcommand_path=subcommand_path,
            help_text=help_text,
            max_doc_chars=max_doc_chars,
            timeout_s=timeout_s,
        )
    except LlmAgentError:
        # LLM failed - return what we have (no flags, no subcommands)
        return {
            "generated_at": _utc_now_iso(),
            "sources": {"help": True, "man": False},
            "flags": {},
            "subcommands": {},
        }

    # Apply subcommand limit
    discovered = extraction.subcommands
    if max_subcommands is not None and max_subcommands > 0:
        discovered = discovered[:max_subcommands]

    # Recurse into subcommands in parallel if depth allows
    subcommands: dict[str, SubcommandData] = {}
    if depth_remaining > 0 and discovered:
        # Submit all subcommand scans to the executor
        futures: dict[concurrent.futures.Future[SubcommandData], str] = {}
        for sub_name in discovered:
            future = executor.submit(
                _scan_subcommand_parallel,
                help_provider=help_provider,
                agent=agent,
                command=command,
                subcommand_path=subcommand_path + (sub_name,),
                depth_remaining=depth_remaining - 1,
                max_subcommands=max_subcommands,
                max_doc_chars=max_doc_chars,
                timeout_s=timeout_s,
                executor=executor,
            )
            futures[future] = sub_name

        # Collect results as they complete
        for future in concurrent.futures.as_completed(futures):
            sub_name = futures[future]
            try:
                subcommands[sub_name] = future.result()
            except Exception:
                # If a subcommand scan fails, skip it
                pass

    return {
        "generated_at": _utc_now_iso(),
        "sources": {"help": True, "man": False},
        "flags": extraction.flags,
        "subcommands": subcommands,
    }


def _scan_command_tree_parallel(
    *,
    help_provider: HelpProvider,
    agent: StructuredJsonAgent,
    command: str,
    max_depth: int,
    max_subcommands: int | None,
    max_doc_chars: int,
    timeout_s: int,
    max_workers: int | None = None,
) -> CommandData:
    """Scan a command and all its subcommands using parallel LLM extraction.

    This is the main entry point for LLM-based scanning. It:
    1. Gets help for the root command
    2. Extracts flags and subcommand names via LLM
    3. Recursively processes subcommands in parallel using ThreadPoolExecutor
    """
    # Get help text for root command
    help_text = help_provider.get_help_text(command=command, subcommand=None)

    if not help_text:
        raise LlmAgentError(f"No help output found for command: {command}")

    # Extract flags and subcommand names via LLM
    extraction = _llm_extract_flags_and_subcommands(
        agent=agent,
        command=command,
        subcommand_path=None,
        help_text=help_text,
        max_doc_chars=max_doc_chars,
        timeout_s=timeout_s,
    )

    # Apply subcommand limit
    discovered = extraction.subcommands
    if max_subcommands is not None and max_subcommands > 0:
        discovered = discovered[:max_subcommands]

    # Recurse into subcommands in parallel
    subcommands: dict[str, SubcommandData] = {}
    if max_depth > 0 and discovered:
        # Use ThreadPoolExecutor for parallel subcommand scanning
        workers = max_workers or min(8, len(discovered))
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures: dict[concurrent.futures.Future[SubcommandData], str] = {}
            for sub_name in discovered:
                future = executor.submit(
                    _scan_subcommand_parallel,
                    help_provider=help_provider,
                    agent=agent,
                    command=command,
                    subcommand_path=(sub_name,),
                    depth_remaining=max_depth - 1,
                    max_subcommands=max_subcommands,
                    max_doc_chars=max_doc_chars,
                    timeout_s=timeout_s,
                    executor=executor,
                )
                futures[future] = sub_name

            # Collect results
            for future in concurrent.futures.as_completed(futures):
                sub_name = futures[future]
                try:
                    subcommands[sub_name] = future.result()
                except Exception:
                    # If a subcommand scan fails, skip it
                    pass

    return {
        "protocol_version": PROTOCOL_VERSION,
        "command": command,
        "generated_at": _utc_now_iso(),
        "sources": {"help": True, "man": False},
        "flags": extraction.flags,
        "subcommands": subcommands,
    }


def generate_command_data_llm(
    *,
    help_provider: HelpProvider,
    agent: StructuredJsonAgent,
    command: str,
    max_subcommands: int,
    max_doc_chars: int,
    timeout_s: int,
    max_depth: int,
) -> CommandData:
    """Generate command JSON using LLM with parallel subcommand scanning."""
    max_subs = max_subcommands if max_subcommands > 0 else None
    return _scan_command_tree_parallel(
        help_provider=help_provider,
        agent=agent,
        command=command,
        max_depth=max_depth,
        max_subcommands=max_subs,
        max_doc_chars=max_doc_chars,
        timeout_s=timeout_s,
    )


def _setting_int(*, config_key: str, default: int) -> int:
    cfg = _config_get(key=config_key)
    if cfg is None:
        return default
    try:
        return int(cfg)
    except (TypeError, ValueError):
        return default


def _max_subcommand_depth() -> int:
    return max(0, _setting_int(config_key="max_subcommand_depth", default=3))


def _llm_agent_from_env() -> StructuredJsonAgent:
    cfg = _config_get(key="llm_model")
    model = cfg.strip() if isinstance(cfg, str) and cfg.strip() else "sonnet"
    return ClaudeCodeAgent(model=model)




def load_command_data(*, command: str) -> dict | None:
    path = command_data_path(command=command)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def save_command_data(*, command: str, data: dict) -> None:
    shellock_data_dir().mkdir(parents=True, exist_ok=True)
    path = command_data_path(command=command)
    # Atomic write: write to temp file then rename
    temp_path = path.with_suffix(f".tmp.{os.getpid()}")
    try:
        temp_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
        temp_path.rename(path)  # Atomic on POSIX
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def is_scan_in_progress(*, command: str) -> bool:
    """Check if a background scan is already running for this command."""
    lock_path = command_data_path(command=command).with_suffix(".scanning")
    if not lock_path.exists():
        return False
    # Stale lock check (5 minute timeout)
    try:
        age = time.time() - lock_path.stat().st_mtime
        return age < 300
    except FileNotFoundError:
        return False


def spawn_background_scan(*, command: str) -> None:
    """Spawn a detached background process to scan a command."""
    script_path = str(Path(__file__).resolve())
    subprocess.Popen(
        [script_path, "scan", command],
        start_new_session=True,  # Detach from terminal
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _notify_fish_scan_complete(*, command: str) -> None:
    """Best-effort notification for interactive fish sessions.

    Background scans run detached, so fish won't re-run `__shellock_explain`
    unless the user types again. We use a universal variable update as an
    inter-process signal: interactive fish sessions can listen for changes
    to `__shellock_scan_done` and refresh the hint.
    """
    payload = f"{command}:{int(time.time() * 1000)}"
    env = {**os.environ, "SHELLOCK_SCAN_DONE": payload}
    candidates = [
        # Avoid loading user config (can be slow); fall back if unsupported.
        (["fish", "--no-config", "-c"], 2),
        (["fish", "-c"], 5),
    ]

    for prefix, timeout_s in candidates:
        try:
            result = subprocess.run(
                [
                    *prefix,
                    'set -U __shellock_scan_done -- "$SHELLOCK_SCAN_DONE"',
                ],
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=timeout_s,
                check=False,
            )
            if result.returncode == 0:
                break
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
        except Exception:
            # Non-fatal: shellock still works without live refresh.
            break

    _signal_fish_pid_from_env()


def _signal_fish_pid_from_env() -> None:
    """Wake up a specific interactive fish session, if provided.

    Some fish setups only process universal-variable updates on the next
    keystroke. When Shellock is invoked from fish, the integration can pass
    `SHELLOCK_FISH_PID=$fish_pid` so the detached scan can signal that exact
    shell instance.
    """
    raw = os.environ.get("SHELLOCK_FISH_PID")
    if not raw:
        return

    try:
        pid = int(raw)
    except ValueError:
        return
    if pid <= 0:
        return

    try:
        sig = signal.SIGUSR1
    except AttributeError:
        return

    try:
        os.kill(pid, sig)
    except Exception:
        return


def ensure_command_scanned(
    *, help_provider: HelpProvider, command: str, refresh: bool
) -> dict:
    """Ensure command data exists, generating via LLM if needed."""
    data = load_command_data(command=command)

    if data is None or refresh:
        max_depth = _max_subcommand_depth()
        max_subcommands = _setting_int(config_key="llm_max_subcommands", default=25)
        max_doc_chars = _setting_int(config_key="llm_max_doc_chars", default=30000)
        timeout_s = _setting_int(config_key="llm_timeout_s", default=180)

        llm_data = generate_command_data_llm(
            help_provider=help_provider,
            agent=_llm_agent_from_env(),
            command=command,
            max_subcommands=max_subcommands,
            max_doc_chars=max_doc_chars,
            timeout_s=timeout_s,
            max_depth=max_depth,
        )
        save_command_data(command=command, data=llm_data)
        return llm_data

    return data


def ensure_subcommand_scanned(
    *, help_provider: HelpProvider, command: str, subcommand: SubcommandPath
) -> dict:
    """Ensure subcommand data exists within the command's data structure.

    If the subcommand path doesn't exist in the stored data, this will scan
    that specific branch using LLM extraction.
    """
    data = ensure_command_scanned(
        help_provider=help_provider, command=command, refresh=False
    )
    max_depth = _max_subcommand_depth()
    max_subcommands = _setting_int(config_key="llm_max_subcommands", default=25)
    max_doc_chars = _setting_int(config_key="llm_max_doc_chars", default=30000)
    timeout_s = _setting_int(config_key="llm_timeout_s", default=180)
    agent = _llm_agent_from_env()

    cursor: dict | None = data
    updated = False
    for idx, part in enumerate(subcommand):
        if not isinstance(cursor, dict):
            break
        subcommands = cursor.get("subcommands")
        if not isinstance(subcommands, dict):
            subcommands = {}
            cursor["subcommands"] = subcommands
            updated = True
        node = subcommands.get(part)
        if not isinstance(node, dict):
            # Need to scan this subcommand branch
            remaining_depth = max(0, max_depth - (idx + 1))
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                node = _scan_subcommand_parallel(
                    help_provider=help_provider,
                    agent=agent,
                    command=command,
                    subcommand_path=subcommand[: idx + 1],
                    depth_remaining=remaining_depth,
                    max_subcommands=max_subcommands,
                    max_doc_chars=max_doc_chars,
                    timeout_s=timeout_s,
                    executor=executor,
                )
            subcommands[part] = node
            updated = True
        cursor = node

    if updated:
        save_command_data(command=command, data=data)

    return data


def refresh_command(*, help_provider: HelpProvider, command: str) -> None:
    """Refresh command data by re-scanning with LLM."""
    max_depth = _max_subcommand_depth()
    max_subcommands = _setting_int(config_key="llm_max_subcommands", default=25)
    max_doc_chars = _setting_int(config_key="llm_max_doc_chars", default=30000)
    timeout_s = _setting_int(config_key="llm_timeout_s", default=180)

    llm_data = generate_command_data_llm(
        help_provider=help_provider,
        agent=_llm_agent_from_env(),
        command=command,
        max_subcommands=max_subcommands,
        max_doc_chars=max_doc_chars,
        timeout_s=timeout_s,
        max_depth=max_depth,
    )
    save_command_data(command=command, data=llm_data)


def refresh_all(*, help_provider: HelpProvider) -> None:
    """Refresh all known commands."""
    data_dir = shellock_data_dir()
    if not data_dir.exists():
        return
    for path in sorted(data_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        cmd = payload.get("command")
        if isinstance(cmd, str) and cmd:
            refresh_command(help_provider=help_provider, command=cmd)


@dataclass(frozen=True, slots=True)
class Flag:
    """Represents a command-line flag."""

    name: str  # e.g., "-r" or "--raw-output"
    is_long: bool  # True for --flag, False for -f


@dataclass(frozen=True, slots=True)
class FlagDescription:
    """A flag with its description."""

    flag: str
    description: str


@dataclass(frozen=True, slots=True)
class ParsedCommand:
    """Result of parsing a command line."""

    command: str
    flags: tuple[Flag, ...]
    subcommands: SubcommandPath = ()


def parse_command_line(
    *,
    cmdline: str,
    known_flags: set[str] | None = None,
    known_subcommands: dict[str, dict] | None = None,
) -> ParsedCommand | None:
    """
    Parse a command line string to extract command and flags.

    Handles:
    - Short flags: -r, -v, -rf (combined)
    - Long flags: --raw-output, --verbose
    - Subcommands: git commit -m
    - Subcommand paths: git remote add -f
    - Known single-dash flags: -ngl (when provided in known_flags)
    """
    try:
        tokens = shlex.split(cmdline)
    except ValueError:
        # Incomplete quotes, try simple split
        tokens = cmdline.split()

    if not tokens:
        return None

    command = tokens[0]
    flags: list[Flag] = []
    subcommands: list[str] = []

    # Known commands with subcommands
    subcommand_commands = {
        "git",
        "docker",
        "kubectl",
        "npm",
        "cargo",
        "go",
        "pip",
        "uv",
    }

    known_flag_set = set(known_flags or ())
    subcommand_map = known_subcommands if isinstance(known_subcommands, dict) else None
    parsing_subcommands = True
    i = 1
    skip_next = False  # Track if next token is a flag value
    while i < len(tokens):
        token = tokens[i]

        if token == "--":
            # Option terminator: stop parsing flags, keep prior hints.
            break

        # Skip this token if it's a flag value (from previous flag)
        if skip_next:
            skip_next = False
            i += 1
            continue

        # Check for subcommand (first non-flag after command)
        # Don't treat key=value pairs as subcommands (likely flag values)
        if parsing_subcommands and not token.startswith("-") and "=" not in token:
            if subcommand_map is not None:
                sub_payload = subcommand_map.get(token)
                if isinstance(sub_payload, dict):
                    subcommands.append(token)
                    subcommand_map = sub_payload.get("subcommands")
                    if not isinstance(subcommand_map, dict):
                        subcommand_map = {}
                    i += 1
                    continue
                parsing_subcommands = False
            elif command in subcommand_commands and not subcommands:
                subcommands.append(token)
                parsing_subcommands = False
                i += 1
                continue

        if token.startswith("--"):
            # Long flag: --raw-output or --key=value
            flag_name = token.split("=")[0]
            flags.append(Flag(name=flag_name, is_long=True))
        elif token.startswith("-") and len(token) > 1 and token[1] != "-":
            # Could be:
            # - Short flag(s): -r or -rf (combined)
            # - Single-dash long flag: -name, -type (find, java style)
            # Heuristic: 4+ lowercase letters after dash = single-dash long flag
            # This handles -name (4), -type (4), -exec (4) but not -avz (3), -rf (2)
            flag_token = token.split("=", 1)[0]
            flag_part = flag_token[1:]
            if flag_token in known_flag_set:
                flags.append(Flag(name=flag_token, is_long=False))
            elif len(flag_part) >= 4 and flag_part.isalpha() and flag_part.islower():
                # Likely a single-dash long flag (e.g., -name, -type, -exec)
                flags.append(Flag(name=flag_token, is_long=False))
            else:
                # Short flag(s): -r or -rf (combined)
                for char in flag_part:
                    if char.isalpha():
                        flags.append(Flag(name=f"-{char}", is_long=False))
                    else:
                        # Stop at non-alpha (e.g., -9 for kill)
                        flags.append(Flag(name=f"-{char}", is_long=False))
                        break

        i += 1

    return ParsedCommand(
        command=command,
        flags=tuple(flags),
        subcommands=tuple(subcommands),
    )


def truncate_description(text: str, max_length: int = 160) -> str:
    """Truncate description at sentence or word boundary."""
    if len(text) <= max_length:
        return text

    limit = max_length - 3  # Room for "..."
    truncated = text[:limit]

    # Try sentence boundary (. ! ?) followed by space
    for punct in (". ", "! ", "? "):
        pos = truncated.rfind(punct)
        if pos > limit // 2:
            return text[: pos + 1]

    # Fall back to word boundary
    last_space = truncated.rfind(" ")
    if last_space > limit // 2:
        return truncated[:last_space] + "..."

    return truncated + "..."


def wrap_text(text: str, width: int = 80) -> list[str]:
    """Wrap text to multiple lines at word boundaries."""
    if len(text) <= width:
        return [text]

    lines = []
    while len(text) > width:
        # Find last space before width
        wrap_pos = text.rfind(" ", 0, width)
        if wrap_pos == -1:  # No space found, hard wrap
            wrap_pos = width
        lines.append(text[:wrap_pos])
        text = text[wrap_pos:].lstrip()

    if text:
        lines.append(text)

    return lines


def _subcommand_tokens(subcommand: SubcommandPath | str | None) -> list[str]:
    if not subcommand:
        return []
    if isinstance(subcommand, str):
        try:
            return [tok for tok in shlex.split(subcommand) if tok]
        except ValueError:
            return [tok for tok in subcommand.split() if tok]
    return list(subcommand)


def get_help_text(
    *, command: str, subcommand: SubcommandPath | None = None
) -> str | None:
    """
    Get help output for a command.

    Tries multiple approaches: --help, -h, help subcommand.
    Uses command-specific overrides for complex commands like curl.
    """
    # Check for command-specific help overrides
    sub_tokens = _subcommand_tokens(subcommand)

    if command in HELP_SOURCE_OVERRIDES:
        help_source = HELP_SOURCE_OVERRIDES[command]
        cmd: list[str] | None = None
        if isinstance(help_source, list):
            cmd = help_source.copy()
        elif callable(help_source):
            maybe = help_source(tuple(sub_tokens))
            if isinstance(maybe, list):
                cmd = maybe

        if cmd is not None:
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=15,
                )
                output = result.stdout or result.stderr
                if output and len(output) > 50 and "-" in output:
                    return output
            except (subprocess.TimeoutExpired, FileNotFoundError, PermissionError):
                pass

    # Fall back to standard help options
    base_cmd = [command, *sub_tokens]

    # Try different help options
    help_variants = [
        base_cmd + ["--help"],
        base_cmd + ["-h"],
    ]
    # For some commands, help is a subcommand
    if not sub_tokens:
        help_variants.append([command, "help"])

    for cmd in help_variants:
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=15,  # Some commands (llama-cli) need time to initialize
            )
            output = result.stdout or result.stderr
            # Check if we got meaningful output (not just an error)
            if output and len(output) > 50 and "-" in output:
                return output
        except (subprocess.TimeoutExpired, FileNotFoundError, PermissionError):
            continue

    return None


def _is_man_error_output(*, text: str) -> bool:
    # Some man implementations write "No manual entry for ..." to stdout.
    # Treat short error messages as missing docs so we can fall back to --help.
    if not text or not text.strip():
        return True
    if len(text) > 300:
        return False
    lowered = text.strip().lower()
    patterns = [
        "no manual entry",
        "no entry for",
        "nothing appropriate",
        "man: no entry",
        "not found",
    ]
    return any(pat in lowered for pat in patterns)


def get_man_text(
    *, command: str, subcommand: SubcommandPath | None = None
) -> str | None:
    """
    Get man page text for a command.

    Uses col -b to strip formatting for cleaner output.
    """
    import os

    # Build man page name (git-commit for git commit, etc.)
    sub_tokens = _subcommand_tokens(subcommand)
    man_page = f"{command}-{'-'.join(sub_tokens)}" if sub_tokens else command

    # Try with col -b to strip backspaces
    try:
        # Use sh -c to pipe man through col
        result = subprocess.run(
            ["sh", "-c", f"man {man_page} 2>/dev/null | col -b"],
            capture_output=True,
            text=True,
            timeout=10,
            env={
                **os.environ,
                "MANPAGER": "cat",
                "PAGER": "cat",
                "MAN_KEEP_FORMATTING": "",
            },
        )
        if result.returncode == 0 and result.stdout:
            if not _is_man_error_output(text=result.stdout):
                return result.stdout
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    # Fallback: try without col
    try:
        result = subprocess.run(
            ["man", man_page],
            capture_output=True,
            text=True,
            timeout=10,
            env={**os.environ, "MANPAGER": "cat", "PAGER": "cat"},
        )
        if result.returncode == 0 and result.stdout:
            if not _is_man_error_output(text=result.stdout):
                return result.stdout
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    return None


def lookup_flag(
    *,
    help_provider: HelpProvider,
    command: str,
    flag: str,
    subcommand: SubcommandPath | None = None,
) -> str | None:
    """Look up the description for a specific flag.

    - Ensures durable JSON exists under `~/.config/fish/shellock/data/`.
    - Falls back from deeper subcommand paths to parent scopes when needed.
    - Supports combined short flags (e.g., `-rf`).
    """
    if subcommand:
        data = ensure_subcommand_scanned(
            help_provider=help_provider, command=command, subcommand=subcommand
        )
    else:
        data = ensure_command_scanned(
            help_provider=help_provider, command=command, refresh=False
        )

    return lookup_flag_from_data(
        data=data, command=command, flag=flag, subcommand=subcommand
    )


def _collect_known_flags(payload: dict) -> set[str]:
    raw_flags = payload.get("flags")
    flags = set(raw_flags.keys()) if isinstance(raw_flags, dict) else set()
    subcommands = payload.get("subcommands")
    if isinstance(subcommands, dict):
        for sub_payload in subcommands.values():
            if isinstance(sub_payload, dict):
                flags.update(_collect_known_flags(sub_payload))
    return flags


def explain_command(
    *, help_provider: HelpProvider, cmdline: str
) -> list[FlagDescription]:
    """Parse a command line and return descriptions for all flags.

    Non-blocking: if command data doesn't exist, spawns a background scan
    and returns a "Learning..." indicator instead of blocking.
    """
    parsed = parse_command_line(cmdline=cmdline)
    if not parsed:
        return []

    if not _command_exists(command=parsed.command):
        return []

    # Only trigger scans when the user has either started typing flags or has
    # completed the command token (i.e., typed a space right after it).
    try:
        tokens = shlex.split(cmdline)
    except ValueError:
        tokens = cmdline.split()
    trailing_space = cmdline != cmdline.rstrip()
    command_only = len(tokens) == 1
    has_flag_token = any(tok.startswith("-") for tok in tokens[1:])
    wants_scan = (
        bool(parsed.flags) or (trailing_space and command_only) or has_flag_token
    )

    # Non-blocking: check if data exists without triggering a scan
    data = load_command_data(command=parsed.command)

    if data is None:
        scan_in_progress = is_scan_in_progress(command=parsed.command)
        if not wants_scan and not scan_in_progress:
            return []
        # No data yet - spawn background scan if not already running.
        #
        # When invoked from the fish integration, fish can manage the scan as a
        # tracked child process and refresh the UI via `--on-process-exit`.
        # In that mode we skip detached background scans here to avoid
        # duplicate scans and to keep refresh behavior reliable.
        fish_managed = os.environ.get("SHELLOCK_FISH_MANAGED_SCAN")
        if not fish_managed and not scan_in_progress:
            spawn_background_scan(command=parsed.command)
        # Return "Learning..." indicator
        return [
            FlagDescription(
                flag="__scanning__", description=f"Learning {parsed.command}..."
            )
        ]

    if not parsed.flags:
        return []

    # Data exists - re-parse with known flags to avoid splitting
    # single-dash long flags like -ngl when they are real options.
    known_flags = _collect_known_flags(data)
    known_subcommands = data.get("subcommands", {})
    if not isinstance(known_subcommands, dict):
        known_subcommands = None

    parsed = (
        parse_command_line(
            cmdline=cmdline,
            known_flags=known_flags,
            known_subcommands=known_subcommands,
        )
        or parsed
    )

    # Proceed with lookup
    results: list[FlagDescription] = []
    seen: set[str] = set()

    for flag in parsed.flags:
        if flag.name in seen:
            continue
        seen.add(flag.name)

        desc = lookup_flag_from_data(
            data=data,
            command=parsed.command,
            flag=flag.name,
            subcommand=parsed.subcommands,
        )
        results.append(
            FlagDescription(flag=flag.name, description=desc or "Unknown flag")
        )

    return results


def lookup_flag_from_data(
    *,
    data: dict,
    command: str,
    flag: str,
    subcommand: SubcommandPath | None = None,
) -> str | None:
    """Look up flag description from already-loaded data (non-blocking)."""
    nodes: list[dict] = []
    if subcommand:
        cursor: dict | None = data
        for part in subcommand:
            if not isinstance(cursor, dict):
                break
            subcommands = cursor.get("subcommands")
            if not isinstance(subcommands, dict):
                break
            next_node = subcommands.get(part)
            if not isinstance(next_node, dict):
                break
            nodes.append(next_node)
            cursor = next_node

    search_nodes = list(reversed(nodes)) + [data]

    for node in search_nodes:
        flags = node.get("flags", {}) if isinstance(node, dict) else {}
        if not isinstance(flags, dict):
            flags = {}
        if flag in flags:
            return flags[flag]

        # Combined short flags: -rf => -r + -f
        if re.match(r"^-[a-zA-Z]{2,}$", flag):
            chars = flag[1:]
            parts: list[str] = []
            for char in chars:
                single_flag = f"-{char}"
                if single_flag in flags:
                    parts.append(f"{single_flag}: {flags[single_flag]}")
            if parts:
                return "Combination: " + " | ".join(parts)

    return None


def format_explanations(
    *,
    explanations: list[FlagDescription],
    use_color: bool = True,
) -> str:
    """Format flag explanations for terminal display."""
    if not explanations:
        return ""

    # Handle scanning indicator specially
    if len(explanations) == 1 and explanations[0].flag == "__scanning__":
        if use_color:
            return f"{DIM}  {explanations[0].description}{RESET}"
        return f"  {explanations[0].description}"

    lines: list[str] = []
    max_flag_len = max(len(e.flag) for e in explanations)

    for exp in explanations:
        flag_padded = exp.flag.ljust(max_flag_len)
        desc_lines = wrap_text(exp.description, width=80)

        # First line with flag
        if use_color:
            lines.append(
                f"{DIM}  {CYAN}{flag_padded}{RESET}  {DIM}{desc_lines[0]}{RESET}"
            )
        else:
            lines.append(f"  {flag_padded}  {desc_lines[0]}")

        # Continuation lines (indented to align with description)
        for cont_line in desc_lines[1:]:
            indent = " " * (max_flag_len + 4)  # 2 spaces + flag + 2 spaces
            if use_color:
                lines.append(f"{DIM}{indent}{cont_line}{RESET}")
            else:
                lines.append(f"{indent}{cont_line}")

    return "\n".join(lines)


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Real-time CLI flag explainer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-r",
        "--refresh",
        nargs="?",
        const="__prompt__",
        help="Refresh command data (use with a command or with -a/--all)",
    )
    parser.add_argument(
        "-a",
        "--all",
        action="store_true",
        help="With --refresh, refresh all known commands",
    )

    subparsers = parser.add_subparsers(dest="action")

    # parse command
    parse_p = subparsers.add_parser("parse", help="Parse flags from command line")
    parse_p.add_argument("cmdline", help="Command line to parse")

    # lookup command
    lookup_p = subparsers.add_parser("lookup", help="Look up a specific flag")
    lookup_p.add_argument("command", help="Command name")
    lookup_p.add_argument("flag", help="Flag to look up (e.g., -r or --raw)")
    lookup_p.add_argument(
        "--subcommand",
        nargs="+",
        help="Subcommand path (e.g., remote add for git)",
    )

    # explain command
    explain_p = subparsers.add_parser("explain", help="Explain all flags in command")
    explain_p.add_argument("cmdline", help="Command line to explain")
    explain_p.add_argument("--no-color", action="store_true", help="Disable colors")

    # refresh command (explicit)
    refresh_p = subparsers.add_parser("refresh", help="Refresh command data")
    refresh_p.add_argument("command", nargs="?", help="Command name")

    # scan command (for background scanning)
    scan_p = subparsers.add_parser("scan", help="Scan and cache command data")
    scan_p.add_argument("command", help="Command to scan")

    # generate command (LLM-based)
    model_default = None
    cfg = _config_get(key="llm_model")
    if isinstance(cfg, str) and cfg.strip():
        model_default = cfg.strip()
    gen_p = subparsers.add_parser(
        "generate",
        help="Generate and cache command data using an LLM agent (honors doc_order)",
    )
    gen_p.add_argument("command", help="Command to generate data for")
    gen_p.add_argument("--agent", default="claude", choices=["claude"])
    gen_p.add_argument("--model", default=model_default or "sonnet")
    gen_p.add_argument(
        "--max-subcommands",
        type=int,
        default=_setting_int(config_key="llm_max_subcommands", default=25),
    )
    gen_p.add_argument(
        "--max-depth",
        type=int,
        default=_max_subcommand_depth(),
        help="Maximum subcommand depth to scan (default from config)",
    )
    gen_p.add_argument(
        "--max-doc-chars",
        type=int,
        default=_setting_int(config_key="llm_max_doc_chars", default=30000),
    )
    gen_p.add_argument(
        "--timeout-s",
        type=int,
        default=_setting_int(config_key="llm_timeout_s", default=180),
    )
    gen_p.add_argument(
        "--print",
        action="store_true",
        help="Print the generated JSON to stdout (also saves to data dir)",
    )

    args = parser.parse_args()

    help_provider = DefaultHelpProvider()

    # Handle top-level refresh convenience flags:
    #   shellock -r tree
    #   shellock -r -a
    if args.refresh is not None:
        if args.all:
            refresh_all(help_provider=help_provider)
            return 0

        if args.refresh == "__prompt__":
            print("Missing command for --refresh", file=sys.stderr)
            return 2

        refresh_command(help_provider=help_provider, command=args.refresh)
        return 0

    if args.action is None:
        parser.print_usage(sys.stderr)
        return 2

    if args.action == "parse":
        result = parse_command_line(cmdline=args.cmdline)
        if result:
            subcommand_path = list(result.subcommands)
            subcommand_label = (
                " ".join(result.subcommands) if result.subcommands else None
            )
            print(
                json.dumps(
                    {
                        "command": result.command,
                        "subcommand": subcommand_label,
                        "subcommands": subcommand_path,
                        "flags": [
                            {"name": f.name, "is_long": f.is_long} for f in result.flags
                        ],
                    }
                )
            )
        else:
            print("{}")
        return 0

    elif args.action == "lookup":
        subcommand_path: SubcommandPath | None = None
        if args.subcommand:
            tokens: list[str] = []
            for item in args.subcommand:
                tokens.extend(_subcommand_tokens(item))
            if tokens:
                subcommand_path = tuple(tokens)
        desc = lookup_flag(
            help_provider=help_provider,
            command=args.command,
            flag=args.flag,
            subcommand=subcommand_path,
        )
        if desc:
            print(desc)
            return 0
        else:
            print("Unknown flag", file=sys.stderr)
            return 1

    elif args.action == "explain":
        explanations = explain_command(
            help_provider=help_provider, cmdline=args.cmdline
        )
        output = format_explanations(
            explanations=explanations, use_color=not args.no_color
        )
        if output:
            print(output)
        return 0

    elif args.action == "refresh":
        if args.all:
            refresh_all(help_provider=help_provider)
            return 0
        if not args.command:
            print("Missing command for refresh", file=sys.stderr)
            return 2
        refresh_command(help_provider=help_provider, command=args.command)
        return 0

    elif args.action == "scan":
        # Background scan with lock file to prevent duplicate scans
        lock_path = command_data_path(command=args.command).with_suffix(".scanning")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            lock_path.touch()
            ensure_command_scanned(
                help_provider=help_provider, command=args.command, refresh=False
            )
            # When fish manages the scan as a child process, it can refresh via
            # `--on-process-exit` and doesn't need the universal-variable signal.
            if not os.environ.get("SHELLOCK_FISH_MANAGED_SCAN"):
                _notify_fish_scan_complete(command=args.command)
        finally:
            lock_path.unlink(missing_ok=True)
        return 0

    elif args.action == "generate":
        if args.agent != "claude":
            print(f"Unsupported agent: {args.agent}", file=sys.stderr)
            return 2

        agent = ClaudeCodeAgent(model=args.model)
        data = generate_command_data_llm(
            help_provider=help_provider,
            agent=agent,
            command=args.command,
            max_subcommands=max(0, args.max_subcommands),
            max_doc_chars=max(1, args.max_doc_chars),
            timeout_s=max(1, args.timeout_s),
            max_depth=max(0, args.max_depth),
        )
        save_command_data(command=args.command, data=data)
        if args.print:
            print(json.dumps(data, indent=2, sort_keys=True))
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
