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
import json
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Final, Protocol, TypedDict

# Constants
PROTOCOL_VERSION: Final[int] = 1


class Sources(TypedDict):
    help: bool
    man: bool


class SubcommandData(TypedDict):
    generated_at: str
    sources: Sources
    flags: dict[str, str]


class CommandData(TypedDict):
    protocol_version: int
    command: str
    generated_at: str
    sources: Sources
    flags: dict[str, str]
    subcommands: dict[str, SubcommandData]


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
                "additionalProperties": {
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
                    },
                    "required": ["generated_at", "sources", "flags"],
                },
            },
        },
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
        },
        "required": ["generated_at", "sources", "flags"],
    }


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
    executable: str = "claude"

    def run_json(
        self, *, prompt: str, json_schema: dict, timeout_s: int | None = None
    ) -> object:
        schema_arg = json.dumps(json_schema, separators=(",", ":"), sort_keys=True)
        cmd = [
            self.executable,
            "-p",
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
            raise LlmAgentError(
                f"Agent executable not found: {self.executable}"
            ) from e
        except subprocess.TimeoutExpired as e:
            raise LlmAgentError("Agent call timed out") from e

        if result.returncode != 0:
            stderr = (result.stderr or "").strip()
            raise LlmAgentError(
                f"Agent failed (exit {result.returncode}): {stderr or 'no stderr'}"
            )

        raw = (result.stdout or "").strip()
        if not raw:
            raise LlmAgentError("Agent returned empty output")
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            try:
                return _parse_first_json_value(text=raw)
            except ValueError as e:
                raise LlmAgentError(
                    f"Agent output was not valid JSON: {raw[:4000]}"
                ) from e


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
        if not isinstance(sub_payload, dict):
            raise ValueError("subcommand payload must be an object")
        sub_generated_at = sub_payload.get("generated_at")
        if not isinstance(sub_generated_at, str) or not sub_generated_at:
            raise ValueError("subcommand generated_at must be a non-empty string")
        sub_sources = _validate_sources(sources=sub_payload.get("sources"))
        sub_flags = _validate_flags(flags=sub_payload.get("flags"))
        subcommands[sub_name] = {
            "generated_at": sub_generated_at,
            "sources": sub_sources,
            "flags": sub_flags,
        }

    return {
        "protocol_version": PROTOCOL_VERSION,
        "command": command,
        "generated_at": generated_at,
        "sources": sources,
        "flags": flags,
        "subcommands": subcommands,
    }


def validate_subcommand_data(*, payload: object) -> SubcommandData:
    if not isinstance(payload, dict):
        raise ValueError("payload must be an object")
    generated_at = payload.get("generated_at")
    if not isinstance(generated_at, str) or not generated_at:
        raise ValueError("generated_at must be a non-empty string")
    sources = _validate_sources(sources=payload.get("sources"))
    flags = _validate_flags(flags=payload.get("flags"))
    return {"generated_at": generated_at, "sources": sources, "flags": flags}

# ANSI colors for terminal output
DIM: Final[str] = "\033[2m"
RESET: Final[str] = "\033[0m"
CYAN: Final[str] = "\033[36m"

# Command-specific help overrides
# Some commands need special help invocations to show all flags
HELP_SOURCE_OVERRIDES: Final[dict[str, object]] = {
    "curl": ["curl", "--help", "all"],
    "git": lambda subcmd: ["git", "help", subcmd] if subcmd else ["git", "-h"],
    "docker": lambda subcmd: ["docker", subcmd, "--help"]
    if subcmd
    else ["docker", "--help"],
    "kubectl": lambda subcmd: ["kubectl", subcmd, "--help"]
    if subcmd
    else ["kubectl", "--help"],
    "npm": lambda subcmd: ["npm", subcmd, "--help"] if subcmd else ["npm", "--help"],
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
    return _load_config().get(key)


def _command_file_name(command: str) -> str:
    """Use the command token as the name, sanitized for filesystem."""
    safe = command.replace("/", "_").replace("\\", "_")
    safe = safe.replace(":", "_")
    safe = re.sub(r"\s+", "_", safe)
    return safe


def command_data_path(*, command: str) -> Path:
    return shellock_data_dir() / f"{_command_file_name(command)}.json"


class HelpProvider(Protocol):
    def get_help_text(self, *, command: str, subcommand: str | None) -> str | None: ...

    def get_man_text(self, *, command: str, subcommand: str | None) -> str | None: ...


class FlagExtractor(Protocol):
    def extract_from_help(self, *, help_text: str) -> dict[str, str]: ...

    def extract_from_man(self, *, man_text: str) -> dict[str, str]: ...


class SubcommandDiscoverer(Protocol):
    def discover(
        self, *, command: str, help_text: str | None, man_text: str | None
    ) -> set[str]: ...


@dataclass(frozen=True, slots=True)
class ScanResult:
    flags: dict[str, str]
    sources: dict[str, bool]
    discovered_subcommands: set[str]


@dataclass(frozen=True, slots=True)
class CommandProtocol:
    help_provider: HelpProvider
    extractor: FlagExtractor
    subcommands: SubcommandDiscoverer


class DefaultHelpProvider:
    def get_help_text(self, *, command: str, subcommand: str | None) -> str | None:
        return get_help_text(command=command, subcommand=subcommand)

    def get_man_text(self, *, command: str, subcommand: str | None) -> str | None:
        return get_man_text(command=command, subcommand=subcommand)


class RegexExtractor:
    def extract_from_help(self, *, help_text: str) -> dict[str, str]:
        return parse_help_output(help_text=help_text)

    def extract_from_man(self, *, man_text: str) -> dict[str, str]:
        return parse_man_page(man_text=man_text)


class HeuristicSubcommandDiscoverer:
    _section_markers = re.compile(
        r"(?im)^\s*(commands|available commands|subcommands)\s*:?:\s*$"
    )

    def discover(
        self, *, command: str, help_text: str | None, man_text: str | None
    ) -> set[str]:
        text = "\n".join([t for t in [help_text, man_text] if t])
        if not text:
            return set()

        discovered: set[str] = set()

        list_entry = re.compile(r"^\s{2,}([a-z0-9][a-z0-9-]*)\s{2,}.*$", re.IGNORECASE)

        in_section = False
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                if in_section:
                    continue
                continue

            if self._section_markers.match(line):
                in_section = True
                continue

            if in_section and re.match(r"^[A-Z][A-Z0-9 _-]+$", stripped):
                in_section = False
                continue

            match = list_entry.match(line)
            if match:
                token = match.group(1)
                if token.startswith("-"):
                    continue
                if token in {"options", "usage", "help"}:
                    continue
                if token == command:
                    continue
                discovered.add(token)

        return discovered


@dataclass(frozen=True, slots=True)
class HelpAndMan:
    help_text: str | None
    man_text: str | None


def _get_help_and_man(
    *, protocol: CommandProtocol, command: str, subcommand: str | None
) -> HelpAndMan:
    return HelpAndMan(
        help_text=protocol.help_provider.get_help_text(
            command=command, subcommand=subcommand
        ),
        man_text=protocol.help_provider.get_man_text(
            command=command, subcommand=subcommand
        ),
    )


def _sources_from_docs(*, docs: HelpAndMan) -> Sources:
    return {"help": bool(docs.help_text), "man": bool(docs.man_text)}


def _primary_doc_text(*, docs: HelpAndMan) -> tuple[str, str] | None:
    if docs.man_text:
        return ("man", docs.man_text)
    if docs.help_text:
        return ("help", docs.help_text)
    return None


def _truncate_for_prompt(*, text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n\n[... truncated ...]\n"


def generate_command_data_llm(
    *,
    protocol: CommandProtocol,
    agent: StructuredJsonAgent,
    command: str,
    max_subcommands: int,
    max_doc_chars: int,
    timeout_s: int,
) -> CommandData:
    """Generate and persist command JSON using an external LLM agent.

    Priority: man page if present, otherwise help output.
    """
    root_docs = _get_help_and_man(protocol=protocol, command=command, subcommand=None)
    root_primary = _primary_doc_text(docs=root_docs)
    if root_primary is None:
        raise LlmAgentError(f"No man page or help output found for: {command}")

    discovered = sorted(
        protocol.subcommands.discover(
            command=command, help_text=root_docs.help_text, man_text=root_docs.man_text
        )
    )
    subcommands = discovered[: max(0, max_subcommands)]

    now = _utc_now_iso()

    prompt_parts: list[str] = []
    prompt_parts.append(
        "\n".join(
            [
                "You are given documentation text for a CLI command and some of its subcommands.",
                "Produce a single JSON value that matches the provided JSON Schema exactly.",
                "",
                "Extraction rules:",
                "- Do not invent flags, subcommands, or descriptions.",
                "- Only include options/flags (typically tokens that start with '-' or '--').",
                "- Use the flag token only (omit argument placeholders like '<path>' or '=VALUE').",
                "- Keep descriptions concise (ideally <= 160 chars).",
                "",
                f"Target command: {command}",
                f"protocol_version must be: {PROTOCOL_VERSION}",
                f"generated_at must be: {now}",
                "",
                "Fill `sources` booleans based on whether the docs are present (help/man).",
                "Prefer man pages when both help and man are available.",
                "",
            ]
        )
    )

    root_kind, root_text = root_primary
    prompt_parts.append(f"== {command} ({root_kind}) ==")
    prompt_parts.append(_truncate_for_prompt(text=root_text, max_chars=max_doc_chars))

    sub_docs: dict[str, HelpAndMan] = {}
    for sub in subcommands:
        docs = _get_help_and_man(protocol=protocol, command=command, subcommand=sub)
        primary = _primary_doc_text(docs=docs)
        if primary is None:
            continue
        sub_docs[sub] = docs
        kind, text = primary
        prompt_parts.append(f"== {command} {sub} ({kind}) ==")
        prompt_parts.append(_truncate_for_prompt(text=text, max_chars=max_doc_chars))

    prompt = "\n".join(prompt_parts).strip() + "\n"

    candidate = agent.run_json(
        prompt=prompt,
        json_schema=command_data_json_schema(),
        timeout_s=timeout_s,
    )

    if not isinstance(candidate, dict):
        raise LlmAgentError("Agent output was not a JSON object")

    # Normalize/override non-LLM fields to ensure determinism.
    candidate["protocol_version"] = PROTOCOL_VERSION
    candidate["command"] = command
    candidate["generated_at"] = now
    candidate["sources"] = _sources_from_docs(docs=root_docs)

    if not isinstance(candidate.get("flags"), dict):
        candidate["flags"] = {}

    raw_subs = candidate.get("subcommands")
    if not isinstance(raw_subs, dict):
        raw_subs = {}

    normalized_subs: dict[str, dict] = {}
    for sub in sub_docs.keys():
        sub_payload = raw_subs.get(sub)
        if not isinstance(sub_payload, dict):
            sub_payload = {}
        if not isinstance(sub_payload.get("flags"), dict):
            sub_payload["flags"] = {}
        sub_payload["generated_at"] = now
        sub_payload["sources"] = _sources_from_docs(docs=sub_docs[sub])
        normalized_subs[sub] = sub_payload

    candidate["subcommands"] = normalized_subs

    return validate_command_data(payload=candidate, command=command)


def generate_subcommand_data_llm(
    *,
    protocol: CommandProtocol,
    agent: StructuredJsonAgent,
    command: str,
    subcommand: str,
    max_doc_chars: int,
    timeout_s: int,
) -> SubcommandData:
    docs = _get_help_and_man(protocol=protocol, command=command, subcommand=subcommand)
    primary = _primary_doc_text(docs=docs)
    if primary is None:
        raise LlmAgentError(
            f"No man page or help output found for: {command} {subcommand}"
        )
    now = _utc_now_iso()
    kind, text = primary
    prompt = (
        "\n".join(
            [
                "You are given documentation text for a CLI subcommand.",
                "Produce a single JSON value that matches the provided JSON Schema exactly.",
                "",
                "Extraction rules:",
                "- Do not invent flags or descriptions.",
                "- Only include options/flags (typically tokens that start with '-' or '--').",
                "- Use the flag token only (omit argument placeholders like '<path>' or '=VALUE').",
                "- Keep descriptions concise (ideally <= 160 chars).",
                "",
                f"Target: {command} {subcommand}",
                f"generated_at must be: {now}",
                "",
                f"== {command} {subcommand} ({kind}) ==",
                _truncate_for_prompt(text=text, max_chars=max_doc_chars),
            ]
        ).strip()
        + "\n"
    )

    candidate = agent.run_json(
        prompt=prompt,
        json_schema=subcommand_data_json_schema(),
        timeout_s=timeout_s,
    )

    if not isinstance(candidate, dict):
        raise LlmAgentError("Agent output was not a JSON object")

    candidate["generated_at"] = now
    candidate["sources"] = _sources_from_docs(docs=docs)
    if not isinstance(candidate.get("flags"), dict):
        candidate["flags"] = {}

    return validate_subcommand_data(payload=candidate)


def _scan_backend() -> str:
    cfg = _config_get(key="scan_backend")
    if isinstance(cfg, str) and cfg.strip():
        return cfg.strip().lower()
    return "llm"


def _setting_int(*, config_key: str, default: int) -> int:
    cfg = _config_get(key=config_key)
    if cfg is None:
        return default
    try:
        return int(cfg)
    except (TypeError, ValueError):
        return default


def _llm_agent_from_env() -> StructuredJsonAgent:
    cfg = _config_get(key="llm_model")
    model = cfg.strip() if isinstance(cfg, str) and cfg.strip() else "sonnet"
    return ClaudeCodeAgent(model=model)


def _scan_flags_and_subcommands(
    *, protocol: CommandProtocol, command: str, subcommand: str | None
) -> ScanResult:
    help_text = protocol.help_provider.get_help_text(
        command=command, subcommand=subcommand
    )
    man_text = protocol.help_provider.get_man_text(
        command=command, subcommand=subcommand
    )

    flags: dict[str, str] = {}
    sources = {"help": False, "man": False}

    if help_text:
        sources["help"] = True
        flags.update(protocol.extractor.extract_from_help(help_text=help_text))

    if man_text:
        sources["man"] = True
        for k, v in protocol.extractor.extract_from_man(man_text=man_text).items():
            flags.setdefault(k, v)

    discovered_subcommands = protocol.subcommands.discover(
        command=command, help_text=help_text, man_text=man_text
    )

    return ScanResult(
        flags=flags, sources=sources, discovered_subcommands=discovered_subcommands
    )


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


def ensure_command_scanned(
    *, protocol: CommandProtocol, command: str, refresh: bool
) -> dict:
    data = load_command_data(command=command)

    if data is None or refresh:
        backend = _scan_backend()
        if backend in {"llm", "llm-only", "auto"}:
            try:
                llm_data = generate_command_data_llm(
                    protocol=protocol,
                    agent=_llm_agent_from_env(),
                    command=command,
                    max_subcommands=_setting_int(
                        config_key="llm_max_subcommands",
                        default=25,
                    ),
                    max_doc_chars=_setting_int(
                        config_key="llm_max_doc_chars",
                        default=30000,
                    ),
                    timeout_s=_setting_int(
                        config_key="llm_timeout_s",
                        default=180,
                    ),
                )
                save_command_data(command=command, data=llm_data)
                return llm_data
            except LlmAgentError:
                if backend == "llm-only":
                    raise

        scan = _scan_flags_and_subcommands(
            protocol=protocol, command=command, subcommand=None
        )
        subcommands: dict[str, dict] = {}

        for sub in sorted(scan.discovered_subcommands):
            sub_scan = _scan_flags_and_subcommands(
                protocol=protocol, command=command, subcommand=sub
            )
            subcommands[sub] = {
                "generated_at": _utc_now_iso(),
                "sources": sub_scan.sources,
                "flags": sub_scan.flags,
            }

        data = {
            "protocol_version": PROTOCOL_VERSION,
            "command": command,
            "generated_at": _utc_now_iso(),
            "sources": scan.sources,
            "flags": scan.flags,
            "subcommands": subcommands,
        }
        save_command_data(command=command, data=data)
        return data

    return data


def ensure_subcommand_scanned(
    *, protocol: CommandProtocol, command: str, subcommand: str
) -> dict:
    data = ensure_command_scanned(protocol=protocol, command=command, refresh=False)
    subcommands = data.get("subcommands", {})

    if subcommand not in subcommands:
        backend = _scan_backend()
        if backend in {"llm", "llm-only", "auto"}:
            try:
                sub_data = generate_subcommand_data_llm(
                    protocol=protocol,
                    agent=_llm_agent_from_env(),
                    command=command,
                    subcommand=subcommand,
                    max_doc_chars=_setting_int(
                        config_key="llm_max_doc_chars",
                        default=30000,
                    ),
                    timeout_s=_setting_int(
                        config_key="llm_timeout_s",
                        default=180,
                    ),
                )
                subcommands[subcommand] = sub_data
            except LlmAgentError:
                if backend == "llm-only":
                    raise
        if subcommand not in subcommands:
            sub_scan = _scan_flags_and_subcommands(
                protocol=protocol, command=command, subcommand=subcommand
            )
            subcommands[subcommand] = {
                "generated_at": _utc_now_iso(),
                "sources": sub_scan.sources,
                "flags": sub_scan.flags,
            }
        data["subcommands"] = subcommands
        save_command_data(command=command, data=data)

    return data


def refresh_command(*, protocol: CommandProtocol, command: str) -> None:
    existing = load_command_data(command=command) or {}

    backend = _scan_backend()
    if backend in {"llm", "llm-only", "auto"}:
        try:
            llm_data = generate_command_data_llm(
                protocol=protocol,
                agent=_llm_agent_from_env(),
                command=command,
                max_subcommands=_setting_int(
                    config_key="llm_max_subcommands",
                    default=25,
                ),
                max_doc_chars=_setting_int(
                    config_key="llm_max_doc_chars",
                    default=30000,
                ),
                timeout_s=_setting_int(
                    config_key="llm_timeout_s",
                    default=180,
                ),
            )
            save_command_data(command=command, data=llm_data)
            return
        except LlmAgentError:
            if backend == "llm-only":
                raise

    scan = _scan_flags_and_subcommands(
        protocol=protocol, command=command, subcommand=None
    )

    recorded_subs = set((existing.get("subcommands") or {}).keys())
    discovered_subs = set(scan.discovered_subcommands)
    all_subs = sorted(recorded_subs | discovered_subs)

    subcommands: dict[str, dict] = {}
    for sub in all_subs:
        sub_scan = _scan_flags_and_subcommands(
            protocol=protocol, command=command, subcommand=sub
        )
        subcommands[sub] = {
            "generated_at": _utc_now_iso(),
            "sources": sub_scan.sources,
            "flags": sub_scan.flags,
        }

    data = {
        "protocol_version": PROTOCOL_VERSION,
        "command": command,
        "generated_at": _utc_now_iso(),
        "sources": scan.sources,
        "flags": scan.flags,
        "subcommands": subcommands,
    }

    save_command_data(command=command, data=data)


def refresh_all(*, protocol: CommandProtocol) -> None:
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
            refresh_command(protocol=protocol, command=cmd)


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
    subcommand: str | None = None


def parse_command_line(*, cmdline: str) -> ParsedCommand | None:
    """
    Parse a command line string to extract command and flags.

    Handles:
    - Short flags: -r, -v, -rf (combined)
    - Long flags: --raw-output, --verbose
    - Subcommands: git commit -m
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
    subcommand: str | None = None

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
        if (
            command in subcommand_commands
            and subcommand is None
            and not token.startswith("-")
            and "=" not in token  # key=value pairs are likely flag values
        ):
            subcommand = token
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
            flag_part = token[1:]
            if len(flag_part) >= 4 and flag_part.isalpha() and flag_part.islower():
                # Likely a single-dash long flag (e.g., -name, -type, -exec)
                flags.append(Flag(name=token, is_long=False))
            else:
                # Short flag(s): -r or -rf (combined)
                for char in token[1:]:
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
        subcommand=subcommand,
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


def strip_man_formatting(*, text: str) -> str:
    """
    Strip man page formatting (bold/underline via backspace sequences).

    Man pages use X\bX for bold and _\bX for underline.
    """
    # Remove backspace sequences (X\bX -> X, _\bX -> X)
    result = re.sub(r".\x08", "", text)

    # Note: We don't dedupe doubled characters (like NNAAMMEE -> NAME)
    # because col -b already handles this, and blind deduplication
    # can corrupt normal text like "commit" -> "comit"

    return result


def parse_help_output(*, help_text: str) -> dict[str, str]:
    """
    Parse --help output to extract flag descriptions.

    Handles common formats:
    - "-r, --raw-output  description"
    - "-r          description"
    - "  --flag    description"
    """
    flags: dict[str, str] = {}

    # Clean up man page formatting if present
    help_text = strip_man_formatting(text=help_text)

    # Pattern for flags with descriptions
    patterns = [
        # llama-cli style: -m,    --model FNAME                    description
        r"^\s*(-\w+),\s+(--[\w-]+)\s+\S+\s{2,}(.+)$",
        # llama-cli style without arg: -h,    --help, --usage       description
        r"^\s*(-\w+),\s+(--[\w-]+)(?:,\s+--[\w-]+)*\s{2,}(.+)$",
        # claude style: --camelCase, --kebab-case <arg>  description
        r"^\s*(--[\w]+),\s+(--[\w-]+)\s+<[^>]+>\s{2,}(.+)$",
        # GNU style: -x, --long-option  Description
        r"^\s*(-\w)(?:[,\s]+(--[\w-]+))?\s{2,}(.+)$",
        # Long then short: --option, -x  Description
        r"^\s*(--[\w-]+)(?:[,\s]+(-\w))?\s{2,}(.+)$",
        # With equals and optional value: --option[=VALUE]  Description
        r"^\s*(--[\w-]+)(?:\[?=\S*\]?)?\s{2,}(.+)$",
        # Long flag with positional arg: --temp N  description
        r"^\s*(--[\w-]+)\s+\S+\s{2,}(.+)$",
        # Short flag with arg: -x arg  description
        r"^\s*(-[^\s])\s+\S+\s{2,}(.+)$",
        # Short only: -x  Description
        r"^\s*(-[^\s])\s{2,}(.+)$",
        # Brackets style: [-x]  Description
        r"^\s*\[(-[^\s])\]\s{2,}(.+)$",
    ]

    # Special handling for lines with multiple mode flags (e.g., tar -c Create -r Add/Replace)
    import re

    mode_flags = {}
    for line in help_text.split("\n"):
        line = line.rstrip()
        # Look for mode flag patterns like: -c Create  -r Add/Replace
        if re.search(r"-\w\s+[A-Z][a-z]+", line):
            # Split by flag indicators
            parts = re.finditer(r"(-[^\s,\s]+)\s+([A-Z][a-zA-Z]+(?:\s+\w+)*)", line)
            for match in parts:
                flag, desc = match.groups()
                mode_flags[flag] = truncate_description(desc)

    # Add mode flags to the results
    flags.update(mode_flags)

    for line in help_text.split("\n"):
        line = line.rstrip()

        for pattern in patterns:
            match = re.match(pattern, line)
            if match:
                groups = match.groups()
                if len(groups) == 3:
                    first, second, desc = groups
                    desc = truncate_description(desc.strip())
                    if first:
                        flags[first] = desc
                    if second:
                        flags[second] = desc
                elif len(groups) == 2:
                    flag, desc = groups
                    flags[flag] = truncate_description(desc.strip())
                break

    return flags


def parse_man_page(*, man_text: str) -> dict[str, str]:
    """
    Parse man page to extract flag descriptions.

    Man pages have various formats, often in OPTIONS section.
    Handles git-style man pages where description is on next line.
    """
    flags: dict[str, str] = {}

    # Clean up formatting
    man_text = strip_man_formatting(text=man_text)

    # Try to find OPTIONS section
    options_match = re.search(
        r"^OPTIONS?\s*$(.+?)(?=^[A-Z]+\s*$|\Z)",
        man_text,
        re.MULTILINE | re.DOTALL,
    )
    if options_match:
        options_text = options_match.group(1)
    else:
        options_text = man_text

    # Patterns to extract flags from a line
    # These patterns handle various man page formats:
    # - Git style: exactly 7 spaces, flags on own line, tab-indented description below
    # - GNU style: -x, --long-option (on own line, description below)
    # - BSD style: -x      Description on same line
    # - Multiple flags: -R, -r, --recursive
    # - Tree style: -L level (arg without angle brackets)
    flag_line_patterns = [
        # Git style: exactly 7 spaces, -x, --long-option
        r"^(       )(-[^\s])(?:,\s+(--[\w-]+))?\s*$",
        # Git style: -x <arg>, --long-option=<arg> (like -C <path>)
        r"^(       )(-[^\s])\s+<[^>]+>(?:,\s+(--[\w-]+)(?:=<[^>]+>)?)?\s*$",
        # Git style: -x <name>=<value> (like -c <name>=<value>)
        r"^(       )(-[^\s])\s+<[^>]+=<[^>]+>\s*$",
        # Git style: --long-option only (like --bare)
        r"^(       )(--[\w-]+)(?:\[?=<[^>]+>\]?)?\s*$",
        # Git style: --config-env=<name>=<envvar>
        r"^(       )(--[\w-]+=)<[^>]+=<[^>]+>\s*$",
        # GNU: -x <arg>, --long-option=<arg> on own line
        r"^(\s{1,12})(-[^\s])(?:\s+<\S+>)?(?:,\s*(--[\w-]+)(?:=<\S+>)?)?\s*$",
        # GNU: --long-option=<arg>, -x <arg> on own line
        r"^(\s{1,12})(--[\w-]+)(?:=<\S+>)?(?:,\s*(-[^\s])(?:\s+<\S+>)?)?\s*$",
        # GNU: --long-option on own line
        r"^(\s{1,12})(--[\w-]+)(?:=\S+)?\s*$",
        # Multiple short/long flags: -R, -r, --recursive (no description)
        r"^(\s{1,12})(-[^\s])(?:,\s*(-[^\s]))?(?:,\s*(--[\w-]+))?\s*$",
        # Tree style: -X arg (arg without brackets, on own line, may have no indent)
        r"^(\s*)(-[^\s])\s+[a-z]+\s*$",
        # Find style: -name pattern (single-dash long flag with arg, 4+ letters)
        r"^(\s*)(-[a-z]{4,})\s+[a-z]+\s*$",
        # Find style with XY suffix: -newerXY reference
        r"^(\s*)(-[a-z]+(?:XY|[A-Z]{2}))\s+\S+\s*$",
        # Find style with complex args: -exec utility [argument ...] ;
        r"^(\s*)(-[a-z]{4,})\s+\S+",
        # BSD style: -x      Description (6+ spaces before description)
        r"^(\s{1,12})(-[^\s])\s{4,}(\S.*)$",
        # Long flag with description on same line
        r"^(\s{1,12})(--[\w-]+)\s{4,}(\S.*)$",
        # Multiple flags with description on same line (BSD/GNU style)
        r"^(\s*)(-\w)(?:,\s*(-\w))*\s{4,}(\S.*)$",
    ]

    current_flags: list[str] = []
    current_desc: list[str] = []
    current_indent = 0

    lines = options_text.split("\n")
    i = 0

    while i < len(lines):
        line = lines[i]

        # Check if this line starts a new flag definition
        is_flag_line = False
        for pattern in flag_line_patterns:
            match = re.match(pattern, line)
            if match:
                # Found a flag line
                # Save previous flag(s)
                if current_flags and current_desc:
                    desc = truncate_description(" ".join(current_desc).strip())
                    for f in current_flags:
                        flags[f] = desc

                # Extract flags and possibly description from this line
                # First group is always indent
                groups = match.groups()
                current_indent = len(groups[0]) if groups[0] else 0

                # Separate flags (start with -) from potential description
                current_flags = []
                inline_desc = None
                for g in groups[1:]:
                    if g:
                        if g.startswith("-"):
                            current_flags.append(g)
                        elif not g.isspace():
                            # This is likely an inline description (BSD style)
                            inline_desc = g

                if inline_desc:
                    current_desc = [inline_desc]
                else:
                    current_desc = []

                is_flag_line = True
                break

        if not is_flag_line and current_flags:
            # Check if this is a description line
            # Description lines are more indented than the flag line
            # Common patterns: tab-indented, or many spaces
            stripped = line.strip()
            # Skip lines that look like new flag definitions
            if stripped and not re.match(r"^-\w", stripped):
                # Check indentation - description should be more indented than flag
                leading_spaces = len(line) - len(line.lstrip())
                if line.startswith("\t") or leading_spaces > current_indent + 2:
                    current_desc.append(stripped)

        i += 1

    # Save last flag(s)
    if current_flags and current_desc:
        desc = truncate_description(" ".join(current_desc).strip())
        for f in current_flags:
            flags[f] = desc

    return flags


def get_help_text(*, command: str, subcommand: str | None = None) -> str | None:
    """
    Get help output for a command.

    Tries multiple approaches: --help, -h, help subcommand.
    Uses command-specific overrides for complex commands like curl.
    """
    # Check for command-specific help overrides
    if command in HELP_SOURCE_OVERRIDES:
        help_source = HELP_SOURCE_OVERRIDES[command]
        cmd: list[str] | None = None
        if isinstance(help_source, list):
            cmd = help_source.copy()
        elif callable(help_source):
            maybe = help_source(subcommand)
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
    base_cmd = [command]
    if subcommand:
        base_cmd.append(subcommand)

    # Try different help options
    help_variants = [
        base_cmd + ["--help"],
        base_cmd + ["-h"],
    ]
    # For some commands, help is a subcommand
    if not subcommand:
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


def get_man_text(*, command: str, subcommand: str | None = None) -> str | None:
    """
    Get man page text for a command.

    Uses col -b to strip formatting for cleaner output.
    """
    import os

    # Build man page name (git-commit for git commit, etc.)
    man_page = f"{command}-{subcommand}" if subcommand else command

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
        if result.returncode == 0:
            return result.stdout
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    return None


def lookup_flag(
    *, protocol: CommandProtocol, command: str, flag: str, subcommand: str | None = None
) -> str | None:
    """Look up the description for a specific flag.

    - Ensures durable JSON exists under `~/.config/fish/shellock/data/`.
    - Falls back from `(command, subcommand)` to `(command)` when needed.
    - Supports combined short flags (e.g., `-rf`).
    """
    if subcommand is None:
        data = ensure_command_scanned(protocol=protocol, command=command, refresh=False)
        flags = data.get("flags", {})
    else:
        data = ensure_subcommand_scanned(
            protocol=protocol, command=command, subcommand=subcommand
        )
        flags = (data.get("subcommands", {}).get(subcommand, {}) or {}).get("flags", {})

    if flag in flags:
        return flags[flag]

    if subcommand is not None:
        # fall back to parent command
        return lookup_flag(
            protocol=protocol, command=command, flag=flag, subcommand=None
        )

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


def explain_command(
    *, protocol: CommandProtocol, cmdline: str
) -> list[FlagDescription]:
    """Parse a command line and return descriptions for all flags.

    Non-blocking: if command data doesn't exist, spawns a background scan
    and returns a "Learning..." indicator instead of blocking.
    """
    parsed = parse_command_line(cmdline=cmdline)
    if not parsed:
        return []

    # Non-blocking: check if data exists without triggering a scan
    data = load_command_data(command=parsed.command)

    if data is None:
        # No data yet - spawn background scan if not already running
        if not is_scan_in_progress(command=parsed.command):
            spawn_background_scan(command=parsed.command)
        # Return "Learning..." indicator
        return [
            FlagDescription(
                flag="__scanning__", description=f"Learning {parsed.command}..."
            )
        ]

    # Data exists - proceed with lookup
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
            subcommand=parsed.subcommand,
        )
        results.append(
            FlagDescription(flag=flag.name, description=desc or "Unknown flag")
        )

    return results


def lookup_flag_from_data(
    *, data: dict, command: str, flag: str, subcommand: str | None = None
) -> str | None:
    """Look up flag description from already-loaded data (non-blocking)."""
    if subcommand:
        subcommand_data = data.get("subcommands", {}).get(subcommand, {})
        if subcommand_data:
            flags = subcommand_data.get("flags", {})
            if flag in flags:
                return flags[flag]
        # Fall back to main command flags

    flags = data.get("flags", {})
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


def _default_protocol() -> CommandProtocol:
    return CommandProtocol(
        help_provider=DefaultHelpProvider(),
        extractor=RegexExtractor(),
        subcommands=HeuristicSubcommandDiscoverer(),
    )


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
    lookup_p.add_argument("--subcommand", help="Subcommand (e.g., commit for git)")

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
        help="Generate and cache command data using an LLM agent (prefers man pages)",
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

    protocol = _default_protocol()

    # Handle top-level refresh convenience flags:
    #   shellock -r tree
    #   shellock -r -a
    if args.refresh is not None:
        if args.all:
            refresh_all(protocol=protocol)
            return 0

        if args.refresh == "__prompt__":
            print("Missing command for --refresh", file=sys.stderr)
            return 2

        refresh_command(protocol=protocol, command=args.refresh)
        return 0

    if args.action is None:
        parser.print_usage(sys.stderr)
        return 2

    if args.action == "parse":
        result = parse_command_line(cmdline=args.cmdline)
        if result:
            print(
                json.dumps(
                    {
                        "command": result.command,
                        "subcommand": result.subcommand,
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
        desc = lookup_flag(
            protocol=protocol,
            command=args.command,
            flag=args.flag,
            subcommand=args.subcommand,
        )
        if desc:
            print(desc)
            return 0
        else:
            print("Unknown flag", file=sys.stderr)
            return 1

    elif args.action == "explain":
        explanations = explain_command(protocol=protocol, cmdline=args.cmdline)
        output = format_explanations(
            explanations=explanations, use_color=not args.no_color
        )
        if output:
            print(output)
        return 0

    elif args.action == "refresh":
        if args.all:
            refresh_all(protocol=protocol)
            return 0
        if not args.command:
            print("Missing command for refresh", file=sys.stderr)
            return 2
        refresh_command(protocol=protocol, command=args.command)
        return 0

    elif args.action == "scan":
        # Background scan with lock file to prevent duplicate scans
        lock_path = command_data_path(command=args.command).with_suffix(".scanning")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            lock_path.touch()
            ensure_command_scanned(
                protocol=protocol, command=args.command, refresh=False
            )
        finally:
            lock_path.unlink(missing_ok=True)
        return 0

    elif args.action == "generate":
        if args.agent != "claude":
            print(f"Unsupported agent: {args.agent}", file=sys.stderr)
            return 2

        agent = ClaudeCodeAgent(model=args.model)
        data = generate_command_data_llm(
            protocol=protocol,
            agent=agent,
            command=args.command,
            max_subcommands=max(0, args.max_subcommands),
            max_doc_chars=max(1, args.max_doc_chars),
            timeout_s=max(1, args.timeout_s),
        )
        save_command_data(command=args.command, data=data)
        if args.print:
            print(json.dumps(data, indent=2, sort_keys=True))
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
