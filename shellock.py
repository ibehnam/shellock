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
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Final, Protocol

# Constants
PROTOCOL_VERSION: Final[int] = 1

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
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def ensure_command_scanned(
    *, protocol: CommandProtocol, command: str, refresh: bool
) -> dict:
    data = load_command_data(command=command)

    if data is None or refresh:
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
                mode_flags[flag] = desc

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
                    desc = desc.strip()
                    if first:
                        flags[first] = desc
                    if second:
                        flags[second] = desc
                elif len(groups) == 2:
                    flag, desc = groups
                    flags[flag] = desc.strip()
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
                    desc = " ".join(current_desc).strip()
                    if len(desc) > 160:
                        desc = desc[:157] + "..."
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
        desc = " ".join(current_desc).strip()
        if len(desc) > 160:
            desc = desc[:157] + "..."
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
    """Parse a command line and return descriptions for all flags."""
    parsed = parse_command_line(cmdline=cmdline)
    if not parsed:
        return []

    # Ensure command DB exists; on first encounter this eagerly scans subcommands too.
    ensure_command_scanned(protocol=protocol, command=parsed.command, refresh=False)

    results: list[FlagDescription] = []
    seen: set[str] = set()

    for flag in parsed.flags:
        if flag.name in seen:
            continue
        seen.add(flag.name)

        desc = lookup_flag(
            protocol=protocol,
            command=parsed.command,
            flag=flag.name,
            subcommand=parsed.subcommand,
        )
        results.append(
            FlagDescription(flag=flag.name, description=desc or "Unknown flag")
        )

    return results


def format_explanations(
    *,
    explanations: list[FlagDescription],
    use_color: bool = True,
) -> str:
    """Format flag explanations for terminal display."""
    if not explanations:
        return ""

    lines: list[str] = []
    max_flag_len = max(len(e.flag) for e in explanations)

    for exp in explanations:
        flag_padded = exp.flag.ljust(max_flag_len)
        if use_color:
            lines.append(
                f"{DIM}  {CYAN}{flag_padded}{RESET}  {DIM}{exp.description}{RESET}"
            )
        else:
            lines.append(f"  {flag_padded}  {exp.description}")

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

    return 1


if __name__ == "__main__":
    sys.exit(main())
