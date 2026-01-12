#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.13"
# dependencies = []
# ///
"""
Shellock - Real-time CLI flag explainer.

Parses command-line flags and retrieves their descriptions from --help or man pages.
Designed to integrate with fish shell for real-time explanations as you type.

Usage:
    shellock parse "jq -r -s file.json"     # Parse flags from command
    shellock lookup jq -r                    # Look up description for -r flag of jq
    shellock explain "jq -r -s file.json"   # Parse and explain all flags
    shellock clear-cache                     # Clear the flag description cache
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final

# Constants
CACHE_DIR: Final[Path] = Path.home() / ".cache" / "shellock"
CACHE_TTL_DAYS: Final[int] = 30

# ANSI colors for terminal output
DIM: Final[str] = "\033[2m"
RESET: Final[str] = "\033[0m"
CYAN: Final[str] = "\033[36m"

# Command-specific help overrides
# Some commands need special help invocations to show all flags
HELP_SOURCE_OVERRIDES: Final[dict[str, list[str] | callable]] = {
    "curl": ["curl", "--help", "all"],
    "git": lambda subcmd: ["git", "help", subcmd] if subcmd else ["git", "-h"],
    "docker": lambda subcmd: ["docker", subcmd, "--help"] if subcmd else ["docker", "--help"],
    "kubectl": lambda subcmd: ["kubectl", subcmd, "--help"] if subcmd else ["kubectl", "--help"],
    "npm": lambda subcmd: ["npm", subcmd, "--help"] if subcmd else ["npm", "--help"],
}


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
    subcommand_commands = {"git", "docker", "kubectl", "npm", "cargo", "go", "pip", "uv"}

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


def get_cache_path(*, command: str, subcommand: str | None = None) -> Path:
    """Get the cache file path for a command."""
    key = f"{command}:{subcommand or ''}"
    cache_hash = hashlib.md5(key.encode()).hexdigest()[:12]
    return CACHE_DIR / f"{command}-{cache_hash}.json"


def load_cache(*, command: str, subcommand: str | None = None) -> dict[str, str] | None:
    """Load cached flag descriptions for a command."""
    cache_path = get_cache_path(command=command, subcommand=subcommand)

    if not cache_path.exists():
        return None

    try:
        data = json.loads(cache_path.read_text())
        return data.get("flags", {})
    except (json.JSONDecodeError, KeyError):
        return None


def save_cache(
    *,
    command: str,
    subcommand: str | None = None,
    flags: dict[str, str],
) -> None:
    """Save flag descriptions to cache."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = get_cache_path(command=command, subcommand=subcommand)

    data = {
        "command": command,
        "subcommand": subcommand,
        "flags": flags,
    }
    cache_path.write_text(json.dumps(data, indent=2))


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
        # Handle callable (for commands that need subcommand handling)
        if callable(help_source):
            cmd = help_source(subcommand)
        else:
            cmd = help_source.copy()

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
            env={**os.environ, "MANPAGER": "cat", "PAGER": "cat", "MAN_KEEP_FORMATTING": ""},
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
    *,
    command: str,
    flag: str,
    subcommand: str | None = None,
) -> str | None:
    """
    Look up the description for a specific flag.

    Checks cache first, then --help, then man page.
    If not found in subcommand context, falls back to parent command.
    """
    # Check cache
    cached = load_cache(command=command, subcommand=subcommand)
    if cached and flag in cached:
        return cached[flag]

    # Get and parse help
    all_flags: dict[str, str] = {}

    help_text = get_help_text(command=command, subcommand=subcommand)
    if help_text:
        all_flags.update(parse_help_output(help_text=help_text))

    # Try man page if flag not found in help
    if flag not in all_flags:
        man_text = get_man_text(command=command, subcommand=subcommand)
        if man_text:
            all_flags.update(parse_man_page(man_text=man_text))

    # Save to cache
    if all_flags:
        save_cache(command=command, subcommand=subcommand, flags=all_flags)

    # If flag not found and we were looking in a subcommand context,
    # fall back to the parent command (e.g., git -c is a git-level flag, not git-status)
    if flag not in all_flags and subcommand is not None:
        return lookup_flag(command=command, flag=flag, subcommand=None)

    # If flag not found, check if it's a combination of short flags
    # (e.g., -xf = -x + -f, -rf = -r + -f), but not --long-flags
    if flag not in all_flags and re.match(r"^-[a-zA-Z]{2,}$", flag):
        # Extract individual short flags (skip the leading dash)
        chars = flag[1:]
        descriptions: list[str] = []
        for char in chars:
            if char.isalpha():
                single_flag = f"-{char}"
                if single_flag in all_flags:
                    descriptions.append(f"{single_flag}: {all_flags[single_flag]}")

        if descriptions:
            # Return combined description
            return "Combination: " + " | ".join(descriptions)

    return all_flags.get(flag)


def explain_command(*, cmdline: str) -> list[FlagDescription]:
    """
    Parse a command line and return descriptions for all flags.
    """
    parsed = parse_command_line(cmdline=cmdline)
    if not parsed:
        return []

    results: list[FlagDescription] = []
    seen: set[str] = set()

    for flag in parsed.flags:
        if flag.name in seen:
            continue
        seen.add(flag.name)

        desc = lookup_flag(
            command=parsed.command,
            flag=flag.name,
            subcommand=parsed.subcommand,
        )
        results.append(
            FlagDescription(
                flag=flag.name,
                description=desc or "Unknown flag",
            )
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
            lines.append(f"{DIM}  {CYAN}{flag_padded}{RESET}  {DIM}{exp.description}{RESET}")
        else:
            lines.append(f"  {flag_padded}  {exp.description}")

    return "\n".join(lines)


def clear_cache() -> int:
    """Clear all cached flag descriptions."""
    if CACHE_DIR.exists():
        count = 0
        for f in CACHE_DIR.glob("*.json"):
            f.unlink()
            count += 1
        print(f"Cleared {count} cached entries")
    else:
        print("Cache directory does not exist")
    return 0


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Real-time CLI flag explainer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="action", required=True)

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

    # clear-cache command
    subparsers.add_parser("clear-cache", help="Clear the flag cache")

    args = parser.parse_args()

    if args.action == "parse":
        result = parse_command_line(cmdline=args.cmdline)
        if result:
            print(json.dumps({
                "command": result.command,
                "subcommand": result.subcommand,
                "flags": [{"name": f.name, "is_long": f.is_long} for f in result.flags],
            }))
        else:
            print("{}")
        return 0

    elif args.action == "lookup":
        desc = lookup_flag(
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
        explanations = explain_command(cmdline=args.cmdline)
        output = format_explanations(
            explanations=explanations,
            use_color=not args.no_color,
        )
        if output:
            print(output)
        return 0

    elif args.action == "clear-cache":
        return clear_cache()

    return 1


if __name__ == "__main__":
    sys.exit(main())
