# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

Shellock is a real-time CLI flag explainer for fish shell, written in Python 3.13+. It shows flag descriptions below the prompt as you type commands by combining Python-based parsing with fish shell integration.

## Architecture

**Two-Component System:**
- **Python Backend** (`shellock.py`): Core logic for parsing commands, extracting flags, retrieving descriptions from help sources, and storing durable command metadata
- **Fish Shell Frontend** (`shellock.fish`, `shellock_bindings.fish`): Terminal UI that hooks into fish key bindings, renders explanations using ANSI escape sequences, and handles cleanup

**Data Flow:**
1. User types in fish → Key binding triggers `__shellock_explain`
2. Fish sends command to Python script → `parse_command_line()` extracts flags
3. Python looks up flags → Non-blocking check for command metadata
   - If data exists: serves from JSON immediately
   - If missing: spawns background scan process, returns "Learning \<command\>..." message
4. Background scan (first use only) → `parse_help_output()` and `parse_man_page()` extract flags from help/man pages
5. Results returned to fish → Rendered below prompt with ANSI colors
6. Command metadata stored → `~/.config/fish/shellock/data/<command>.json` (atomic writes)
7. Lock files → `.scanning` files prevent duplicate background scans (5-minute timeout)

**Command-Specific Handling:**
The `HELP_SOURCE_OVERRIDES` dict provides special handling for complex commands like git, docker, kubectl that need subcommand-aware help invocation (e.g., `git help commit` vs `git -h`).

## Development Commands

**Running shellock directly:**
```bash
# From the shellock directory
./shellock.py explain "git commit -am 'message'"
./shellock.py lookup git -m --subcommand commit
./shellock.py parse "rsync -avz src/ dest/"
./shellock.py -r git
```

**Testing changes:**
- No unit tests exist (tests/ directory is empty)
- Test manually in fish shell after making changes
- Test edge cases: combined short flags (`-rf`), single-dash long flags (`-name`), subcommands, unusual man page formats
- The code uses `uv` (embedded in shebang) - no need for virtual environments or pip installs

**Key implementation notes:**
- Python 3.13+ required (uses modern typing features like `|` union syntax)
- Command metadata stored per command under `~/.config/fish/shellock/data/`
- Fish integration uses global `__shellock_lines` variable for cleanup tracking
- Pattern matching in fish functions uses `string match` for cross-platform compatibility
- **Non-blocking architecture**: First-time command scans happen in background processes (detached via `start_new_session=True`)
- **Smart truncation**: Descriptions truncated at sentence/word boundaries, not mid-word
- **Key bindings**: Handles Ctrl+U, Ctrl+K, Ctrl+W, Alt+D for proper hint cleanup on line edits
