# Shellock

Real-time CLI flag explainer for fish shell. Shows flag descriptions as you type commands.

## Demo

```
$ tree -L 2 -a -C
  -L  Max display depth of the directory tree.
  -a  All files are printed...
  -C  Turn colorization on always...
```

## Installation

### One-liner install (recommended)

```fish
curl -fsSL https://raw.githubusercontent.com/ibehnam/shellock/main/install-curl.fish | fish
```

> **Note**: If you get a 404 error, wait a few seconds and try again. GitHub's CDN may take a moment to update.

### Clone and install

```fish
# Clone to your preferred location
git clone https://github.com/ibehnam/shellock.git ~/downloads/shellock
cd ~/downloads/shellock
fish install.fish
```

The installer creates:
- `~/.config/fish/shellock/config.json` (defaults; not overwritten if present)
- `~/.config/fish/shellock/data/` (durable per-command metadata)

You can override the base directory with `SHELLOCK_HOME`.

## Usage

After installation, explanations appear automatically as you type:

- Type any command with flags (e.g., `ls -la`)
- Flag explanations appear below the prompt
- Press `Enter` to execute (clears explanations)
- Press `Ctrl+C` to cancel (clears explanations)
- Press `Ctrl+H` for manual help on current command

**Line editing keys also update hints:**
- `Ctrl+U` - Kill entire line (clears hints)
- `Ctrl+K` - Kill to end of line
- `Ctrl+W` - Delete word backward
- `Alt+D` - Delete word forward
- `Backspace`/`Delete` - Remove characters

## How It Works

1. **Command Parsing**: Extracts command name and flags from your input
2. **Flag Lookup**: Non-blocking check for command metadata
   - If cached: instant display from `~/.config/fish/shellock/data/`
   - If new command: shows "Learning \<command\>..." and scans in background
3. **Background Scanning** (first use only): Uses LLM to extract flags and subcommands
   - Reads `command --help` and asks LLM to extract both flags and subcommand names
   - For each discovered subcommand, recursively scans `command subcommand --help` (in parallel)
   - Continues until max depth (default: 3 levels) or no subcommands found
   - Spawns detached process so fish remains responsive
4. **Data Store**: Saves per-command flag metadata with atomic writes for fast subsequent lookups
5. **Display**: Renders explanations below your prompt using ANSI escape codes
   - **Text Wrapping**: Long descriptions wrap at 80 chars on word boundaries, with continuation lines indented
   - **Smart Truncation**: Descriptions end at sentence/word boundaries (160 char max), not mid-word
6. **Proper Cleanup**: Multi-line output ensures fish clears all wrapped text when you delete flags

## CLI Tool

You can also use shellock directly:

```bash
# From the installation directory
~/.config/fish/functions/shellock.py explain "git commit -am 'message'"

# Or use the full path
~/.config/fish/functions/shellock.py lookup git -m --subcommand commit

# Parse command structure (JSON output)
~/.config/fish/functions/shellock.py parse "rsync -avz src/ dest/"

# Refresh a command
~/.config/fish/functions/shellock.py -r git

# Refresh everything
~/.config/fish/functions/shellock.py -r -a
```

## LLM-based Extraction

Shellock uses an LLM to extract flag descriptions and discover subcommands from help output.

**Requirements:**
- **`ccs`** or **`claude`** CLI must be installed and available on your PATH
  - Install Claude Code: https://claude.com/code
  - `ccs` is the Claude Code Shortcuts CLI (comes with Claude Code)
  - `claude` is the main Claude Code CLI

**Proxy Configuration:**

By default, Shellock uses a working proxy setup via `ccs glm` to leverage alternative LLM models. This is configured in `shellock.py`:

```python
@dataclass(frozen=True, slots=True)
class ClaudeCodeAgent:
    model: str = "sonnet"
    executable: str = "ccs glm"  # Using GLM proxy via ccs
```

**To use standard Claude Code (no proxy):**

Edit `shellock.py` line 260 and change:
```python
executable: str = "ccs glm"
```
to:
```python
executable: str = "claude"
```

**To use your own proxy:**

Change the `executable` to your proxy command (e.g., `"ccs gemini"`, `"my-llm-proxy"`). The proxy must:
- Accept `--output-format json` flag
- Return structured output in the Claude Code JSON event format
- Support `--json-schema` for structured extraction

**Configuration:**

Shellock reads `~/.config/fish/shellock/config.json` for configuration. Example:

```json
{
  "max_subcommand_depth": 3,
  "llm_model": "sonnet",
  "llm_max_subcommands": 25,
  "llm_max_doc_chars": 30000,
  "llm_timeout_s": 180
}
```

**Defaults:**
- `max_subcommand_depth`: `3` (how deep to recursively scan subcommands)
- `llm_model`: `sonnet` (can be `opus`, `sonnet`, or `haiku`)
- `llm_max_subcommands`: `25` (max subcommands to scan per level)
- `llm_max_doc_chars`: `30000` (truncate help text if longer)
- `llm_timeout_s`: `180` (timeout per LLM call)

**Manual generation:**

```bash
# Generate and cache JSON for a command (also prints it)
~/.config/fish/functions/shellock.py generate git --print

# Control depth and subcommand limits
~/.config/fish/functions/shellock.py generate docker --max-depth 2 --max-subcommands 10
```

## Data Storage

Shellock stores durable command metadata in:

```
~/.config/fish/shellock/data/
```

You can override this location with `SHELLOCK_HOME`.

## Known Limitations

- Requires `ccs` or `claude` CLI (Claude Code) for LLM-based extraction
- First-time command scans take time (runs in background)
- LLM extraction quality depends on help text formatting
- Combined short flags like `-rf` are split into individual flags during lookup
- Proxy configuration requires manual code edit in `shellock.py` (line 260)

## Files

Shellock installs to the standard fish functions directory:

```
~/.config/fish/
├── functions/
│   ├── shellock.py              # Main Python script
│   ├── shellock.fish            # Fish shell functions
│   └── shellock_bindings.fish   # Key bindings
├── conf.d/
│   └── shellock_bindings.fish   # Symlink for auto-loading
└── shellock/
    └── data/                    # Durable command metadata
```

## Requirements

- Python 3.13+ (for optimal performance and latest features)
- fish shell
- Standard Unix tools (man, col)

## License

MIT
