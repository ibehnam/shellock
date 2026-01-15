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
3. **Background Scanning** (first use only): Spawns detached process to parse `--help` output and man pages
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

## Data Storage

Shellock stores durable command metadata in:

```
~/.config/fish/shellock/data/
```

You can override this location with `SHELLOCK_HOME`.

## Supported Formats

Shellock parses various man page formats:

- **GNU style**: `-x, --long-option` with description on next line
- **BSD style**: `-x      Description on same line`
- **Multiple flags**: `-R, -r, --recursive`
- **Single-dash long flags**: `-name pattern` (find, java style)
- **Tree style**: `-L level` with description below

## Known Limitations

- Some commands with unusual man page formats may not parse correctly
- Commands that open interactive help (like `git commit --help` opening a pager) require man page fallback
- Combined short flags like `-rf` are split into individual flags

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