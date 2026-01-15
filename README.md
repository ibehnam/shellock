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

## How It Works

1. **Command Parsing**: Extracts command name and flags from your input
2. **Flag Lookup**: Searches `--help` output and man pages for descriptions
3. **Data Store**: Saves per-command flag metadata in `~/.config/fish/shellock/data/` for fast subsequent lookups
4. **Display**: Renders explanations below your prompt using ANSI escape codes

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