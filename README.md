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
curl -fsSL https://raw.githubusercontent.com/ibehnam/shellock/main/install.fish | fish
```

### Clone and install

```fish
# Clone to your preferred location (e.g., ~/.local/share/shellock)
git clone https://github.com/ibehnam/shellock.git ~/.local/share/shellock
cd ~/.local/share/shellock
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
3. **Caching**: Caches results in `~/.cache/shellock/` for fast subsequent lookups
4. **Display**: Renders explanations below your prompt using ANSI escape codes

## CLI Tool

You can also use shellock directly from its directory:

```bash
# From the shellock directory
./shellock.py explain "git commit -am 'message'"

# Or use the full path to shellock.py
/path/to/shellock/shellock.py lookup git -m --subcommand commit

# Parse command structure (JSON output)
./shellock.py parse "rsync -avz src/ dest/"

# Clear cache
./shellock.py clear-cache

# For fish shell integration, make sure to run the installer
```

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

```
shellock/
├── shellock.py          # Main Python script
├── shellock.fish        # Fish shell functions
├── shellock_bindings.fish # Key bindings (symlinked to ~/.config/fish/conf.d/)
├── install.fish         # Installer script
└── README.md            # This file
```

## Requirements

- Python 3.13+ (for optimal performance and latest features)
- fish shell
- Standard Unix tools (man, col)

## License

MIT
