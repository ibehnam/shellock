#!/usr/bin/env fish
# Shellock installer for fish shell - installs to fish functions directory

echo "Installing Shellock to fish functions directory..."

# Define source and destination
set -l source_dir (dirname (realpath (status filename)))
set -l functions_dir "$HOME/.config/fish/functions"
set -l conf_dir "$HOME/.config/fish/conf.d"
set -l shellock_home (set -q SHELLOCK_HOME; and echo "$SHELLOCK_HOME"; or echo "$HOME/.config/fish/shellock")
set -l shellock_config "$shellock_home/config.json"
set -l shellock_data "$shellock_home/data"

# Create directories if needed
mkdir -p "$functions_dir"
mkdir -p "$conf_dir"
mkdir -p "$shellock_data"

# Copy files to functions directory
echo "Copying files to $functions_dir..."
cp "$source_dir/shellock.py" "$functions_dir/"
cp "$source_dir/shellock.fish" "$functions_dir/"
cp "$source_dir/shellock_bindings.fish" "$functions_dir/"

# Make shellock.py executable
chmod +x "$functions_dir/shellock.py"

# Create symlink in conf.d for auto-loading
set -l target "$conf_dir/shellock_bindings.fish"
if test -e "$target"
    echo "Removing existing $target"
    rm "$target"
end
ln -s "$functions_dir/shellock_bindings.fish" "$target"
echo "Created symlink: $target"

# Create Shellock data directory
echo "Created data directory: $shellock_data"

# Create default config if missing (do not overwrite existing config)
if not test -e "$shellock_config"
    mkdir -p "$shellock_home"
    set -l tmp_cfg "$shellock_config.tmp.$fish_pid"
    printf '%s\n' \
        '{' \
        '  "scan_backend": "llm",' \
        '  "doc_order": ["help", "man"],' \
        '  "llm_model": "sonnet",' \
        '  "llm_max_subcommands": 25,' \
        '  "llm_max_doc_chars": 30000,' \
        '  "llm_timeout_s": 180' \
        '}' \
        > "$tmp_cfg"
    mv "$tmp_cfg" "$shellock_config"
    echo "Created default config: $shellock_config"
end

echo ""
echo "Installation complete!"
echo ""
echo "To activate now, run:"
echo "  source $target"
echo ""
echo "Or start a new fish shell."
echo ""
echo "Usage:"
echo "  - Type a command with flags (e.g., 'ls -la')"
echo "  - Explanations appear below as you type"
echo "  - Press Ctrl+H for manual help on current command"
