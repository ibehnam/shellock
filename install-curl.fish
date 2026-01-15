#!/usr/bin/env fish
# Shellock installer for fish shell (curl-pipe compatible) - installs to fish functions directory

set -l functions_dir "$HOME/.config/fish/functions"
set -l conf_dir "$HOME/.config/fish/conf.d"
echo "Installing Shellock to fish functions directory..."

# Create temporary directory for download
set -l tmp_dir (mktemp -d)
cd $tmp_dir

# Download all files
echo "Downloading Shellock..."
curl -fsSLO https://raw.githubusercontent.com/ibehnam/shellock/main/shellock.py
curl -fsSLO https://raw.githubusercontent.com/ibehnam/shellock/main/shellock.fish
curl -fsSLO https://raw.githubusercontent.com/ibehnam/shellock/main/shellock_bindings.fish

# Make shellock.py executable
chmod +x shellock.py

# Create directories if needed
mkdir -p "$functions_dir"
mkdir -p "$conf_dir"

# Copy files to functions directory
echo "Installing to $functions_dir..."
cp shellock.py "$functions_dir/"
cp shellock.fish "$functions_dir/"
cp shellock_bindings.fish "$functions_dir/"

# Create symlink in conf.d for auto-loading
set -l target "$conf_dir/shellock_bindings.fish"
if test -e "$target"
    echo "Removing existing $target"
    rm "$target"
end
ln -s "$functions_dir/shellock_bindings.fish" "$target"
echo "Created symlink: $target"

# Create Shellock data directory
mkdir -p "$HOME/.config/fish/shellock/data"
echo "Created data directory: ~/.config/fish/shellock/data"

# Clean up temp directory
rm -rf $tmp_dir

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