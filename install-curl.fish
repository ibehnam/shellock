#!/usr/bin/env fish
# Shellock installer for fish shell (curl-pipe compatible)

set -l config_dir "$HOME/.config/fish/conf.d"
echo "Installing Shellock..."

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

# Create config directory if needed
mkdir -p "$config_dir"

# Create symlink to bindings
set -l target "$config_dir/shellock_bindings.fish"
if test -e "$target"
    echo "Removing existing $target"
    rm "$target"
end

ln -s "$tmp_dir/shellock_bindings.fish" "$target"
echo "Created symlink: $target"

# Create cache directory
mkdir -p "$HOME/.cache/shellock"
echo "Created cache directory: ~/.cache/shellock"

echo ""
echo "Installation complete!"
echo ""
echo "To activate now, run:"
echo "  source $target"
echo ""
echo "Or restart your fish shell."
echo ""
echo "Usage:"
echo "  - Type a command with flags (e.g., 'ls -la')"
echo "  - Explanations appear below as you type"
echo "  - Press Ctrl+H for manual help on current command"
