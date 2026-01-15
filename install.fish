#!/usr/bin/env fish
# Shellock installer for fish shell - installs to fish functions directory

echo "Installing Shellock to fish functions directory..."

# Define source and destination
set -l source_dir (dirname (status filename))
set -l functions_dir "$HOME/.config/fish/functions"
set -l conf_dir "$HOME/.config/fish/conf.d"

# Create directories if needed
mkdir -p "$functions_dir"
mkdir -p "$conf_dir"

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
mkdir -p "$HOME/.config/fish/shellock/data"
echo "Created data directory: ~/.config/fish/shellock/data"

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