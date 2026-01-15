#!/usr/bin/env fish
# Shellock uninstaller for fish shell - removes functions, bindings, and data

echo "Uninstalling Shellock from fish config..."

set -l functions_dir "$HOME/.config/fish/functions"
set -l conf_dir "$HOME/.config/fish/conf.d"
set -l data_dir "$HOME/.config/fish/shellock"

# Remove symlink in conf.d
set -l target "$conf_dir/shellock_bindings.fish"
if test -L "$target"
    echo "Removing symlink: $target"
    rm "$target"
else if test -e "$target"
    echo "Removing file: $target"
    rm "$target"
end

# Remove installed functions
set -l files_to_remove \
    "$functions_dir/shellock.py" \
    "$functions_dir/shellock.fish" \
    "$functions_dir/shellock_bindings.fish"

for f in $files_to_remove
    if test -e "$f"
        echo "Removing: $f"
        rm "$f"
    end
end

# Remove data directory
if test -d "$data_dir"
    echo "Removing data: $data_dir"
    rm -rf "$data_dir"
end

echo ""
echo "Uninstall complete!"
echo ""
echo "If shellock is currently active, restart fish or run:"
echo "  functions -e __shellock_explain __shellock_cleanup __shellock_render"
