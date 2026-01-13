# Shellock key bindings for fish shell
# Place in ~/.config/fish/conf.d/ or source from config.fish

# Load the main shellock functions
set -l shellock_dir (dirname (realpath (status filename)))
source "$shellock_dir/shellock.fish"

# Bind dash key to trigger explanation
# Works in both insert and default modes
bind -M insert '-' __shellock_on_dash
bind -M default '-' __shellock_on_dash

# Bind space to update after typing flags
bind -M insert ' ' __shellock_on_space
bind -M default ' ' __shellock_on_space

# Clear on Enter before execution
bind -M insert \r __shellock_on_enter
bind -M insert \n __shellock_on_enter
bind -M default \r __shellock_on_enter
bind -M default \n __shellock_on_enter

# Clear on Ctrl+C
bind -M insert \cc __shellock_on_ctrl_c
bind -M default \cc __shellock_on_ctrl_c

# Manual trigger with Ctrl+H (for "help")
bind -M insert \ch __shellock_manual
bind -M default \ch __shellock_manual

# Update on backspace (when deleting characters)
bind -M insert \x7f __shellock_on_backspace
bind -M default \x7f __shellock_on_backspace
bind -M insert \b __shellock_on_backspace
bind -M default \b __shellock_on_backspace

# Update on delete key
bind -M insert \e\[3~ __shellock_on_delete
bind -M default \e\[3~ __shellock_on_delete
