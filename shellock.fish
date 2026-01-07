# Shellock - Fish shell integration
# Real-time CLI flag explainer

# Track the number of lines we displayed (for cleanup)
set -g __shellock_lines 0

function __shellock_explain
    # Get current command line
    set -l cmdline (commandline -b)

    # Skip if empty or just whitespace
    if test -z (string trim "$cmdline")
        __shellock_clear
        return
    end

    # Skip if no flags (nothing starting with -)
    if not string match -q '*-*' "$cmdline"
        __shellock_clear
        return
    end

    # Get explanations from shellock
    set -l shellock_path "$HOME/Downloads/shellock/shellock.py"
    if not test -x "$shellock_path"
        return
    end

    # Run shellock and capture output
    set -l output ($shellock_path explain "$cmdline" 2>/dev/null)

    if test -z "$output"
        __shellock_clear
        return
    end

    # Clear previous output
    __shellock_clear

    # Count lines
    set -g __shellock_lines (count $output)

    # Save cursor position and move to line below prompt
    # Use terminal escape sequences
    printf '\e[s'  # Save cursor

    # Print each line of explanation
    for line in $output
        printf '\n%s' "$line"
    end

    printf '\e[u'  # Restore cursor
end

function __shellock_clear
    if test $__shellock_lines -gt 0
        # Save cursor, clear the lines we wrote, restore cursor
        printf '\e[s'
        for i in (seq $__shellock_lines)
            printf '\n\e[2K'  # Move down and clear line
        end
        printf '\e[u'
        set -g __shellock_lines 0
    end
end

function __shellock_on_dash
    # Insert the dash character
    commandline -i '-'
    # Then show explanations
    __shellock_explain
end

function __shellock_on_space
    # Insert space
    commandline -i ' '
    # Check if we should update explanations
    set -l cmdline (commandline -b)
    # Only update if there are flags
    if string match -q '*-*' "$cmdline"
        __shellock_explain
    end
end

function __shellock_on_enter
    # Clear explanations before executing
    __shellock_clear
    # Execute the command
    commandline -f execute
end

function __shellock_on_ctrl_c
    # Clear explanations on cancel
    __shellock_clear
    commandline -f cancel-commandline
end

function __shellock_manual
    # Manual trigger with Ctrl+H
    __shellock_explain
end

function __shellock_on_backspace
    # Perform normal backspace
    commandline -f backward-delete-char
    # Update explanations (will clear if no flags left)
    __shellock_explain
end

function __shellock_on_delete
    # Perform normal delete
    commandline -f delete-char
    # Update explanations (will clear if no flags left)
    __shellock_explain
end
