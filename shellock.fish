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
    set -l shellock_path (dirname (status filename))/shellock.py
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

function __shellock_on_scan_done --on-variable __shellock_scan_done
    # Payload format: "<command>:<epoch_ms>"
    set -l payload "$__shellock_scan_done"
    if test -z "$payload"
        return
    end

    set -l scanned_cmd (string split -m1 : -- "$payload")[1]
    if test -z "$scanned_cmd"
        return
    end

    set -l cmdline (commandline -b)
    set -l trimmed (string trim -- "$cmdline")
    if test -z "$trimmed"
        return
    end

    # Only refresh if the current command looks like it matches the scanned command.
    # (Prevents unrelated scans from triggering redraws.)
    if not string match -q '*-*' "$trimmed"
        return
    end

    set -l scanned_cmd_re (string escape --style=regex -- "$scanned_cmd")
    if string match -qr "^$scanned_cmd_re(\\s|\$)" "$trimmed"
        __shellock_explain
    end
end

function __shellock_on_dash
    # Insert the dash character
    commandline -i '-'
    # Then show explanations
    __shellock_explain
end

function __shellock_on_space
    # Expand abbreviations before inserting space (preserves `abbr` behavior)
    commandline -f expand-abbr
    # Insert a literal space
    commandline -i ' '
    # Check if we should update explanations
    set -l cmdline (commandline -b)
    # Only update if there are flags
    if string match -q '*-*' "$cmdline"
        __shellock_explain
    end
end

function __shellock_on_enter
    # Expand abbreviations before execution (preserves `abbr` behavior on Enter)
    commandline -f expand-abbr
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

function __shellock_on_kill_whole_line
    # Kill entire line (Ctrl+U)
    commandline -f kill-whole-line
    # Update explanations (will clear since no content remains)
    __shellock_explain
end

function __shellock_on_kill_line
    # Kill to end of line (Ctrl+K)
    commandline -f kill-line
    # Update explanations
    __shellock_explain
end

function __shellock_on_backward_kill_word
    # Kill word backward (Ctrl+W)
    commandline -f backward-kill-word
    # Update explanations
    __shellock_explain
end

function __shellock_on_kill_word
    # Kill word forward (Alt+D)
    commandline -f kill-word
    # Update explanations
    __shellock_explain
end
