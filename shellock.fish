# Shellock - Fish shell integration
# Real-time CLI flag explainer

# Track the number of lines we displayed (for cleanup)
set -g __shellock_lines 0

# Ensure the data directory exists after install (even if empty).
function __shellock_ensure_data_dir
    set -l shellock_home (set -q SHELLOCK_HOME; and echo "$SHELLOCK_HOME"; or echo "$HOME/.config/fish/shellock")
    set -l shellock_data "$shellock_home/data"
    if not test -d "$shellock_data"
        mkdir -p "$shellock_data" >/dev/null 2>&1
    end
end
__shellock_ensure_data_dir
functions -e __shellock_ensure_data_dir

function __shellock_is_command
    set -l cmd "$argv[1]"
    if test -z "$cmd"
        return 1
    end

    # Paths like ./tool or /usr/local/bin/tool
    if string match -q -- '*/*' "$cmd"
        if test -x "$cmd"
            return 0
        end
        return 1
    end

    # External commands only; avoids triggering on random strings.
    command -sq -- "$cmd"
    return $status
end

# Return 0 if the command line contains a whitespace-delimited flag token.
function __shellock_has_flag
    set -l trimmed (string trim -- "$argv[1]")
    if test -z "$trimmed"
        return 1
    end

    set -l tokens (string split --no-empty ' ' -- "$trimmed")
    if test (count $tokens) -le 1
        return 1
    end

    set -l idx 1
    for token in $tokens
        if test $idx -eq 1
            set idx 2
            continue
        end
        if string match -q -- '-*' "$token"
            return 0
        end
    end

    return 1
end

function __shellock_should_explain
    set -l cmdline "$argv[1]"
    set -l trimmed (string trim -- "$cmdline")
    if test -z "$trimmed"
        return 1
    end

    set -l cmd (string split -m1 ' ' -- "$trimmed")[1]
    if not __shellock_is_command "$cmd"
        return 1
    end

    # Always explain when a flag token exists.
    if __shellock_has_flag "$trimmed"
        return 0
    end

    # Trigger on "command␠" (space right after a real command).
    set -l tokens (string split --no-empty ' ' -- "$trimmed")
    if test (count $tokens) -eq 1; and string match -qr -- '[[:space:]]$' -- "$cmdline"
        return 0
    end

    return 1
end

# Cleanup legacy signal handler name (older Shellock versions).
if functions -q __shellock_on_winch
    functions -e __shellock_on_winch
end

function __shellock_explain
    # Get current command line (or use the provided snapshot)
    set -l cmdline ""
    if set -q argv[1]
        set cmdline "$argv[1]"
    else
        set cmdline (commandline -b)
        if test $status -ne 0
            return
        end
    end

    # Skip if empty or just whitespace
    if test -z (string trim "$cmdline")
        __shellock_clear
        return
    end

    if not __shellock_should_explain "$cmdline"
        __shellock_clear
        return
    end

    # Get explanations from shellock
    set -l shellock_path (dirname (status filename))/shellock.py
    if not test -x "$shellock_path"
        return
    end

    # Run shellock and capture output
    set -l output (env SHELLOCK_FISH_PID=$fish_pid "$shellock_path" explain "$cmdline" 2>/dev/null)

    if test -z "$output"
        __shellock_clear
        return
    end

    # Clear previous output before we update learning state for this render.
    __shellock_clear

    # Track "learning" state so we can refresh on scan completion without a keystroke.
    if test (count $output) -eq 1; and string match -q '*Learning *' -- "$output[1]"
        set -g __shellock_learning_cmdline "$cmdline"
        set -l trimmed (string trim -- "$cmdline")
        set -g __shellock_learning_cmd (string split -m1 ' ' -- "$trimmed")[1]
    else
        set -e __shellock_learning_cmd
        set -e __shellock_learning_cmdline
    end

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
    set -e __shellock_learning_cmd
    set -e __shellock_learning_cmdline
end

function __shellock_on_shellock_signal --on-signal USR1
    if not set -q __shellock_learning_cmdline
        return
    end
    if test -z "$__shellock_learning_cmdline"
        return
    end

    # If commandline is available, ensure it still matches the learning snapshot.
    set -l current_cmdline (commandline -b)
    set -l current_status $status
    if test $current_status -eq 0
        set -l current_trimmed (string trim -- "$current_cmdline")
        if test -z "$current_trimmed"
            return
        end
        set -l learning_trimmed (string trim -- "$__shellock_learning_cmdline")
        if test "$current_trimmed" != "$learning_trimmed"
            return
        end
    end

    __shellock_explain "$__shellock_learning_cmdline"
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

    set -l cmdline ""
    if set -q __shellock_learning_cmdline
        set cmdline "$__shellock_learning_cmdline"
    else
        set cmdline (commandline -b)
        if test $status -ne 0
            return
        end
    end

    set -l trimmed (string trim -- "$cmdline")
    if test -z "$trimmed"
        return
    end

    # Only refresh if the current command looks like it matches the scanned command.
    # (Prevents unrelated scans from triggering redraws.)
    if not __shellock_should_explain "$cmdline"
        return
    end

    set -l scanned_cmd_re (string escape --style=regex -- "$scanned_cmd")
    if string match -qr "^$scanned_cmd_re(\\s|\$)" "$trimmed"
        __shellock_explain "$cmdline"
    end
end

function __shellock_on_dash
    # Insert the dash character
    commandline -i '-'
    # Only trigger when starting a flag token (not when typing a dashed command name).
    set -l cmdline (commandline -b)
    set -l trimmed (string trim -- "$cmdline")
    set -l tokens (string split --no-empty ' ' -- "$trimmed")
    if test (count $tokens) -ge 2; and string match -q -- '-*' "$tokens[-1]"
        __shellock_explain
    end
end

function __shellock_on_space
    # Expand abbreviations before inserting space (preserves `abbr` behavior)
    commandline -f expand-abbr
    # Insert a literal space
    commandline -i ' '
    set -l cmdline (commandline -b)
    if __shellock_should_explain "$cmdline"
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

# CLI entrypoint wrapper so `shellock ...` works as a command.
function shellock
    set -l shellock_path (dirname (status filename))/shellock.py
    if not test -x "$shellock_path"
        echo "shellock: missing executable at $shellock_path" >&2
        return 127
    end
    command "$shellock_path" $argv
end
