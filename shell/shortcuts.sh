# Shell shortcuts for transcription-local-tool (bash and zsh).
# Load from ~/.zshrc or ~/.bash_profile:
#   . /path/to/transcription-local-tool/shell/shortcuts.sh
#
#   wholelink                                 -> full command ready to edit
#   wholelink <link> <class> <session> [...]  -> transcribe an online video
#   livetranscribe                            -> full command ready to edit
#   livetranscribe <class> <session> [...]    -> live mic, continuous 60s chunks

if [ -n "$ZSH_VERSION" ]; then
  eval '_tx_src="${(%):-%x}"'
else
  _tx_src="${BASH_SOURCE[0]}"
fi
_TX_DIR="$(cd "$(dirname "$_tx_src")/.." && pwd)"
unset _tx_src

_tx_offer() {
  printf '%s' "$1" | pbcopy 2>/dev/null
  if [ -n "$ZSH_VERSION" ]; then
    print -z -- "$1"
  else
    echo "$1"
    echo "(copied - paste, edit, Enter)"
  fi
}

wholelink() {
  if [ $# -eq 0 ]; then
    _tx_offer "\"$_TX_DIR/.venv/bin/python\" \"$_TX_DIR/transcribe_video.py\" \"LINK\" --class CLASS --session SESSION"
  elif [ $# -lt 3 ]; then
    echo "usage: wholelink <link> <class> <session> [extra flags...]"
    echo "   or: wholelink        (template to edit)"
    return 1
  else
    local link="$1" class="$2" session="$3"
    shift 3
    "$_TX_DIR/.venv/bin/python" "$_TX_DIR/transcribe_video.py" "$link" --class "$class" --session "$session" "$@"
  fi
}

livetranscribe() {
  if [ $# -eq 0 ]; then
    _tx_offer "\"$_TX_DIR/.venv/bin/python\" \"$_TX_DIR/transcribe_mic.py\" --class CLASS --session SESSION --continuous --seconds 60"
  elif [ $# -lt 2 ]; then
    echo "usage: livetranscribe <class> <session> [extra flags...]"
    echo "   or: livetranscribe   (template to edit)"
    return 1
  else
    local class="$1" session="$2"
    shift 2
    "$_TX_DIR/.venv/bin/python" "$_TX_DIR/transcribe_mic.py" --class "$class" --session "$session" --continuous --seconds 60 "$@"
  fi
}
