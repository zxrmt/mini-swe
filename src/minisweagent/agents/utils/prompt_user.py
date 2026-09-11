from prompt_toolkit.formatted_text.html import HTML
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.shortcuts import PromptSession

from minisweagent import global_config_dir

_history = FileHistory(global_config_dir / "interactive_history.txt")
prompt_session = PromptSession(history=_history)

# Escape is only a submit prefix while it is still ambiguous: pause between Esc and Enter and
# prompt_toolkit flushes the lone Escape, so Enter would just add another line and nothing submits.
_multiline_bindings = KeyBindings()
_multiline_bindings.add("enter")(lambda event: event.current_buffer.validate_and_handle())
_multiline_bindings.add("c-j")(lambda event: event.current_buffer.insert_text("\n"))

_multiline_prompt_session = PromptSession(history=_history, multiline=True, key_bindings=_multiline_bindings)


def _multiline_prompt() -> str:
    return _multiline_prompt_session.prompt(
        "",
        bottom_toolbar=HTML(
            "Submit message: <b fg='yellow' bg='black'>Enter</b> | "
            "New line: <b fg='yellow' bg='black'>Ctrl+J</b> | "
            "Navigate history: <b fg='yellow' bg='black'>Arrow Up/Down</b> | "
            "Search history: <b fg='yellow' bg='black'>Ctrl+R</b>"
        ),
    )
