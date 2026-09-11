"""Helper function for pretty-printing content strings."""

import json

_REASONING_BLOCK_TYPES = ("thinking", "reasoning", "reasoning_text", "reasoning.text")
_REASONING_LIST_FIELDS = ("thinking_blocks", "reasoning_items", "reasoning_details")


def _format_tool_call(args_str: str) -> str:
    """Format tool call arguments, extracting command if it's a bash call."""
    try:
        args = json.loads(args_str) if isinstance(args_str, str) else args_str
        if isinstance(args, dict) and "command" in args:
            return f"```\n{args['command']}\n```"
    except Exception:
        pass
    return f"```\n{args_str}\n```"


def _format_observation(content: str, quiet: bool = False) -> str | None:
    """Try to format an observation JSON as key-value pairs (or, if quiet, as the bare output)."""
    try:
        data = json.loads(content)
        if isinstance(data, dict) and "returncode" in data:
            if quiet:
                return "\n".join(str(value) for key, value in data.items() if key != "returncode")
            lines = []
            for key, value in data.items():
                lines.append(f"<{key}>")
                lines.append(str(value))
            return "\n".join(lines)
        return content
    except Exception:
        return content


def _reasoning_texts_from_item(item: dict) -> list[str]:
    """Extract human-readable reasoning from a content block or reasoning item."""
    if not isinstance(item, dict) or item.get("type") == "redacted_thinking":
        return []
    texts = []
    if thinking := item.get("thinking"):
        texts.append(thinking)
    if summary := item.get("summary"):
        if isinstance(summary, str):
            texts.append(summary)
        elif isinstance(summary, list):
            for part in summary:
                if isinstance(part, str) and part:
                    texts.append(part)
                elif isinstance(part, dict) and (text := part.get("text")):
                    texts.append(text)
    for key in ("text", "reasoning"):
        if isinstance(value := item.get(key), str) and value:
            texts.append(value)
    if isinstance(content := item.get("content"), str):
        texts.append(content)
    elif isinstance(content, list):
        for part in content:
            if isinstance(part, str) and part:
                texts.append(part)
            elif isinstance(part, dict):
                texts.extend(_reasoning_texts_from_item(part))
    return texts


def _extract_reasoning_texts(message: dict) -> list[str]:
    """Collect reasoning/thinking text from all supported message formats."""
    texts = []
    if isinstance(reasoning := message.get("reasoning_content"), str):
        texts.append(reasoning)
    for field in ("reasoning", "thinking"):
        if isinstance(value := message.get(field), str):
            texts.append(value)
    for field in _REASONING_LIST_FIELDS:
        if isinstance(items := message.get(field), list):
            for item in items:
                if isinstance(item, str):
                    texts.append(item)
                elif isinstance(item, dict):
                    texts.extend(_reasoning_texts_from_item(item))
    if isinstance(content := message.get("content"), list):
        for item in content:
            if isinstance(item, dict) and item.get("type") in _REASONING_BLOCK_TYPES:
                texts.extend(_reasoning_texts_from_item(item))
    if isinstance(output := message.get("output"), list):
        for item in output:
            if isinstance(item, dict) and item.get("type") in _REASONING_BLOCK_TYPES:
                texts.extend(_reasoning_texts_from_item(item))
    deduped = list(dict.fromkeys(text for text in texts if text))
    # Some providers expose both a concatenated reasoning string and the individual
    # blocks it was built from. Keep the concatenated form instead of printing both.
    if len(deduped) > 1 and "".join(deduped[1:]) == deduped[0]:
        return deduped[:1]
    return deduped


def get_reasoning_string(message: dict) -> str:
    """Extract only the reasoning/thinking text from any supported message format."""
    return "\n\n".join(_extract_reasoning_texts(message))


def get_content_string(
    message: dict, *, quiet: bool = False, skip_tool_calls: bool = False, include_reasoning: bool = True
) -> str:
    """Extract text content from any message format for display.
    Should support both OpenAI and Anthropic message formats.

    Set ``quiet`` to print observations without the returncode/key wrappers.
    Set ``skip_tool_calls`` to get only the reasoning text, without the commands.
    Set ``include_reasoning`` to control whether thinking/reasoning tokens are included.

    Handles:
    - Traditional chat: {"content": "text"}
    - Multimodal chat: {"content": [{"type": "text", "text": "..."}]}
    - Anthropic tool use: {"content": [{"type": "tool_use", "input": {...}}]}
    - Anthropic tool result: {"content": [{"type": "tool_result", "content": "..."}]}
    - Anthropic thinking: {"content": [{"type": "thinking", "thinking": "..."}]}
    - Reasoning fields: {"reasoning_content": "..."} and {"thinking_blocks": [...]}
    - Observation messages: {"content": "{\"returncode\": 0, \"output\": \"...\"}"}
    - Traditional tool calls: {"tool_calls": [{"function": {"name": "...", "arguments": "..."}}]}
    - Responses API: {"output": [{"type": "message", "content": [...]}]}
    """
    texts = _extract_reasoning_texts(message) if include_reasoning else []

    # Extract content (string or multimodal list)
    content = message.get("content")
    if isinstance(content, str):
        texts.append(_format_observation(content, quiet))
    elif isinstance(content, list):
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") in (*_REASONING_BLOCK_TYPES, "redacted_thinking"):
                continue
            if item.get("type") == "tool_use":
                if not skip_tool_calls:
                    texts.append(_format_tool_call(json.dumps(item.get("input", {}))))
            elif item.get("type") == "tool_result":
                rc = item.get("content", "")
                if isinstance(rc, str):
                    texts.append(_format_observation(rc, quiet))
            elif text := item.get("text"):
                texts.append(text)

    # Handle traditional tool_calls format (OpenAI/LiteLLM style)
    if (tool_calls := message.get("tool_calls")) and not skip_tool_calls:
        for tc in tool_calls:
            func = tc.get("function", {}) if isinstance(tc, dict) else getattr(tc, "function", None)
            if func:
                args = func.get("arguments", "{}") if isinstance(func, dict) else getattr(func, "arguments", "{}")
                texts.append(_format_tool_call(args))

    # Handle Responses API format (output array)
    if output := message.get("output"):
        if isinstance(output, str):
            texts.append(_format_observation(output))
        elif isinstance(output, list):
            for item in output:
                if not isinstance(item, dict):
                    continue
                if item.get("type") == "message":
                    for c in item.get("content", []):
                        if isinstance(c, dict) and (text := c.get("text")):
                            texts.append(text)
                elif item.get("type") == "function_call" and not skip_tool_calls:
                    texts.append(_format_tool_call(item.get("arguments", "{}")))

    return "\n\n".join(t for t in texts if t)
