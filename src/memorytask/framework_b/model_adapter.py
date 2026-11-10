"""smolagents Model adapter that routes every generate() through memorytask.client.chat."""

from __future__ import annotations

from typing import Any, Callable

from smolagents.models import ChatMessage, MessageRole, Model, get_clean_message_list, tool_role_conversions
from smolagents.monitoring import TokenUsage

from memorytask.client import chat as default_chat

ChatFn = Callable[..., dict[str, Any]]

# Subject step budget (brief §3); override configs/models.yaml max_tokens=256.
SUBJECT_MAX_TOKENS = 2048
SUBJECT_TEMPERATURE = 1.0


def _assistant_text(response_raw: dict[str, Any]) -> str:
    choices = response_raw.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    content = message.get("content")
    if content is None:
        return ""
    if isinstance(content, list):
        parts: list[str] = []
        for p in content:
            if isinstance(p, dict) and p.get("type") == "text":
                parts.append(p.get("text") or "")
            elif isinstance(p, str):
                parts.append(p)
        return "".join(parts)
    return str(content)


def _apply_stop(content: str, stop_sequences: list[str] | None) -> str:
    if not content or not stop_sequences:
        return content
    cut = len(content)
    for seq in stop_sequences:
        if not seq:
            continue
        idx = content.find(seq)
        if idx != -1:
            cut = min(cut, idx)
    return content[:cut]


def _normalize_message(message: ChatMessage | dict) -> ChatMessage:
    if isinstance(message, dict):
        message = ChatMessage.from_dict(message)
    content = message.content
    if isinstance(content, str):
        message.content = [{"type": "text", "text": content}]
    elif content is None:
        message.content = [{"type": "text", "text": ""}]
    return message


def messages_to_openai_dicts(messages: list[ChatMessage | dict]) -> list[dict[str, Any]]:
    """Convert smolagents ChatMessages to OpenAI-style role/content dicts for client.chat."""
    normalized = [_normalize_message(m) for m in messages]
    cleaned = get_clean_message_list(
        normalized,
        role_conversions=tool_role_conversions,
        convert_images_to_image_urls=True,
        flatten_messages_as_text=True,
    )
    out: list[dict[str, Any]] = []
    for m in cleaned:
        role = m["role"]
        if hasattr(role, "value"):
            role = role.value
        content = m.get("content")
        out.append({"role": str(role), "content": content if content is not None else ""})
    return out


class MemorytaskModel(Model):
    """smolagents Model whose generate() calls memorytask.client.chat."""

    def __init__(
        self,
        *,
        model_alias: str,
        run_id: str,
        task_id: str,
        cache_namespace: str,
        episode_id: str,
        budget_cap_usd: float,
        episode_sample: int = 0,
        chat_fn: ChatFn = default_chat,
        temperature: float = SUBJECT_TEMPERATURE,
        max_tokens: int = SUBJECT_MAX_TOKENS,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_id=model_alias, flatten_messages_as_text=True, **kwargs)
        self.model_alias = model_alias
        self.run_id = run_id
        self.task_id = task_id
        self.cache_namespace = cache_namespace
        self.episode_id = episode_id
        self.budget_cap_usd = budget_cap_usd
        self.episode_sample = int(episode_sample)
        self.chat_fn = chat_fn
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.step_counter = 0
        self.call_ids: list[str] = []
        self.last_messages: list[dict[str, Any]] | None = None

    def next_sample_index(self) -> int:
        """Unique per call within an episode: episode_sample*1000 + step_counter."""
        idx = self.episode_sample * 1000 + self.step_counter
        self.step_counter += 1
        return idx

    def generate(
        self,
        messages: list[ChatMessage | dict],
        stop_sequences: list[str] | None = None,
        response_format: dict[str, str] | None = None,
        tools_to_call_from: list | None = None,
        **kwargs: Any,
    ) -> ChatMessage:
        openai_messages = messages_to_openai_dicts(messages)
        self.last_messages = openai_messages
        sample_index = self.next_sample_index()

        result = self.chat_fn(
            model_alias=self.model_alias,
            messages=openai_messages,
            tools=None,  # CodeAgent uses code blocks, not native tool calling
            response_format=response_format,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stop=list(stop_sequences) if stop_sequences else None,
            cache_namespace=self.cache_namespace,
            run_id=self.run_id,
            task_id=self.task_id,
            role="subject",
            sample_index=sample_index,
            budget_cap_usd=self.budget_cap_usd,
        )
        call_id = result.get("call_id")
        if call_id:
            self.call_ids.append(call_id)

        content = _assistant_text(result.get("response_raw") or {})
        content = _apply_stop(content, stop_sequences)

        usage = result.get("usage") or {}
        token_usage = TokenUsage(
            input_tokens=int(usage.get("prompt_tokens") or 0),
            output_tokens=int(usage.get("completion_tokens") or 0),
        )
        return ChatMessage(
            role=MessageRole.ASSISTANT,
            content=content,
            raw=result.get("response_raw"),
            token_usage=token_usage,
        )
