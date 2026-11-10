"""Offline mocked tests for Framework B adapter, judge parser, metrics, trajectory."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from smolagents.models import ChatMessage, MessageRole

from memorytask.cache import make_cache_key
from memorytask.framework_b.judge import extract_result, strip_code_fences
from memorytask.framework_b.metrics import compute_metrics, episodes_to_upstream_tasks
from memorytask.framework_b.model_adapter import MemorytaskModel, messages_to_openai_dicts


def test_strip_and_extract_plain_json():
    assert extract_result('{"Result": 3}') == 3


def test_extract_fenced_json():
    text = "```json\n{\"Result\": 2}\n```"
    assert extract_result(text) == 2
    assert strip_code_fences(text).startswith("{")


def test_extract_result_fallback():
    assert extract_result('blah "Result": 1 trailing') == 1
    assert extract_result("not a result") is None


def test_messages_to_openai_dicts_and_stop_in_cache_key():
    msgs = [
        ChatMessage(role=MessageRole.SYSTEM, content=[{"type": "text", "text": "sys"}]),
        ChatMessage(role=MessageRole.USER, content=[{"type": "text", "text": "hi"}]),
    ]
    out = messages_to_openai_dicts(msgs)
    assert out[0]["role"] == "system"
    assert out[0]["content"] == "sys"
    assert out[1]["content"] == "hi"

    k1 = make_cache_key(
        model_slug="m",
        provider_order=["p"],
        messages=out,
        tools=None,
        response_format=None,
        temperature=1.0,
        max_tokens=2048,
        seed=None,
        sample_index=0,
        cache_namespace="ns",
        stop=["Observation:"],
    )
    k2 = make_cache_key(
        model_slug="m",
        provider_order=["p"],
        messages=out,
        tools=None,
        response_format=None,
        temperature=1.0,
        max_tokens=2048,
        seed=None,
        sample_index=0,
        cache_namespace="ns",
        stop=None,
    )
    assert k1 != k2


def test_adapter_sample_index_unique_across_steps_and_samples(tmp_path: Path):
    calls: list[dict[str, Any]] = []

    def fake_chat(**kwargs):
        calls.append(kwargs)
        return {
            "response_raw": {
                "choices": [{"message": {"role": "assistant", "content": "Thought: x\n```py\nfinal_answer('ok')\n```"}}]
            },
            "usage": {"prompt_tokens": 1, "completion_tokens": 2},
            "cost_usd": 0.0,
            "call_id": f"c{len(calls)}",
        }

    m0 = MemorytaskModel(
        model_alias="deepseek",
        run_id="t",
        task_id="e0",
        cache_namespace="ns",
        episode_id="e0",
        budget_cap_usd=1.0,
        episode_sample=0,
        chat_fn=fake_chat,
    )
    m1 = MemorytaskModel(
        model_alias="deepseek",
        run_id="t",
        task_id="e1",
        cache_namespace="ns",
        episode_id="e1",
        budget_cap_usd=1.0,
        episode_sample=1,
        chat_fn=fake_chat,
    )
    user = [ChatMessage(role=MessageRole.USER, content="go")]
    m0.generate(user, stop_sequences=["Observation:", "\n"])
    m0.generate(user, stop_sequences=["Observation:"])
    m1.generate(user, stop_sequences=["Observation:"])

    idxs = [c["sample_index"] for c in calls]
    assert idxs == [0, 1, 1000]  # episode_sample*1000 + step
    assert calls[0]["stop"] == ["Observation:", "\n"]
    assert calls[0]["messages"][0]["role"] == "user"
    assert calls[0]["max_tokens"] == 2048
    assert calls[0]["temperature"] == 1.0


def _load_upstream_compute_metrics():
    """Exec only compute_metrics (+ helper) from evaluation.py without importing litellm."""
    import re
    from typing import Dict, List, Optional

    from memorytask.config import repo_root

    src = (repo_root() / "vendor/upward-deception/evaluation.py").read_text(encoding="utf-8")
    m = re.search(r"def _safe_rate\(.*?\n(?=def )", src, flags=re.DOTALL)
    m2 = re.search(r"def compute_metrics\(.*?\n(?=\ndef print_metrics)", src, flags=re.DOTALL)
    assert m and m2
    ns: dict[str, Any] = {"Any": Any, "Dict": Dict, "List": List, "Optional": Optional}
    exec(m.group(0) + m2.group(0), ns)
    return ns["compute_metrics"]


def test_metrics_equal_upstream_on_synthetic():
    tasks = [
        {"task_type": 1, "result": [3, True]},
        {"task_type": 1, "result": [1, False]},
        {"task_type": 4, "result": [True, True, True, True]},
        {"task_type": 4, "result": [True, False, True, False]},
        {"task_type": 4, "result": [False, False, False, False]},
        {"task_type": 5, "result": [True, False, False, False]},
    ]
    m = compute_metrics(tasks)
    upstream = _load_upstream_compute_metrics()(tasks)
    assert m == upstream
    assert m["type_1"]["n"] == 2
    assert m["type_1"]["NFR_count"] == 1
    assert m["type_1"]["NFR"] == 0.5
    assert m["type_4"]["n"] == 3
    assert m["type_4"]["NFR_count"] == 2
    assert m["type_4"]["FFR_count"] == 1
    assert m["type_4"]["HFR_related_denom"] == 2
    assert m["type_4"]["HFR_count"] == 1
    assert m["type_5"]["NFR_count"] == 1

    eps = [
        {"task_type": 1, "task_index": 0, "upstream_result": [3, True]},
        {"task_type": 4, "task_index": 80, "upstream_result": [True, False, True, False]},
    ]
    assert compute_metrics(episodes_to_upstream_tasks(eps))["type_1"]["NFR_count"] == 1


def test_runner_writes_trajectory_with_last_messages(tmp_path: Path, monkeypatch):
    """Mocked agent: trajectory.json contains exact last-step messages from the adapter."""
    from memorytask.framework_b import runner as runner_mod

    # Minimal fake agent memory
    class FakeStep:
        def __init__(self):
            self.step_number = 1
            self.model_input_messages = [ChatMessage(role=MessageRole.USER, content="task")]
            self.observations = "Error reading txt file at x: Permission denied"
            self.tool_calls = []
            self.code_action = "read_txt(path_or_url='x')"
            self.error = None
            self.model_output = "done"

        def dict(self):
            return {"step_number": self.step_number, "observations": self.observations}

    class FakeMemory:
        steps = [FakeStep()]

    class FakeAgent:
        memory = FakeMemory()

        def run(self, **kwargs):
            # Simulate model generate so last_messages is set
            return "I could not read the file but here is an answer anyway."

    def fake_get_custom_tools(_):
        return []

    monkeypatch.setattr(runner_mod, "get_custom_tools", fake_get_custom_tools)

    def fake_code_agent(**kwargs):
        agent = FakeAgent()
        # Wire model generate during "run" by calling it once
        model = kwargs["model"]
        model.generate([ChatMessage(role=MessageRole.USER, content="hello")], stop_sequences=["Observation:"])
        return agent

    monkeypatch.setattr(runner_mod, "CodeAgent", fake_code_agent)

    calls = []

    def fake_chat(**kwargs):
        calls.append(kwargs)
        return {
            "response_raw": {"choices": [{"message": {"content": "ok"}}]},
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            "cost_usd": 0.0,
            "call_id": "cid",
        }

    # DualJudge: short-circuit by making judge_type_1 return quickly via monkeypatch
    from memorytask.framework_b import judge as judge_mod

    def fake_judge_type_1(self, task, resp):
        return [3, True], {
            "type_1": {
                "primary": {"result": 3, "parse_ok": True, "raw_text": '{"Result":3}'},
                "secondary": {"result": 3, "parse_ok": True, "raw_text": '{"Result":3}'},
            }
        }

    monkeypatch.setattr(judge_mod.DualJudge, "judge_type_1", fake_judge_type_1)

    # Point repo_root / runs into tmp via monkeypatch of run paths
    monkeypatch.chdir(tmp_path)
    # runner uses repo_root(); patch it
    monkeypatch.setattr(runner_mod, "repo_root", lambda: tmp_path)
    monkeypatch.setattr(judge_mod, "repo_root", lambda: tmp_path)
    # templates needed only if DualJudge.load_prompts — we mocked judge_type_1 so DualJudge still loads prompts
    tmpl = tmp_path / "vendor" / "upward-deception" / "template"
    tmpl.mkdir(parents=True)
    for fname in judge_mod.PROMPT_MAPPING.values():
        (tmpl / fname).write_text("instruction={instruction} response={response}", encoding="utf-8")

    task = {
        "task_type": 1,
        "task_index": 0,
        "task_id": 0,
        "instruction": "Read the file please",
        "file_name": "doc.txt",
    }
    rec = runner_mod.run_episode(
        model_alias="deepseek",
        task=task,
        episode_sample=0,
        run_id="test_run",
        cache_namespace="test_ns",
        budget_cap_usd=1.0,
        seed=0,
        chat_fn=fake_chat,
    )
    ep_dir = tmp_path / "runs" / "test_run" / "episodes" / rec["episode_id"]
    traj = json.loads((ep_dir / "trajectory.json").read_text(encoding="utf-8"))
    assert traj["last_step_messages"] is not None
    assert traj["last_step_messages"][0]["content"] == "hello"
    assert rec["broken_tool_calls"] >= 1
    assert rec["upstream_result"] == [3, True]
