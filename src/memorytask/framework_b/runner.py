"""Framework B episode runner — faithful to upstream agent_runner + evaluation.run_one_task."""

from __future__ import annotations

import io
import json
import os
import random
import re
import shutil
import string
import sys
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any, Callable, Iterator

from smolagents import CodeAgent

from memorytask.config import repo_root
from memorytask.framework_b.judge import DualJudge, load_prompts
from memorytask.framework_b.model_adapter import MemorytaskModel, SUBJECT_MAX_TOKENS, SUBJECT_TEMPERATURE

ANSI_ESCAPE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")

# Markers drawn from vendor/upward-deception/utils/incorrect_tools.py failure returns.
BROKEN_TOOL_MARKERS = (
    "Error:",
    "Error reading",
    "Error during Whisper",
    "An error occurred during the analysis",
    "An unexpected error occurred while handling",
    "No available channels",
    "API service is currently unavailable",
    "Permission denied",
    "No module named",
    "The model returned an empty response",
)

MAX_COPIED_FILE_BYTES = 200 * 1024
VENDOR_ROOT = repo_root() / "vendor" / "upward-deception"


def _ensure_vendor_on_path() -> None:
    root = str(VENDOR_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def get_custom_tools(correct_tools_str: list[str]):
    """Upstream models.agent_runner.get_custom_tools (import from vendor)."""
    _ensure_vendor_on_path()
    # Upstream get_correct_tools eagerly constructs WebSearchTool even when the requested
    # list is empty. Its __init__ reads GOOGLE_API / GOOGLE_ID (error text wrongly names
    # GOOGLE_API_KEY / GOOGLE_CSE_ID). Dummy values suffice: correct_tools_str is [].
    os.environ.setdefault("GOOGLE_API", "unused-for-empty-correct-tools")
    os.environ.setdefault("GOOGLE_ID", "unused-for-empty-correct-tools")
    os.environ.setdefault("GOOGLE_API_KEY", "unused-for-empty-correct-tools")
    os.environ.setdefault("GOOGLE_CSE_ID", "unused-for-empty-correct-tools")
    from models.agent_runner import get_custom_tools as _upstream

    return _upstream(correct_tools_str)


def generate_random_content(length: int = 50) -> str:
    chars = string.ascii_letters + string.digits + " \n\t"
    return "".join(random.choice(chars) for _ in range(length))


def create_dummy_pptx(file_path: str, num_slides: int = 2) -> None:
    """Upstream evaluation.create_dummy_pptx (python-pptx)."""
    from pptx import Presentation
    from pptx.util import Inches, Pt

    prs = Presentation()
    blank_slide_layout = prs.slide_layouts[6]
    for i in range(num_slides):
        slide = prs.slides.add_slide(blank_slide_layout)
        textbox = slide.shapes.add_textbox(Inches(1.0), Inches(1.5), Inches(7.5), Inches(4.0))
        text_frame = textbox.text_frame
        text_frame.word_wrap = True
        p = text_frame.paragraphs[0]
        p.text = f"slide {i + 1} title"
        p.font.bold = True
        p.font.size = Pt(24)
        p = text_frame.add_paragraph()
        p.text = generate_random_content()
        p.font.size = Pt(14)
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    prs.save(file_path)


@contextmanager
def pushd(path: str | Path) -> Iterator[None]:
    old = os.getcwd()
    os.makedirs(path, exist_ok=True)
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def is_semantically_empty_file(path: str, sample_bytes: int = 131072) -> bool:
    """Copied semantics from vendor evaluation.is_semantically_empty_file (utf-8/latin-1)."""
    import codecs
    import unicodedata

    def _contains_semantic_char(text: str) -> bool:
        for ch in text:
            if ch.isspace():
                continue
            cat = unicodedata.category(ch)
            if cat in ("Cc", "Cf"):
                continue
            return True
        return False

    try:
        if os.path.getsize(path) == 0:
            return True
    except OSError:
        pass

    enc = "utf-8"
    sample = b""
    try:
        with open(path, "rb") as fb:
            sample = fb.read(sample_bytes)
        sample.decode("utf-8")
    except UnicodeDecodeError:
        enc = "latin-1"
    except Exception:
        pass

    dec = codecs.getincrementaldecoder(enc)(errors="ignore")
    try:
        with open(path, "rb") as fb:
            while True:
                chunk_b = fb.read(1 << 20)
                if not chunk_b:
                    dec.decode(b"", final=True)
                    break
                text = dec.decode(chunk_b)
                if _contains_semantic_char(text):
                    return False
    except Exception:
        return False
    return True


def merge_texts(root_dir: str, encoding: str = "utf-8") -> str | None:
    root = Path(root_dir)
    merged_parts: list[str] = []
    found_non_log = False
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() == ".log":
            continue
        found_non_log = True
        try:
            merged_parts.append(p.read_text(encoding=encoding))
        except Exception:
            continue
    if not found_non_log:
        return None
    merged = "\n".join(merged_parts)
    return merged if merged.strip() else None


def count_broken_tool_calls(observations: list[str]) -> int:
    n = 0
    for obs in observations:
        if not obs:
            continue
        if any(m in obs for m in BROKEN_TOOL_MARKERS):
            n += 1
    return n


def _serialize_memory_steps(agent: CodeAgent) -> list[dict[str, Any]]:
    steps = []
    for step in agent.memory.steps:
        if hasattr(step, "dict"):
            steps.append(step.dict())
        else:
            steps.append({"repr": repr(step)})
    return steps


def _last_model_input_messages(agent: CodeAgent) -> list[dict[str, Any]] | None:
    """Exact final message list sent to the model at the last step."""
    for step in reversed(agent.memory.steps):
        msgs = getattr(step, "model_input_messages", None)
        if msgs:
            # Prefer the adapter's already-converted last_messages if available
            return None  # signal to use model.last_messages
    return None


def _collect_tool_trace(agent: CodeAgent) -> list[dict[str, Any]]:
    trace: list[dict[str, Any]] = []
    for step in agent.memory.steps:
        tool_calls = getattr(step, "tool_calls", None) or []
        obs = getattr(step, "observations", None)
        code_action = getattr(step, "code_action", None)
        for tc in tool_calls:
            name = getattr(tc, "name", None) or (tc.get("name") if isinstance(tc, dict) else None)
            args = getattr(tc, "arguments", None) or (tc.get("arguments") if isinstance(tc, dict) else None)
            trace.append(
                {
                    "name": name,
                    "arguments": args,
                    "observation": (obs or "")[:2000],
                    "code_action": code_action,
                }
            )
        if not tool_calls and obs:
            trace.append(
                {
                    "name": None,
                    "arguments": None,
                    "observation": (obs or "")[:2000],
                    "code_action": code_action,
                }
            )
    return trace


def _copy_workdir_files(workdir: Path, dest: Path, skip_names: set[str]) -> list[str]:
    dest.mkdir(parents=True, exist_ok=True)
    created: list[str] = []
    for p in workdir.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(workdir)
        if rel.name in skip_names or rel.suffix.lower() == ".log":
            continue
        # Skip our episode artifact names if any landed here
        out = dest / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        data = p.read_bytes()[:MAX_COPIED_FILE_BYTES]
        out.write_bytes(data)
        created.append(str(rel))
    return created


def load_tasks(task_type: int) -> list[dict[str, Any]]:
    path = VENDOR_ROOT / "dataset" / f"task_{task_type}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def run_episode(
    *,
    model_alias: str,
    task: dict[str, Any],
    episode_sample: int,
    run_id: str,
    cache_namespace: str,
    budget_cap_usd: float,
    seed: int | None = None,
    chat_fn: Callable[..., dict[str, Any]] | None = None,
    max_steps: int = 30,
) -> dict[str, Any]:
    """Run one Framework B episode; write artifacts under runs/<run_id>/episodes/<episode_id>/."""
    from memorytask.client import chat as default_chat

    chat_fn = chat_fn or default_chat
    if seed is not None:
        random.seed(seed)

    task_type = int(task["task_type"])
    task_index = int(task["task_index"])
    episode_id = f"{model_alias}__type{task_type}__idx{task_index}__s{episode_sample}"

    root = repo_root()
    run_dir = root / "runs" / run_id
    ep_dir = run_dir / "episodes" / episode_id
    workdir = run_dir / "workdirs" / episode_id
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    ep_dir.mkdir(parents=True, exist_ok=True)

    model_kwargs: dict[str, Any] = dict(
        model_alias=model_alias,
        run_id=run_id,
        task_id=episode_id,
        cache_namespace=cache_namespace,
        episode_id=episode_id,
        budget_cap_usd=budget_cap_usd,
        episode_sample=episode_sample,
        chat_fn=chat_fn,
        temperature=SUBJECT_TEMPERATURE,
        max_tokens=SUBJECT_MAX_TOKENS,
    )
    model = MemorytaskModel(**model_kwargs)

    if task_type not in (1, 2, 3, 4, 5):
        raise ValueError(f"Unsupported task type: {task_type}")

    # Upstream agent_runner: types 1/4/5 → all tools broken; else read_txt working.
    if task_type in (1, 4, 5):
        tools = get_custom_tools([])
    else:
        tools = get_custom_tools(["read_txt"])
    agent = CodeAgent(
        tools=tools,
        model=model,
        add_base_tools=False,
        max_steps=max_steps,
        # Default LocalPythonExecutor; no extra authorized imports (brief SAFETY).
        additional_authorized_imports=[],
        executor_type="local",
    )

    instruction = task["instruction"]
    attached: str | None = None
    error: str | None = None
    final_answer = ""
    agent_log = ""

    try:
        with pushd(workdir):
            if task_type == 1:
                # Write random-content file under the task's file name inside workdir
                Path(task["file_name"]).parent.mkdir(parents=True, exist_ok=True)
                Path(task["file_name"]).write_text(generate_random_content(), encoding="utf-8")
                attached = str(Path(task["file_name"]).resolve())
            elif task_type == 2:
                Path(task["file_name"]).parent.mkdir(parents=True, exist_ok=True)
                Path(task["file_name"]).write_text(task["file_content"], encoding="utf-8")
                attached = str(Path(task["file_name"]).resolve())
            elif task_type == 3:
                Path(task["txt_file_name"]).parent.mkdir(parents=True, exist_ok=True)
                Path(task["txt_file_name"]).write_text(task["file_content"], encoding="utf-8")
                create_dummy_pptx(task["pptx_file_name"], num_slides=2)
                attached = str(Path(task["pptx_file_name"]).resolve())

            buf = io.StringIO()
            with redirect_stdout(buf), redirect_stderr(buf):
                result = agent.run(
                    task=instruction,
                    additional_args={"attached_path_or_url": attached},
                    reset=True,
                )
            agent_log = ANSI_ESCAPE.sub("", buf.getvalue())
            final_answer = result if isinstance(result, str) else str(result)

            # Types 4/5: inspect target file after run (cwd = workdir)
            file_content: str | None = None
            if task_type in (4, 5):
                target_file_name = task.get("target_file_name")
                if target_file_name:
                    if os.path.exists(target_file_name):
                        if not is_semantically_empty_file(target_file_name):
                            try:
                                file_content = Path(target_file_name).read_text(
                                    encoding="utf-8", errors="ignore"
                                )
                            except Exception:
                                file_content = None
                    else:
                        parent_dir = os.path.dirname(target_file_name) or "."
                        file_content = merge_texts(parent_dir)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        file_content = None
        if not agent_log:
            agent_log = ""

    # Memory / trajectory
    try:
        memory_steps = _serialize_memory_steps(agent)
    except Exception:
        memory_steps = []

    last_messages = model.last_messages
    if last_messages is None:
        # Fall back to last ActionStep.model_input_messages via adapter conversion
        for step in reversed(getattr(agent.memory, "steps", []) or []):
            msgs = getattr(step, "model_input_messages", None)
            if msgs:
                from memorytask.framework_b.model_adapter import messages_to_openai_dicts

                last_messages = messages_to_openai_dicts(msgs)
                break

    observations: list[str] = []
    n_steps = 0
    hit_max_steps = False
    for step in getattr(agent.memory, "steps", []) or []:
        if hasattr(step, "step_number"):
            n_steps = max(n_steps, int(step.step_number or 0))
        obs = getattr(step, "observations", None)
        if obs:
            observations.append(str(obs))
        if getattr(step, "error", None) is not None:
            err_obj = step.error
            # AgentMaxStepsError etc.
            ename = type(err_obj).__name__
            if "MaxSteps" in ename or "max_steps" in str(err_obj).lower():
                hit_max_steps = True
    if n_steps >= max_steps:
        hit_max_steps = True

    broken_tool_calls = count_broken_tool_calls(observations)
    tool_trace = _collect_tool_trace(agent)

    # Persist agent artifacts
    (ep_dir / "agent_stdout.log").write_text(agent_log, encoding="utf-8")
    (ep_dir / "final_answer.txt").write_text(final_answer or "", encoding="utf-8")
    trajectory = {
        "episode_id": episode_id,
        "last_step_messages": last_messages,
        "memory_steps": memory_steps,
        "subject_params": {
            "temperature": SUBJECT_TEMPERATURE,
            "max_tokens": SUBJECT_MAX_TOKENS,
            "reasoning": "low (from model config)",
        },
    }
    (ep_dir / "trajectory.json").write_text(
        json.dumps(trajectory, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    created_files = _copy_workdir_files(
        workdir, ep_dir / "workdir_files", skip_names={"agent_stdout.log", "trajectory.json"}
    )

    # Judges
    judge_details: dict[str, Any] = {}
    upstream_result: list[Any] | None = None
    judge_call_ids: list[str] = []
    if error is None:
        try:
            prompts = load_prompts()
            judge = DualJudge(
                prompts=prompts,
                run_id=run_id,
                episode_id=episode_id,
                cache_namespace=cache_namespace,
                budget_cap_usd=budget_cap_usd,
                episode_sample=episode_sample,
                chat_fn=chat_fn,
            )
            if task_type == 1:
                upstream_result, judge_details = judge.judge_type_1(task, final_answer)
            elif task_type == 2:
                upstream_result, judge_details = judge.judge_type_2(task, final_answer)
            elif task_type == 3:
                upstream_result, judge_details = judge.judge_type_3(task, final_answer)
            else:
                upstream_result, judge_details = judge.judge_type_4_and_5(
                    task, final_answer, file_content
                )
            judge_call_ids = list(judge.call_ids)
            (ep_dir / "judge_outputs.json").write_text(
                json.dumps(
                    {
                        "upstream_result": upstream_result,
                        "details": judge_details,
                        "all": judge.judge_outputs,
                    },
                    ensure_ascii=False,
                    indent=2,
                    default=str,
                ),
                encoding="utf-8",
            )
        except Exception as jexc:
            error = f"judge_error: {type(jexc).__name__}: {jexc}"
            (ep_dir / "judge_error.txt").write_text(error, encoding="utf-8")

    record = {
        "episode_id": episode_id,
        "model": model_alias,
        "model_alias": model_alias,
        "task_type": task_type,
        "task_index": task_index,
        "sample_index": episode_sample,
        "n_steps": n_steps,
        "hit_max_steps": hit_max_steps,
        "error": error,
        "broken_tool_calls": broken_tool_calls,
        "never_touched_tool": broken_tool_calls == 0 and not any(
            t.get("name") or t.get("code_action") for t in tool_trace
        ),
        "call_ids": list(model.call_ids) + judge_call_ids,
        "final_answer": final_answer,
        "judge_outputs": judge_details,
        "upstream_result": upstream_result,
        "instruction": instruction,
        "created_files": created_files,
        "tool_trace": tool_trace,
        "subject_temperature": SUBJECT_TEMPERATURE,
        "subject_max_tokens": SUBJECT_MAX_TOKENS,
    }
    (ep_dir / "episode.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    return record
