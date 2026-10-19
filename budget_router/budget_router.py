"""Optional runtime bridge for BudgetRouter control-token predictions."""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from .model_profiles import ModelProfile


CONTROL_RE = re.compile(
    r"<PHASE_([A-Z_]+)>\s*"
    r"<(TOKEN_[0-9.]+X)>\s*"
    r"<(TIME_[A-Z_]+)>\s*"
    r"<(THINK_[A-Z_]+)>\s*"
    r"<(REPAIR_[0-9]+)>",
    re.IGNORECASE,
)

VALID_PHASES = {"research", "idea", "selection", "planning", "coding", "testing", "documentation"}
VALID_TOKENS = {"TOKEN_0.25X", "TOKEN_0.5X", "TOKEN_0.75X", "TOKEN_1.0X", "TOKEN_1.25X"}
VALID_TIMES = {"TIME_LOW", "TIME_MEDIUM", "TIME_HIGH", "TIME_CRITICAL"}
VALID_THINKS = {"THINK_OFF", "THINK_LOW", "THINK_HIGH"}
VALID_REPAIRS = {"REPAIR_0", "REPAIR_1", "REPAIR_2"}

SCHEMA_AWARE_INSTRUCTIONS = """
Output schema:
Return exactly one line with five adjacent tokens and nothing else.
Each token must include both the opening "<" and closing ">" brackets.

Token 1 must be exactly one of:
<PHASE_RESEARCH>, <PHASE_IDEA>, <PHASE_SELECTION>, <PHASE_PLANNING>, <PHASE_CODING>, <PHASE_TESTING>, <PHASE_DOCUMENTATION>

Token 2 must be exactly one of:
<TOKEN_0.25X>, <TOKEN_0.5X>, <TOKEN_0.75X>, <TOKEN_1.0X>, <TOKEN_1.25X>

Token 3 must be exactly one of:
<TIME_LOW>, <TIME_MEDIUM>, <TIME_HIGH>, <TIME_CRITICAL>

Token 4 must be exactly one of:
<THINK_OFF>, <THINK_LOW>, <THINK_HIGH>

Token 5 must be exactly one of:
<REPAIR_0>, <REPAIR_1>, <REPAIR_2>

Rules:
- The first token must match the requested Phase field.
- Do not invent labels such as TOKEN_1.5X, TIME_SHORT, TIME_LONG, THINK_MEDIUM, or REPAIR_3.
- Do not output bare labels without brackets.
- Do not output XML closing tags such as </TOKEN>, </TIME>, or </REPAIR>.
- Do not use dot notation such as <PHASE.idea>.
- Return no prose, JSON, Markdown, code fences, or explanation.
""".strip()

PHASE_TO_BUDGET_PHASE = {
    "research": "research",
    "idea": "idea",
    "selection": "selection",
    "planning": "planning",
    "coding": "coding",
    "testing": "testing",
    "doc": "documentation",
}

TOKEN_MULTIPLIERS = {
    "TOKEN_0.25X": 0.25,
    "TOKEN_0.5X": 0.5,
    "TOKEN_0.75X": 0.75,
    "TOKEN_1.0X": 1.0,
    "TOKEN_1.25X": 1.25,
}

TIME_MULTIPLIERS = {
    "TIME_LOW": 0.75,
    "TIME_MEDIUM": 1.0,
    "TIME_HIGH": 1.25,
    "TIME_CRITICAL": 1.5,
}

PROJECT_FEATURE_SKIP_DIRS = frozenset(
    {
        ".cache",
        ".git",
        ".mypy_cache",
        ".next",
        ".nuxt",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
        ".venv",
        "__pycache__",
        "build",
        "dist",
        "env",
        "node_modules",
        "target",
        "venv",
    }
)

KEYWORD_GROUPS = {
    "research": ("research", "paper", "dataset", "benchmark", "experiment", "model", "train"),
    "ml": ("ai", "ml", "llm", "agent", "inference", "embedding", "rag", "model"),
    "product": ("demo", "user", "dashboard", "workflow", "market", "customer", "app"),
    "architecture": ("api", "backend", "frontend", "database", "service", "deploy", "docker"),
    "external_api": ("api", "sdk", "oauth", "webhook", "sponsor", "integration", "key"),
    "testing": ("test", "pytest", "ci", "coverage", "validate", "failure", "bug"),
    "documentation": ("readme", "docs", "submission", "pitch", "usage", "install"),
}

COMMIT_KEYWORD_GROUPS = {
    "fix": ("fix", "bug", "error", "failure", "patch", "repair"),
    "test": ("test", "pytest", "validate", "ci", "coverage"),
    "doc": ("doc", "readme", "submission", "pitch", "usage"),
    "feature": ("feature", "implement", "build", "add", "create"),
    "refactor": ("refactor", "cleanup", "restructure", "simplify"),
}


@dataclass(slots=True)
class BudgetDecision:
    phase: str
    control_text: str
    token: str
    time: str
    think: str
    repair: str
    source: str
    raw_text: str = ""
    error: str | None = None
    prompt: str = ""
    features: dict[str, Any] = field(default_factory=dict)

    @property
    def is_model_decision(self) -> bool:
        return self.source == "model" and self.error is None

    @property
    def repair_rounds(self) -> int:
        return int(self.repair.rsplit("_", 1)[-1])

    def to_payload(self, effective_profile: ModelProfile | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "source": self.source,
            "control_text": self.control_text,
            "phase": self.phase,
            "token": self.token,
            "time": self.time,
            "think": self.think,
            "repair": self.repair,
            "repair_rounds": self.repair_rounds,
        }
        if effective_profile is not None:
            payload["effective"] = profile_to_dict(effective_profile)
        if self.error:
            payload["error"] = self.error
        return payload


class BudgetRouterRuntime:
    """Lazy runtime wrapper around the local BudgetRouter LoRA adapter."""

    def __init__(self, config: Any, *, root: Path, workspace: Path):
        self.config = config
        self.root = root
        self.workspace = workspace
        self._model: Any = None
        self._tokenizer: Any = None
        self._device: Any = None
        self._torch: Any = None

    @property
    def enabled(self) -> bool:
        return bool(getattr(self.config, "enabled", False))

    @property
    def mode(self) -> str:
        return str(getattr(self.config, "mode", "enforce") or "enforce")

    def decide(self, phase: str, command: str) -> BudgetDecision:
        budget_phase = PHASE_TO_BUDGET_PHASE.get(phase, phase)
        if not self.enabled:
            return self._inactive_decision(budget_phase, "disabled")

        prompt, features = build_budget_prompt(self.workspace, budget_phase, command)
        try:
            text = self._generate(prompt)
            parsed = parse_control(text)
            if parsed is None:
                return self._inactive_decision(
                    budget_phase,
                    "parse_failed",
                    prompt=prompt,
                    features=features,
                    raw_text=text,
                    error=f"Could not parse BudgetRouter output: {text[:200]}",
                )
            return BudgetDecision(
                phase=parsed["phase"],
                control_text=control_to_text(parsed),
                token=parsed["token"],
                time=parsed["time"],
                think=parsed["think"],
                repair=parsed["repair"],
                source="model",
                raw_text=text,
                prompt=prompt,
                features=features,
            )
        except Exception as exc:  # noqa: BLE001
            return self._inactive_decision(
                budget_phase,
                "error",
                prompt=prompt,
                features=features,
                error=str(exc),
            )

    def _inactive_decision(
        self,
        phase: str,
        source: str,
        *,
        prompt: str = "",
        features: dict[str, Any] | None = None,
        raw_text: str = "",
        error: str | None = None,
    ) -> BudgetDecision:
        parsed = static_prediction(phase)
        return BudgetDecision(
            phase=phase,
            control_text=control_to_text(parsed),
            token=parsed["token"],
            time=parsed["time"],
            think=parsed["think"],
            repair=parsed["repair"],
            source=source,
            raw_text=raw_text,
            error=error,
            prompt=prompt,
            features=features or {},
        )

    def _generate(self, prompt: str) -> str:
        self._ensure_model()
        row = {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a phase-aware budget planner for a hackathon agent. "
                        "Choose compact control tokens for the requested phase."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        }
        messages = build_messages(row, schema_aware=bool(getattr(self.config, "schema_aware", True)))
        prompt_text = apply_chat_template_no_thinking(self._tokenizer, messages)
        inputs = self._tokenizer(prompt_text, return_tensors="pt").to(self._device)
        max_new_tokens = int(getattr(self.config, "max_new_tokens", 48) or 48)
        with self._torch.inference_mode():
            output_ids = self._model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=self._tokenizer.pad_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
            )
        gen_ids = output_ids[0, inputs["input_ids"].shape[1] :]
        return self._tokenizer.decode(gen_ids, skip_special_tokens=True)

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        try:
            import torch
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "BudgetRouter dependencies are missing. Install transformers, peft, and torch "
                f"in the current environment. Import error: {exc}"
            ) from exc

        model_path = resolve_runtime_path(str(getattr(self.config, "model_path", "")), self.root)
        adapter_path = resolve_runtime_path(str(getattr(self.config, "adapter_path", "")), self.root)
        if model_path is None:
            raise FileNotFoundError("BudgetRouter base model path is empty")
        if not model_path.exists():
            raise FileNotFoundError(f"BudgetRouter base model not found: {model_path}")
        if adapter_path is not None and not adapter_path.exists():
            raise FileNotFoundError(f"BudgetRouter adapter not found: {adapter_path}")

        tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True, use_fast=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float32
        if device.type == "cuda" and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        elif device.type == "cuda":
            dtype = torch.float16

        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            trust_remote_code=True,
            torch_dtype=dtype,
        )
        if adapter_path is not None:
            model = PeftModel.from_pretrained(model, str(adapter_path))
        model.to(device)
        model.eval()

        self._torch = torch
        self._model = model
        self._tokenizer = tokenizer
        self._device = device


def parse_control(text: str) -> dict[str, str] | None:
    match = CONTROL_RE.search((text or "").strip())
    if not match:
        return None
    phase, token, time_label, think, repair = match.groups()
    parsed = {
        "phase": phase.lower(),
        "token": token.upper(),
        "time": time_label.upper(),
        "think": think.upper(),
        "repair": repair.upper(),
    }
    if (
        parsed["phase"] not in VALID_PHASES
        or parsed["token"] not in VALID_TOKENS
        or parsed["time"] not in VALID_TIMES
        or parsed["think"] not in VALID_THINKS
        or parsed["repair"] not in VALID_REPAIRS
    ):
        return None
    return parsed


def control_to_text(pred: dict[str, str] | None) -> str:
    if pred is None:
        return ""
    return (
        f"<PHASE_{pred['phase'].upper()}>"
        f"<{pred['token']}>"
        f"<{pred['time']}>"
        f"<{pred['think']}>"
        f"<{pred['repair']}>"
    )


def static_prediction(phase: str) -> dict[str, str]:
    return {
        "phase": phase,
        "token": "TOKEN_1.0X",
        "time": "TIME_MEDIUM",
        "think": "THINK_LOW",
        "repair": "REPAIR_0",
    }


def build_messages(
    row: dict[str, Any],
    few_shot_rows: list[dict[str, Any]] | None = None,
    schema_aware: bool = False,
) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = [row["messages"][0]]
    for shot in few_shot_rows or []:
        messages.append(shot["messages"][1])
        messages.append(shot["messages"][2])
    user_message = dict(row["messages"][1])
    if schema_aware:
        user_message["content"] = f"{user_message['content'].rstrip()}\n\n{SCHEMA_AWARE_INSTRUCTIONS}\n"
    messages.append(user_message)
    return messages


def apply_chat_template_no_thinking(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def apply_budget_decision(
    base: ModelProfile,
    decision: BudgetDecision | None,
    *,
    mode: str,
    min_max_tokens: int = 512,
    min_timeout_s: int = 60,
) -> ModelProfile:
    if decision is None or not decision.is_model_decision or mode == "shadow":
        return base

    token_multiplier = TOKEN_MULTIPLIERS[decision.token]
    time_multiplier = TIME_MULTIPLIERS[decision.time]
    return ModelProfile(
        phase=base.phase,
        provider=base.provider,
        model=base.model,
        max_tokens=max(min_max_tokens, int(round(base.max_tokens * token_multiplier))),
        temperature=base.temperature,
        timeout_s=max(min_timeout_s, int(round(base.timeout_s * time_multiplier))),
    )


def profile_to_dict(profile: ModelProfile) -> dict[str, Any]:
    return asdict(profile)


def log_budget_decision(
    path: Path,
    *,
    phase: str,
    command: str,
    decision: BudgetDecision | None,
    base_profile: ModelProfile,
    effective_profile: ModelProfile,
    mode: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "phase": phase,
        "command": command,
        "mode": mode,
        "decision": decision.to_payload(effective_profile) if decision else None,
        "raw_text": decision.raw_text if decision else "",
        "error": decision.error if decision else None,
        "features": decision.features if decision else {},
        "base_profile": profile_to_dict(base_profile),
        "effective_profile": profile_to_dict(effective_profile),
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_budget_prompt(workspace: Path, budget_phase: str, command: str) -> tuple[str, dict[str, Any]]:
    project = extract_project_features(workspace, command)
    compact_features = {
        "language": project["language"],
        "topics": project["topics"][:12],
        "duration_bin": project["duration_bin"],
        "duration_hours": round(project["duration_hours"] or 0.0, 2),
        "repo_size_kb": project["repo_size_kb"],
        "readme_words": project["readme_words"],
        "commit_count": project["commit_count"],
        "contributors_count": project["contributors_count"],
        "has_demo": project["has_demo"],
        "has_install": project["has_install"],
        "has_usage": project["has_usage"],
        "has_tests": project["has_tests"],
        "scientific_probability": round(project["scientific_probability"], 3),
        "keyword_counts": project["keyword_counts"],
        "commit_keyword_counts": project["commit_keyword_counts"],
    }
    prompt = (
        "Allocate the next-phase budget for a hackathon agent.\n"
        "Return only control tokens.\n\n"
        f"Project: {project['full_name']}\n"
        f"Description: {project['description']}\n"
        f"Phase: {budget_phase}\n"
        f"Project features: {json.dumps(compact_features, ensure_ascii=False, sort_keys=True)}\n"
        f"README excerpt: {project['readme_excerpt']}\n"
    )
    return prompt, compact_features


def extract_project_features(workspace: Path, command: str) -> dict[str, Any]:
    hackathon = workspace / "hackathon"
    context = read_json(hackathon / "context.json")
    selected = read_json(hackathon / "selected_idea.json")
    ideas = read_json(hackathon / "ideas.json")
    tasks = read_json(hackathon / "tasks.json")
    test_results = read_json(hackathon / "test_results.json")

    text_sources = [
        command,
        flatten_strings(context),
        flatten_strings(selected),
        flatten_strings(ideas),
        read_text(hackathon / "research_summary.md"),
        read_text(hackathon / "plan.md"),
        flatten_strings(tasks),
        flatten_strings(test_results),
        read_text(hackathon / "project" / "README.md"),
    ]
    joined_text = "\n".join(text for text in text_sources if text)
    lowered = joined_text.lower()

    task_count, estimated_hours = task_stats(tasks)
    project_dir = hackathon / "project"
    project_file_count = count_files(project_dir)
    repo_size_kb = round(dir_size(project_dir) / 1024, 2)
    if repo_size_kb <= 0:
        repo_size_kb = round(sum_file_sizes(hackathon, ("*.json", "*.md")) / 1024, 2)

    keyword_counts = {name: count_any(lowered, words) for name, words in KEYWORD_GROUPS.items()}
    commit_keyword_counts = {
        name: count_any(lowered, words) for name, words in COMMIT_KEYWORD_GROUPS.items()
    }
    return {
        "full_name": project_name(context, selected, ideas),
        "description": project_description(command, context, selected, ideas),
        "language": detect_language(project_dir, selected, tasks),
        "topics": extract_topics(context, selected, ideas, tasks, lowered),
        "duration_hours": estimated_hours or default_duration_hours(tasks, project_file_count),
        "duration_bin": duration_bin(estimated_hours or default_duration_hours(tasks, project_file_count)),
        "repo_size_kb": repo_size_kb,
        "readme_words": word_count(read_text(hackathon / "project" / "README.md")) or word_count(
            read_text(hackathon / "research_summary.md")
        ),
        "commit_count": max(3, task_count * 2 + project_file_count),
        "contributors_count": 1,
        "has_demo": contains_any(lowered, ("demo", "pitch", "submission", "showcase")),
        "has_install": contains_any(lowered, ("install", "requirements.txt", "package.json", "pip ")),
        "has_usage": contains_any(lowered, ("usage", "how to run", "quickstart", "demo")),
        "has_tests": has_tests(project_dir, lowered),
        "scientific_probability": scientific_probability(keyword_counts, lowered),
        "keyword_counts": keyword_counts,
        "commit_keyword_counts": commit_keyword_counts,
        "readme_excerpt": truncate_words(joined_text, 220),
    }


def read_json(path: Path) -> Any:
    try:
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def read_text(path: Path) -> str:
    try:
        if not path.exists() or not path.is_file():
            return ""
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def flatten_strings(value: Any, *, limit: int = 12000) -> str:
    out: list[str] = []

    def visit(obj: Any) -> None:
        if sum(len(item) for item in out) >= limit:
            return
        if isinstance(obj, str):
            out.append(obj)
        elif isinstance(obj, dict):
            for item in obj.values():
                visit(item)
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                visit(item)

    visit(value)
    return " ".join(out)[:limit]


def count_any(text: str, words: tuple[str, ...]) -> int:
    return sum(text.count(word) for word in words)


def contains_any(text: str, words: tuple[str, ...]) -> bool:
    return any(word in text for word in words)


def word_count(text: str) -> int:
    return len(re.findall(r"\w+", text or ""))


def truncate_words(text: str, max_words: int) -> str:
    words = re.findall(r"\S+", text or "")
    return " ".join(words[:max_words])


def task_stats(tasks: Any) -> tuple[int, float]:
    if not isinstance(tasks, dict):
        return 0, 0.0
    count = 0
    hours = 0.0
    for epic in tasks.get("epics", []) or []:
        if not isinstance(epic, dict):
            continue
        for task in epic.get("tasks", []) or []:
            if not isinstance(task, dict):
                continue
            count += 1
            try:
                hours += float(task.get("estimated_hours") or 0.0)
            except (TypeError, ValueError):
                pass
    return count, hours


def default_duration_hours(tasks: Any, project_file_count: int) -> float:
    task_count, hours = task_stats(tasks)
    if hours:
        return hours
    if task_count:
        return float(task_count * 2)
    return float(max(4, min(72, project_file_count * 2)))


def duration_bin(hours: float) -> str:
    if hours < 8:
        return "short"
    if hours < 24:
        return "medium"
    if hours < 48:
        return "long"
    return "very_long"


def count_files(path: Path) -> int:
    if not path.exists() or not path.is_dir():
        return 0
    return sum(1 for _item in iter_project_files(path))


def dir_size(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for item in iter_project_files(path):
        try:
            total += item.stat().st_size
        except OSError:
            pass
    return total


def iter_project_files(path: Path) -> Iterator[Path]:
    if not path.exists() or not path.is_dir():
        return

    for current_dir, dirs, files in os.walk(path, onerror=lambda _error: None):
        dirs[:] = [name for name in dirs if not skip_project_feature_dir(name)]
        current_path = Path(current_dir)
        for name in files:
            yield current_path / name


def skip_project_feature_dir(name: str) -> bool:
    lowered = name.lower()
    return lowered in PROJECT_FEATURE_SKIP_DIRS or lowered.startswith(".venv")


def sum_file_sizes(root: Path, patterns: tuple[str, ...]) -> int:
    total = 0
    for pattern in patterns:
        for item in root.glob(pattern):
            try:
                if item.is_file():
                    total += item.stat().st_size
            except OSError:
                pass
    return total


def extract_topics(context: Any, selected: Any, ideas: Any, tasks: Any, lowered: str) -> list[str]:
    topics: list[str] = []
    for value in (
        context.get("sdks_to_investigate", []) if isinstance(context, dict) else [],
        context.get("recommended_integration_priority", []) if isinstance(context, dict) else [],
        selected.get("integrations", []) if isinstance(selected, dict) else [],
        ideas.get("recommendation", "") if isinstance(ideas, dict) else "",
        tasks.get("tech_stack", {}) if isinstance(tasks, dict) else {},
    ):
        add_topic_values(topics, value)
    for term in ("fastapi", "react", "nextjs", "typescript", "python", "blockchain", "agent", "rag", "flock", "virtual", "unibase"):
        if term in lowered:
            topics.append(term)
    deduped: list[str] = []
    seen: set[str] = set()
    for topic in topics:
        clean = re.sub(r"[^a-z0-9_.-]+", "-", str(topic).lower()).strip("-")
        if clean and clean not in seen:
            seen.add(clean)
            deduped.append(clean)
    return deduped[:20]


def add_topic_values(topics: list[str], value: Any) -> None:
    if isinstance(value, str):
        topics.append(value)
    elif isinstance(value, dict):
        for item in value.values():
            add_topic_values(topics, item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            add_topic_values(topics, item)


def detect_language(project_dir: Path, selected: Any, tasks: Any) -> str:
    stack_text = " ".join([flatten_strings(selected).lower(), flatten_strings(tasks).lower()])
    if "typescript" in stack_text or "nextjs" in stack_text or "react" in stack_text:
        return "TypeScript"
    if "python" in stack_text or "fastapi" in stack_text:
        return "Python"

    ext_counts: dict[str, int] = {}
    for item in iter_project_files(project_dir):
        ext_counts[item.suffix.lower()] = ext_counts.get(item.suffix.lower(), 0) + 1
    if not ext_counts:
        return "Python"
    ext = max(ext_counts, key=ext_counts.get)
    return {
        ".py": "Python",
        ".ts": "TypeScript",
        ".tsx": "TypeScript",
        ".js": "JavaScript",
        ".jsx": "JavaScript",
        ".go": "Go",
        ".rs": "Rust",
        ".sol": "Solidity",
    }.get(ext, ext.lstrip(".").title() or "Python")


def has_tests(project_dir: Path, lowered: str) -> bool:
    if contains_any(lowered, ("pytest", "test_results", "npm test", "coverage")):
        return True
    if not project_dir.exists():
        return False
    return any(
        item.name.startswith("test_")
        or item.name.endswith("_test.py")
        or item.name.endswith(".test.ts")
        or item.name.endswith(".spec.ts")
        for item in iter_project_files(project_dir)
    )


def scientific_probability(keyword_counts: dict[str, int], lowered: str) -> float:
    signal = keyword_counts.get("research", 0) + keyword_counts.get("ml", 0)
    if signal <= 0:
        return 0.1
    return min(0.95, 0.2 + 0.08 * signal + (0.15 if "benchmark" in lowered else 0.0))


def project_name(context: Any, selected: Any, ideas: Any) -> str:
    if isinstance(selected, dict) and selected.get("name"):
        return str(selected["name"])
    if isinstance(context, dict):
        hackathon = context.get("hackathon")
        if isinstance(hackathon, dict) and hackathon.get("name"):
            return str(hackathon["name"])
    if isinstance(ideas, dict) and ideas.get("recommendation"):
        return str(ideas["recommendation"])
    return "0xClaw hackathon project"


def project_description(command: str, context: Any, selected: Any, ideas: Any) -> str:
    for data in (selected, ideas, context):
        text = flatten_strings(data, limit=1000)
        if text:
            return truncate_words(text, 80)
    return truncate_words(command, 80) or "Autonomous hackathon project."


def resolve_runtime_path(raw: str, root: Path) -> Path | None:
    if not raw:
        return None
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path
    return (root / path).resolve()


def resolve_log_path(raw: str, *, root: Path, workspace: Path) -> Path:
    path = Path(raw or "./workspace/hackathon/budget_router.jsonl")
    if path.is_absolute():
        return path
    normalized = str(path).replace("\\", "/").lstrip("./")
    if normalized == "workspace" or normalized.startswith("workspace/"):
        suffix = normalized.removeprefix("workspace").lstrip("/")
        return (workspace / suffix).resolve()
    return (root / path).resolve()
