# 0xClaw BudgetRouter

BudgetRouter predicts compact control tokens for a hackathon-agent phase:

```text
<PHASE_CODING><TOKEN_1.0X><TIME_HIGH><THINK_LOW><REPAIR_1>
```

The runtime loads a local Qwen base model plus the trained BudgetRouter LoRA adapter. Model weights are not stored in this repository.

- Base model: [Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B)
- LoRA adapter: [0xclaw-ai/budgetrouter-qwen3.5-2b-lora](https://huggingface.co/0xclaw-ai/budgetrouter-qwen3.5-2b-lora)

## Install

```powershell
pip install -e .
```

## Use with 0xClaw

BudgetRouter can be enabled as an optional 0xClaw control layer. It loads a local Qwen base model plus the trained BudgetRouter LoRA adapter, then emits phase control tokens that 0xClaw can pass to DeadlineFramer.

```powershell
pip install -e ".[budgetrouter]"

$env:OXCLAW_ENABLE_BUDGET_DEADLINE="1"
$env:OXCLAW_BUDGET_ROUTER_MODEL_PATH="<path-to-local-qwen3.5-2b>"
$env:OXCLAW_BUDGET_ROUTER_ADAPTER_PATH="<path-to-budgetrouter-lora>"
0xclaw
```

Download the base model from [Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) and the BudgetRouter LoRA adapter from [0xclaw-ai/budgetrouter-qwen3.5-2b-lora](https://huggingface.co/0xclaw-ai/budgetrouter-qwen3.5-2b-lora), then point the two environment variables above at the local directories.

If Transformers reports that `qwen3_5` is unknown, upgrade Transformers to a version that supports Qwen3.5 before running the command above.

## Usage

```python
from pathlib import Path
from types import SimpleNamespace

from orchestration.budget_router import BudgetRouterRuntime

config = SimpleNamespace(
    enabled=True,
    mode="enforce",
    model_path="<path-to-local-qwen3.5-2b>",
    adapter_path="<path-to-budgetrouter-lora>",
    max_new_tokens=48,
    schema_aware=True,
)

router = BudgetRouterRuntime(config, root=Path.cwd(), workspace=Path("workspace"))
decision = router.decide("coding", "build the selected hackathon project")
print(decision.control_text)
```

## Files

- `orchestration/budget_router.py`: BudgetRouter runtime, prompt construction, control parsing, profile adjustment, and JSONL logging.
- `orchestration/model_profiles.py`: minimal profile dataclass used by the current BudgetRouter interface.

This package preserves the current 0xClaw import surface so the implementation can be reused without changing its logic.
