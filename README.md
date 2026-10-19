# 0xClaw BudgetRouter

BudgetRouter predicts compact control tokens for a hackathon-agent phase:

```text
<PHASE_CODING><TOKEN_1.0X><TIME_HIGH><THINK_LOW><REPAIR_1>
```

It is a separately installed component for [0xClaw](https://github.com/0xclaw-ai/0xClaw)
and other Python agents. The model runtime, prompt construction, parsing, and
profile adjustment logic are preserved. The `budget_router` package name avoids
collisions with an application's own `orchestration` package.

- [Paper / preprint](https://spiral.imperial.ac.uk/entities/publication/7a460786-fba9-4b42-b9a0-d128429f67e4)
- [Training pipeline](https://github.com/0xclaw-ai/budget-router-trainer)
- [DeadlineFramer](https://github.com/0xclaw-ai/deadline-framer)

## Install

Use Python 3.11 or newer in the same environment as your agent:

```bash
git clone https://github.com/0xclaw-ai/budget-router.git
python -m pip install -e ./budget-router
```

Model dependencies are installed by this package. The base 0xClaw package does
not need an extra named `budgetrouter`.

## Download the model

The weights are hosted separately: [Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) and
[the BudgetRouter LoRA adapter](https://huggingface.co/0xclaw-ai/budgetrouter-qwen3.5-2b-lora). From your agent checkout:

```bash
hf download Qwen/Qwen3.5-2B --local-dir ./models/qwen3.5-2b
hf download 0xclaw-ai/budgetrouter-qwen3.5-2b-lora --local-dir ./models/budgetrouter-lora
```

`hf` is provided by `huggingface_hub`, installed with Transformers. If access
requires authentication, run `hf auth login` using your own account.
Both local directories are required by the 0xClaw plugin. Inference uses CUDA
when available and CPU otherwise. The model files stay out of Git.

## Use with 0xClaw

Install [0xClaw](https://github.com/0xclaw-ai/0xClaw) and this package, then add
the following to the 0xClaw checkout's `.env`:

```dotenv
OXCLAW_PHASE_CONTROLS=budget-router
OXCLAW_BUDGET_ROUTER_MODEL_PATH=./models/qwen3.5-2b
OXCLAW_BUDGET_ROUTER_ADAPTER_PATH=./models/budgetrouter-lora
```

Run `0xclaw`. Relative model paths resolve from the 0xClaw checkout.
To also enable deadline guidance, install
[DeadlineFramer](https://github.com/0xclaw-ai/deadline-framer) in that environment
and set `OXCLAW_PHASE_CONTROLS=budget-router,deadline-framer`.
This applies to normal phases, `/resume`, and `/redo` in the interactive CLI.

`OXCLAW_BUDGET_ROUTER_MODE=shadow` records predictions without modifying the
profile or injecting deadline guidance. The default mode is `enforce`.
Inference failures emit a warning and leave the original phase profile in use.
Local logs are written to `workspace/hackathon/budget_router.jsonl`.

## Standalone Python usage

```python
from pathlib import Path
from types import SimpleNamespace
from budget_router import BudgetRouterRuntime

config = SimpleNamespace(
    enabled=True,
    mode="enforce",
    model_path="./models/qwen3.5-2b",
    adapter_path="./models/budgetrouter-lora",
    max_new_tokens=48,
    schema_aware=True,
)
router = BudgetRouterRuntime(config, root=Path.cwd(), workspace=Path("workspace"))
decision = router.decide("coding", "build the selected hackathon project")
if not decision.is_model_decision:
    raise RuntimeError(decision.error or decision.source)
print(decision.control_text)
```

`budget_router.budget_router` also exposes the existing parsing, feature,
logging, and profile utilities. No 0xClaw installation is needed for this API.

## License

MIT for this repository's code. The base model and adapter have their own model cards.
