"""evaluation pipeline."""
from .metrics import helpfulness, plan_adherence


async def evaluation_pipeline(prompt: str, output: str) -> dict[str, object]:
    """evaluates the chat's response on varying metrics"""
    metric_functions = {
        "helpfulness": helpfulness,
        "plan_adherence": plan_adherence,
    }
    return {name: await func(prompt, output) for name, func in metric_functions.items()}
