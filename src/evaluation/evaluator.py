"""evaluation pipeline."""
from .metrics import helpfulness, plan_adherence


def evaluation_pipeline(prompt: str, output: str) -> dict[str, object]:
    """evaluates the chat's response on varying metrics"""
    metric_functions = {
        "helpfulness": helpfulness,
        "plan_adherence": plan_adherence,
    }

    # results = {}

    # for name, func in metric_functions.items():
    #     r = func(prompt, output)

    #     if hasattr(r, "model_dump"):
    #         r = r.model_dump()
    #     elif hasattr(r, "__dict__"):
    #         r = vars(r)

    #     results[name] = r

    # return results
    return {name: func(prompt, output) for name, func in metric_functions.items()}
