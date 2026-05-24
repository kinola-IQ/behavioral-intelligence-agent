from src.core.utils import parse_agent_json_response


def test_parse_submit_review_tool_call() -> None:
    messages = [
        {
            "type": "ai",
            "content": "",
            "tool_calls": [
                {
                    "name": "submit_review",
                    "args": {
                        "predicted_rating": 5,
                        "predicted_review": "Excellent value.",
                    },
                }
            ],
        },
    ]
    parsed = parse_agent_json_response(messages)
    assert parsed["predicted_rating"] == 5
    assert parsed["predicted_review"] == "Excellent value."


def test_parse_skips_tool_messages_and_uses_last_ai_json() -> None:
    messages = [
        {"type": "human", "content": "generate a review"},
        {"type": "ai", "content": "", "tool_calls": [{"name": "model_user"}]},
        {"type": "tool", "content": '{"rating_bias": 4}'},
        {
            "type": "ai",
            "content": '```json\n{"predicted_rating": 4, "predicted_review": "Great product."}\n```',
        },
    ]
    parsed = parse_agent_json_response(messages)
    assert parsed["predicted_rating"] == 4
    assert parsed["predicted_review"] == "Great product."
