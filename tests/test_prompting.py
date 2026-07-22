from block_spec.prompting import build_messages, format_failfast_math_prompt


def test_failfast_math_uses_one_user_message_without_system_role():
    messages = build_messages(
        "Natalia sold 48 clips.",
        prompt_style="failfast_math",
        system_prompt="must be ignored",
    )
    assert [message["role"] for message in messages] == ["user"]
    assert messages[0]["content"] == format_failfast_math_prompt(
        "Natalia sold 48 clips."
    )
    assert "must be ignored" not in messages[0]["content"]
    assert "\\boxed{}" in messages[0]["content"]


def test_system_user_keeps_two_roles():
    messages = build_messages(
        "question",
        prompt_style="system_user",
        system_prompt="system",
    )
    assert messages == [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "question"},
    ]
