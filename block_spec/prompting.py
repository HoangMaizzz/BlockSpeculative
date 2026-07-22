from __future__ import annotations


PROMPT_STYLES = ("system_user", "failfast_math")


def format_failfast_math_prompt(problem: str) -> str:
    """Clean equivalent of the GSM8K prompt used by FailFast."""
    problem = problem.strip()
    if not problem:
        raise ValueError("Problem is empty")
    return (
        "Solve the following math problem efficiently and clearly. "
        "Please reason step by step, separate logical reasoning steps with two "
        "newline characters, and put your final answer within \\boxed{}.\n"
        f"Problem: {problem}"
    )


def build_messages(
    user_prompt: str,
    *,
    prompt_style: str = "system_user",
    system_prompt: str = "You are a careful mathematics tutor.",
) -> list[dict[str, str]]:
    if prompt_style == "failfast_math":
        return [{"role": "user", "content": format_failfast_math_prompt(user_prompt)}]
    if prompt_style == "system_user":
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    raise ValueError(f"Unsupported prompt style: {prompt_style}")


def render_chat_prompt(tokenizer, messages: list[dict[str, str]]) -> str:
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
