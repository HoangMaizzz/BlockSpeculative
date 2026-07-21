import math

from block_spec.drafter_adapter import DraftCandidates
from block_spec.drafter_tree_inspection import build_first_step_tree_report, format_first_step_tree
from block_spec.types import BlockCandidate


class TinyTokenizer:
    def convert_ids_to_tokens(self, token_id): return f"tok_{token_id}"
    def decode(self, ids, skip_special_tokens=False): return "|".join(map(str, ids))


def test_first_step_report_contains_token_and_joint_probabilities():
    candidate = BlockCandidate((1, 2, 3), (math.log(0.5), math.log(0.4), math.log(0.25)),
                               math.log(0.5) + math.log(0.4) + math.log(0.25))
    draft = DraftCandidates([candidate], (0.9, 0.8, 0.7), 0.05)
    report = build_first_step_tree_report(
        prompt="question", prompt_token_count=10, draft=draft, tokenizer=TinyTokenizer(),
        block_size=3, per_position_topk=8,
    )
    block = report["blocks"][0]
    assert math.isclose(block["joint_probability"], 0.05)
    assert math.isclose(block["token_probability_product"], 0.05)
    assert block["normalized_probability_over_returned_blocks"] == 1.0
    rendered = format_first_step_tree(report)
    assert "BLOCK #1" in rendered and "token[3]" in rendered and "joint_q=" in rendered
