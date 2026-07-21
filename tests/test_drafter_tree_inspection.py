import math

from block_spec.drafter_adapter import DraftCandidates
import torch

from block_spec.drafter_tree_inspection import (
    build_first_step_tree_report,
    build_multi_step_tree_report,
    format_first_step_tree,
    format_multi_step_tree,
)
from block_spec.tree_builder import AsymmetricTreeBuilder
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


class MultiStepDrafter:
    block_size = 3

    def propose(self, prefix):
        candidates = []
        for index in range(3):
            logps = (-0.1 - index, -0.2, -0.3)
            candidates.append(
                BlockCandidate((index + 1, index + 11, index + 21), logps, sum(logps))
            )
        return DraftCandidates(candidates, (0.9, 0.8, 0.7), 0.5)


def test_multi_step_report_has_branches_and_full_depth_path():
    drafter = MultiStepDrafter()
    tree = AsymmetricTreeBuilder(
        drafter, width_schedule=[2, 2, 1], parent_cap_schedule=[3, 2, 1],
        depth2_min_children_per_parent=1, max_tree_nodes=8, max_tree_tokens=21,
    ).build(torch.tensor([99]))
    report = build_multi_step_tree_report(
        prompt="question", prompt_token_count=1, tree=tree, tokenizer=TinyTokenizer(),
        block_size=3, per_position_topk=3, width_schedule=[2, 2, 1],
    )
    assert report["actual_widths"] == [2, 2, 1]
    assert report["tree_nodes_including_root"] == 6
    assert len(report["deepest_paths"]) == 1
    assert len(report["deepest_paths"][0]["blocks"]) == 3
    assert len(report["deepest_paths"][0]["all_token_ids"]) == 9
    rendered = format_multi_step_tree(report)
    assert "depth=1" in rendered and "depth=3" in rendered
    assert "FULL 3-BLOCK PATHS" in rendered
