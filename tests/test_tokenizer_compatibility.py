import json

import pytest

from block_spec.tokenizer_compatibility import validate_tokenizer_compatibility


class Tok:
    bos_token_id = 1
    eos_token_id = 2
    pad_token_id = 0
    unk_token_id = 3
    def __init__(self, vocab): self.vocab = vocab
    def get_vocab(self): return self.vocab


def test_identical_passes_and_saves(tmp_path):
    target = tmp_path / "report.json"
    r = validate_tokenizer_compatibility(Tok({"a": 4}), Tok({"a": 4}), report_path=target)
    assert r.compatible and json.loads(target.read_text())["compatible"]


def test_mismatch_fails_strict():
    with pytest.raises(ValueError):
        validate_tokenizer_compatibility(Tok({"a": 4}), Tok({"a": 5}))

