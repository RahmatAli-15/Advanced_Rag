"""Reranker distillation (ADR 0015): no leakage from the evaluation set into training,
a train/dev split by source clause, and the shape of the distillation loss."""

from datetime import date

import pytest

from app.distill.data import TrainExample, held_out, split_by_group
from app.eval.dataset import GoldItem


def _gold(key: str, refs: list[str]) -> GoldItem:
    return GoldItem(
        id="x",
        question="q?",
        language="en",
        doc_source_key=key,
        doc_title="t",
        para_refs=refs,
        text_snippet="s",
        as_of=date(2026, 11, 18),
        origin="test",
    )


def test_evaluation_clauses_are_held_out() -> None:
    assert held_out([_gold("12957", ["70", "70(1)"]), _gold("12943", ["43"])]) == {
        ("12957", "70"),
        ("12957", "70(1)"),
        ("12943", "43"),
    }


def _ex(group: str, lang: str) -> TrainExample:
    return TrainExample(group, lang, f"{group}-{lang}", ["a", "b"], [0.9, 0.1])


def test_split_keeps_all_languages_of_a_clause_on_one_side() -> None:
    examples = [_ex(g, lang) for g in "abcdefghij" for lang in ("en", "hi", "hinglish")]
    train, dev = split_by_group(examples, dev_fraction=0.2, seed=1)
    train_groups, dev_groups = {e.group for e in train}, {e.group for e in dev}
    assert not train_groups & dev_groups  # no clause on both sides
    assert len(dev_groups) == 2 and len(train) + len(dev) == len(examples)
    assert split_by_group(examples, 0.2, seed=1) == (train, dev)  # reproducible


def test_saved_student_is_marked_to_output_probabilities() -> None:
    # The base student ships with an Identity output; the gate needs probabilities.
    pytest.importorskip("torch")
    from types import SimpleNamespace

    from app.distill.train import SIGMOID, mark_sigmoid_output

    config = SimpleNamespace(
        sbert_ce_default_activation_function="torch.nn.modules.linear.Identity"
    )
    mark_sigmoid_output(config)
    assert config.sentence_transformers == {"activation_fn": SIGMOID}
    assert not hasattr(config, "sbert_ce_default_activation_function")


def test_loss_prefers_the_teachers_ranking_and_scale() -> None:
    torch = pytest.importorskip("torch")
    from app.distill.train import distill_loss

    teacher = torch.tensor([0.95, 0.40, 0.02])
    logit = torch.log(teacher / (1 - teacher))
    same = distill_loss(logit, teacher, temperature=2.0)
    reversed_order = distill_loss(logit.flip(0), teacher, temperature=2.0)
    shifted_scale = distill_loss(logit + 4.0, teacher, temperature=2.0)
    assert same < reversed_order  # wrong ranking costs more
    assert same < shifted_scale  # same ranking but wrong scores also costs more
