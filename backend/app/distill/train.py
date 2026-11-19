"""Train the student reranker to imitate the teacher (knowledge distillation, ADR 0015).

A plain PyTorch loop, on purpose: every piece is visible.

    for each question (30 candidate clauses, teacher probability for each):
        student logits  = student(question, clause) for the 30 clauses
        rank loss       = KL( softmax(teacher logits / T) || softmax(student logits / T) ) * T^2
        score loss      = binary cross-entropy( sigmoid(student logits), teacher probabilities )
        loss            = rank loss + score loss

Two losses because the score is used twice in production: to rank clauses, and to decide
"not in the RBI rules" (top score below 0.05). Matching only the ranking would leave the
student's scores on a different scale and break that gate.

Training details: AdamW, linear warm-up then decay, fp16 autocast with a gradient
scaler, gradient clipping, gradient accumulation (one question per forward pass keeps a
4 GB GPU happy), and early stopping on the dev loss, keeping the best checkpoint.
"""

import logging
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.amp.grad_scaler import GradScaler
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from app.distill.data import TrainExample

log = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    student: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    epochs: int = 3
    lr: float = 3e-5
    warmup: float = 0.1
    accumulate: int = 4  # questions per optimiser step
    max_length: int = 512  # same as the production reranker
    temperature: float = 2.0
    patience: int = 1  # epochs without dev improvement before stopping
    seed: int = 13


@dataclass
class EpochLog:
    epoch: int
    train_loss: float
    dev_loss: float
    dev_top1_agreement: float  # student's best clause == teacher's best clause
    dev_mean_abs_prob_error: float  # |student prob - teacher prob|, averaged
    seconds: float


@dataclass
class TrainReport:
    config: TrainConfig
    train_questions: int
    dev_questions: int
    epochs: list[EpochLog] = field(default_factory=list)
    best_epoch: int = 0


SIGMOID = "torch.nn.modules.activation.Sigmoid"


def mark_sigmoid_output(config: Any) -> None:
    """Make sentence-transformers apply a sigmoid at inference, so the saved student
    returns probabilities like the teacher (the relevance gate compares them to 0.05).
    The base student was published with an Identity output, and save_pretrained would
    otherwise copy that over: the ranking is the same, the scores are not."""
    config.sentence_transformers = {"activation_fn": SIGMOID}
    if hasattr(config, "sbert_ce_default_activation_function"):
        del config.sbert_ce_default_activation_function


def _logit(p: torch.Tensor) -> torch.Tensor:
    p = p.clamp(1e-6, 1 - 1e-6)
    return torch.log(p / (1 - p))


def distill_loss(
    student_logits: torch.Tensor, teacher_probs: torch.Tensor, temperature: float
) -> torch.Tensor:
    """Rank like the teacher (listwise KL) and score like it (pointwise BCE)."""
    t = temperature
    rank = F.kl_div(
        F.log_softmax(student_logits / t, dim=-1),
        F.softmax(_logit(teacher_probs) / t, dim=-1),
        reduction="sum",
    ) * (t * t)
    score = F.binary_cross_entropy_with_logits(student_logits, teacher_probs)
    return rank + score


def train(
    train_set: list[TrainExample],
    dev_set: list[TrainExample],
    out_dir: Path,
    cfg: TrainConfig,
) -> TrainReport:
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(cfg.student)
    model = AutoModelForSequenceClassification.from_pretrained(cfg.student, num_labels=1).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.01)
    steps = math.ceil(len(train_set) / cfg.accumulate) * cfg.epochs
    sched = get_linear_schedule_with_warmup(opt, int(steps * cfg.warmup), steps)
    scaler = GradScaler(enabled=device == "cuda")
    report = TrainReport(cfg, len(train_set), len(dev_set))

    def forward(ex: TrainExample) -> tuple[torch.Tensor, torch.Tensor]:
        enc = tok(
            [ex.query] * len(ex.passages),
            ex.passages,
            truncation="only_second",
            max_length=cfg.max_length,
            padding=True,
            return_tensors="pt",
        ).to(device)
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=device == "cuda"):
            logits = model(**enc).logits.squeeze(-1)
        return logits.float(), torch.tensor(ex.teacher, device=device)

    def evaluate() -> tuple[float, float, float]:
        model.eval()
        losses, agree, abs_err = [], 0, []
        with torch.no_grad():
            for ex in dev_set:
                s, t = forward(ex)
                losses.append(distill_loss(s, t, cfg.temperature).item())
                agree += int(s.argmax().item() == t.argmax().item())
                abs_err.append((torch.sigmoid(s) - t).abs().mean().item())
        model.train()
        return sum(losses) / len(losses), agree / len(dev_set), sum(abs_err) / len(abs_err)

    best, stale = float("inf"), 0
    model.train()
    for epoch in range(1, cfg.epochs + 1):
        t0, total = time.perf_counter(), 0.0
        order = list(train_set)
        random.shuffle(order)
        opt.zero_grad()
        for i, ex in enumerate(order, start=1):
            s, t = forward(ex)
            loss = distill_loss(s, t, cfg.temperature) / cfg.accumulate
            scaler.scale(loss).backward()
            total += loss.item() * cfg.accumulate
            if i % cfg.accumulate == 0 or i == len(order):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                sched.step()
                opt.zero_grad()
            if i % 200 == 0:
                log.info("epoch %d: %d/%d questions, loss %.4f", epoch, i, len(order), total / i)
        dev_loss, agree, err = evaluate()
        report.epochs.append(
            EpochLog(epoch, total / len(order), dev_loss, agree, err, time.perf_counter() - t0)
        )
        log.info("epoch %d: dev loss %.4f, top-1 agreement %.3f", epoch, dev_loss, agree)
        if dev_loss < best:
            best, stale, report.best_epoch = dev_loss, 0, epoch
            mark_sigmoid_output(model.config)
            model.save_pretrained(out_dir)  # keep the best checkpoint only
            tok.save_pretrained(out_dir)
        else:
            stale += 1
            if stale > cfg.patience:
                log.info("early stopping after epoch %d", epoch)
                break
    return report


def baseline_dev_metrics(
    dev_set: list[TrainExample], cfg: TrainConfig
) -> tuple[float, float, float]:
    """The same dev metrics for the student BEFORE training, for comparison."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(cfg.student)
    model = AutoModelForSequenceClassification.from_pretrained(cfg.student, num_labels=1).to(device)
    model.eval()
    losses, agree, abs_err = [], 0, []
    with torch.no_grad():
        for ex in dev_set:
            enc = tok(
                [ex.query] * len(ex.passages),
                ex.passages,
                truncation="only_second",
                max_length=cfg.max_length,
                padding=True,
                return_tensors="pt",
            ).to(device)
            s = model(**enc).logits.squeeze(-1).float()
            t = torch.tensor(ex.teacher, device=device)
            losses.append(distill_loss(s, t, cfg.temperature).item())
            agree += int(s.argmax().item() == t.argmax().item())
            abs_err.append((torch.sigmoid(s) - t).abs().mean().item())
    return sum(losses) / len(losses), agree / len(dev_set), sum(abs_err) / len(abs_err)
