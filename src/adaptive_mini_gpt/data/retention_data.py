"""Audited Retention and Independent Replay Corpora (M27 Phase 1).

Maintains strict separation between:
  1. EXPANDED_RETENTION_CORPUS: Protected out-of-sample evaluation holdout
     used strictly for measuring representation drift. NEVER seen during training.
  2. INDEPENDENT_REPLAY_CORPUS: Completely disjoint literature used exclusively
     for replay buffer mixing during continual training.

Guarantees 0.0% sentence or paragraph overlap to prevent artificial loss drift
and false retention claims.
"""

from __future__ import annotations

import re
import torch
import torch.nn as nn

from adaptive_mini_gpt.data.subword_tokenizer import BPETokenizer
from adaptive_mini_gpt.model.transformer import MiniGPT

# Protected Evaluation Holdout (Strictly Sealed - Never Used in Training)
EXPANDED_RETENTION_CORPUS: dict[str, str] = {
    "prose_original": (
        "Paris is the capital of France. William Shakespeare wrote Hamlet, Macbeth, and Othello in London. "
        "The speed of light in vacuum is approximately 300,000 km/s. "
        "In quantum physics, photons, bosons, and fermions govern fundamental interactions. "
        "Tokyo is the capital of Japan, and Rome is in Italy."
    ),
    "technical_science": (
        "Thermodynamics defines the relationships between heat, work, temperature, and energy. "
        "The first law states that energy cannot be created or destroyed, only transformed. "
        "Entropy measures molecular disorder and increases in all irreversible processes. "
        "DNA transcription and translation synthesize polypeptides from nucleotide sequences."
    ),
    "geography_history": (
        "The Nile River flows north through northeastern Africa into the Mediterranean Sea. "
        "The Roman Empire expanded across Europe, North Africa, and the Middle East before dividing. "
        "The Industrial Revolution transformed manufacturing from manual labor to machine-driven factories."
    ),
    "mathematics_prose": (
        "Prime numbers are integers greater than one divisible only by one and themselves. "
        "Euclidean geometry explores planes, angles, triangles, and parallel postulates. "
        "Differential calculus computes instantaneous rates of change along smooth curves."
    ),
}

# Independent Replay Training Corpus (Completely Disjoint from Evaluation Holdout)
INDEPENDENT_REPLAY_CORPUS: list[str] = [
    (
        "Ancient civil engineering produced monumental aqueducts that transported water across vast valleys. "
        "Roman builders perfected pozzolanic concrete, combining volcanic ash with lime to harden underwater. "
        "The arch structure distributed compressive forces downward into stone abutments without tensile failure."
    ),
    (
        "Cellular respiration converts biochemical energy from nutrients into adenosine triphosphate. "
        "Glycolysis breaks down glucose in the cytoplasm, producing pyruvate and nicotinamide adenine dinucleotide. "
        "The citric acid cycle operates within the mitochondrial matrix to generate high-energy electron carriers."
    ),
    (
        "Plate tectonics explains the dynamic evolution of Earth's crust and convective mantle currents. "
        "Subduction zones force dense oceanic plates beneath lighter continental lithosphere, melting magma. "
        "Transform faults generate lateral friction, releasing accumulated elastic strain through seismic waves."
    ),
    (
        "Classical electrodynamics describes charged particles interacting through electric and magnetic fields. "
        "Maxwell's equations demonstrate that oscillating electromagnetic fields propagate through space at the speed of light. "
        "Faraday's law shows that a changing magnetic flux induces an electromotive force around a closed conducting loop."
    ),
]


def verify_corpora_disjointness() -> float:
    """Return maximum sentence overlap between retention and replay corpora (must be 0.0)."""
    retention_sentences = set()
    for text in EXPANDED_RETENTION_CORPUS.values():
        sentences = [s.strip().lower() for s in re.split(r"[.!?]", text) if len(s.strip()) > 10]
        retention_sentences.update(sentences)

    replay_sentences = set()
    for text in INDEPENDENT_REPLAY_CORPUS:
        sentences = [s.strip().lower() for s in re.split(r"[.!?]", text) if len(s.strip()) > 10]
        replay_sentences.update(sentences)

    overlap = retention_sentences.intersection(replay_sentences)
    return len(overlap) / max(1, len(retention_sentences))


def evaluate_audited_retention(
    model: MiniGPT,
    tokenizer: BPETokenizer,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate cross-entropy loss on protected retention holdout (strictly disjoint from replay)."""
    model.eval()
    losses: dict[str, float] = {}
    with torch.no_grad():
        for name, text in EXPANDED_RETENTION_CORPUS.items():
            toks = tokenizer.encode(text)
            if len(toks) > model.config.context_length:
                toks = toks[: model.config.context_length]
            inp = torch.tensor([toks], dtype=torch.long, device=device)
            logits, _ = model(inp)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = inp[:, 1:].contiguous()
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, model.config.vocab_size),
                shift_labels.view(-1),
            )
            losses[name] = round(loss.item(), 4)

    losses["mean_loss"] = round(sum(losses.values()) / max(1, len(losses)), 4)
    return losses
