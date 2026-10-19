"""Data loading, deterministic splitting, and batch generation for Mini-GPT."""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

import torch
from torch import Tensor

# Lawful, public-domain text excerpt (Shakespeare's Sonnets) used as default offline corpus
DEFAULT_LAWFUL_CORPUS = """
From fairest creatures we desire increase,
That thereby beauty's rose might never die,
But as the riper should by time decease,
His tender heir might bear his memory:
But thou, contracted to thine own bright eyes,
Feed'st thy light's flame with self-substantial fuel,
Making a famine where abundance lies,
Thyself thy foe, to thy sweet self too cruel.
Thou that art now the world's fresh ornament
And only herald to the gaudy spring,
Within thine own bud buriest thy content
And, tender churl, mak'st waste in niggarding.
Pity the world, or else this glutton be,
To eat the world's due, by the grave and thee.

When forty winters shall besiege thy brow,
And dig deep trenches in thy beauty's field,
Thy youth's proud livery, so gazed on now,
Will be a tatter'd weed, of small worth held:
Then being ask'd where all thy beauty lies,
Where all the treasure of thy lusty days,
To say, within thine own deep-sunken eyes,
Were an all-eating shame and thriftless praise.
How much more praise deserved thy beauty's use,
If thou couldst answer 'This fair child of mine
Shall sum my count and make my old excuse,'
Proving his beauty by succession thine!
This were to be new made when thou art old,
And see thy blood warm when thou feel'st it cold.

Look in thy glass, and tell the face thou viewest
Now is the time that face should form another;
Whose fresh repair if now thou not renewest,
Thou dost beguile the world, unbless some mother.
For where is she so fair whose unear'd womb
Disdains the tillage of thy husbandry?
Or who is he so fond will be the tomb
Of his self-love, to stop posterity?
Thou art thy mother's glass, and she in thee
Calls back the lovely April of her prime:
So thou through windows of thine age shalt see
Despite of wrinkles this thy golden time.
But if thou live, remember'd not to be,
Die single, and thine image dies with thee.

Unthrifty loveliness, why dost thou spend
Upon thyself thy beauty's legacy?
Nature's bequest gives nothing but doth lend,
And being frank she lends to those are free.
Then, beauteous niggard, why dost thou abuse
The bounteous largess given thee to give?
Profitless usurer, why dost thou use
So great a sum of sums, yet canst not live?
For having traffic with thyself alone,
Thou of thyself thy sweet self dost deceive.
Then how, when nature calls thee to be gone,
What acceptable audit canst thou leave?
Thy unused beauty must be tomb'd with thee,
Which, used, lives th' executor to be.

Those hours, that with gentle work did frame
The lovely gaze where every eye doth dwell,
Will play the tyrants to the very same
And that unfair which fairly doth excel;
For never-resting time leads summer on
To hideous winter and confounds him there;
Sap cheque'd with frost and lusty leaves quite gone,
Beauty o'ersnow'd and bareness every where:
Then, were not summer's distillation left,
A liquid prisoner pent in walls of glass,
Beauty's effect with beauty were bereft,
Nor it nor no remembrance what it was:
But flowers distill'd though they with winter meet,
Leese but their show; their substance still lives sweet.

Then let not winter's ragged hand deface
In thee thy summer, ere thou be distill'd:
Make sweet some vial; treasure thou some place
With beauty's treasure, ere it be self-kill'd.
That use is not forbidden usury,
Which happies those that pay the willing loan;
That's for thyself to breed another thee,
Or ten times happier, be it ten for one;
Ten times thyself were happier than thou art,
If ten of thine ten times refigur'd thee:
Then what could death do, if thou shouldst depart,
Leaving thee living in posterity?
Be not self-will'd, for thou art much too fair
To be death's conquest and make worms thine heir.
"""

SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)


def compute_fingerprint(data: str | bytes | Tensor) -> str:
    """Compute a deterministic SHA-256 fingerprint for data."""
    if isinstance(data, str):
        content = data.encode("utf-8")
    elif isinstance(data, bytes):
        content = data
    elif isinstance(data, Tensor):
        content = data.cpu().numpy().tobytes()
    else:
        content = str(data).encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def load_corpus(path: str | Path | None = None) -> tuple[str, str]:
    """Load text corpus from path, cache directory, or lawful default corpus.

    Returns:
        Tuple of (text_content, data_fingerprint).
    """
    if path is not None and Path(path).is_file():
        text = Path(path).read_text(encoding="utf-8")
        return text, compute_fingerprint(text)

    # Check local cached dataset in data/
    cache_path = Path("data/tinyshakespeare.txt")
    if cache_path.is_file():
        text = cache_path.read_text(encoding="utf-8")
        return text, compute_fingerprint(text)

    # Attempt download if path is None
    if path is None:
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(SHAKESPEARE_URL, cache_path)
            if cache_path.is_file() and cache_path.stat().st_size > 0:
                text = cache_path.read_text(encoding="utf-8")
                return text, compute_fingerprint(text)
        except Exception:
            # Fall back cleanly to offline lawful corpus
            pass

    text = DEFAULT_LAWFUL_CORPUS.strip()
    return text, compute_fingerprint(text)


def is_holdout_arithmetic_pair(op_type: int, a: int, b: int) -> bool:
    """Deterministic hash partition: 10% of operand pairs are strictly reserved for holdout eval."""
    return ((a * 131 + b * 17 + op_type) % 10) == 0


def generate_arithmetic_corpus(num_samples: int = 2000, seed: int = 42) -> tuple[str, str]:
    """Generate a deterministic procedural arithmetic & parity corpus (disjoint from holdout)."""
    g = torch.Generator().manual_seed(seed)
    lines: list[str] = []
    while len(lines) < num_samples:
        op_type = int(torch.randint(0, 3, (1,), generator=g).item())
        a = int(torch.randint(0, 100, (1,), generator=g).item())
        b = int(torch.randint(0, 100, (1,), generator=g).item())
        if op_type == 1:
            a, b = max(a, b), min(a, b)
        elif op_type == 2:
            b = 0
        if is_holdout_arithmetic_pair(op_type, a, b):
            continue
        if op_type == 0:
            lines.append(f"ADD {a}+{b}={a + b}")
        elif op_type == 1:
            lines.append(f"SUB {a}-{b}={a - b}")
        else:
            parity = "EVEN" if (a % 2 == 0) else "ODD"
            lines.append(f"PAR {a}={parity}")
    text = "\n".join(lines) + "\n"
    return text, compute_fingerprint(text)


def mask_targets_before_equals(
    x: Tensor,
    y: Tensor,
    equals_token_id: int,
    newline_token_id: int | None = None,
) -> Tensor:
    """Mask targets so that loss is only computed on answer tokens (after '=' up to newline)."""
    y_masked = y.clone()
    is_eq = x == equals_token_id
    is_nl = (x == newline_token_id) if newline_token_id is not None else torch.zeros_like(is_eq)

    b_size, t_size = x.size()
    in_answer = torch.zeros((b_size,), dtype=torch.bool, device=x.device)
    mask = torch.zeros((b_size, t_size), dtype=torch.bool, device=x.device)
    for t in range(t_size):
        in_answer = torch.where(is_eq[:, t], torch.ones_like(in_answer), in_answer)
        mask[:, t] = in_answer
        in_answer = torch.where(is_nl[:, t], torch.zeros_like(in_answer), in_answer)

    y_masked[~mask] = -1
    return y_masked


class TokenDataset:
    """Dataset holding token tensor for autoregressive sequence modeling."""

    def __init__(
        self,
        tokens: Tensor,
        context_length: int,
        mask_prompt_before_equals: bool = False,
        equals_token_id: int | None = None,
        newline_token_id: int | None = None,
    ) -> None:
        if tokens.dim() != 1:
            tokens = tokens.view(-1)
        if len(tokens) <= context_length:
            raise ValueError(
                f"Dataset size ({len(tokens)} tokens) must be strictly greater than "
                f"context length ({context_length}) to form (x, y) target pairs."
            )
        self.tokens = tokens
        self.context_length = context_length
        self.mask_prompt_before_equals = mask_prompt_before_equals
        self.equals_token_id = equals_token_id
        self.newline_token_id = newline_token_id

    def __len__(self) -> int:
        return len(self.tokens) - self.context_length

    def _apply_mask(self, x: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
        if self.mask_prompt_before_equals and self.equals_token_id is not None:
            y = mask_targets_before_equals(x, y, self.equals_token_id, self.newline_token_id)
        return x, y

    def get_batch(
        self,
        batch_size: int,
        device: torch.device | str = "cpu",
        generator: torch.Generator | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Sample a batch of (x, y) sequences randomly with replacement."""
        max_start = len(self.tokens) - self.context_length
        ix = torch.randint(0, max_start, (batch_size,), generator=generator)
        x = torch.stack([self.tokens[i : i + self.context_length] for i in ix]).to(device)
        y = torch.stack([self.tokens[i + 1 : i + self.context_length + 1] for i in ix]).to(device)
        return self._apply_mask(x, y)

    def get_sequential_batch(
        self,
        start_idx: int,
        batch_size: int,
        device: torch.device | str = "cpu",
    ) -> tuple[Tensor, Tensor, int]:
        """Get a deterministic sequential batch starting from start_idx with wrapping.

        Returns:
            Tuple of (x, y, next_start_idx).
        """
        max_start = len(self.tokens) - self.context_length
        ix = [(start_idx + i * self.context_length) % max_start for i in range(batch_size)]
        x = torch.stack([self.tokens[i : i + self.context_length] for i in ix]).to(device)
        y = torch.stack([self.tokens[i + 1 : i + self.context_length + 1] for i in ix]).to(device)
        next_start = (start_idx + batch_size * self.context_length) % max_start
        x, y = self._apply_mask(x, y)
        return x, y, next_start


def split_data(
    tokens: Tensor,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> tuple[Tensor, Tensor, Tensor]:
    """Deterministically split tokens into train, validation, and test sets."""
    n = len(tokens)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_tokens = tokens[:n_train]
    val_tokens = tokens[n_train : n_train + n_val]
    test_tokens = tokens[n_train + n_val :]

    return train_tokens, val_tokens, test_tokens
