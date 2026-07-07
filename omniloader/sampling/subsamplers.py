"""Per-dataset index draws with configurable reuse and exhaustion policies.

An :class:`IndexPool` produces the local indices to visit within a single
dataset for a given epoch. It captures three orthogonal choices:

* **replacement** — whether a draw may repeat an index within one epoch;
* **exhaustion policy** — for without-replacement draws, whether each epoch is a
  fresh random subsample (``"fresh"``) or whether the pool shuffles through the
  entire dataset before any index is reused across epochs (``"exhaust"``);
* **weights** — optional per-index weights for skewed sampling (e.g. class
  balancing within a dataset).

Draws are fully determined by a base ``seed`` and the epoch number, so runs are
reproducible.
"""

from __future__ import annotations

import enum
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from collections.abc import Sequence


class ExhaustionPolicy(enum.Enum):
    """How without-replacement draws relate across epochs.

    Attributes:
        FRESH: Every epoch independently samples without replacement from the
            full dataset; nothing is remembered between epochs.
        EXHAUST: Maintain a running shuffled ordering of the whole dataset and
            consume it across epochs, reshuffling only once every index has been
            used. This guarantees uniform coverage before any reuse.

    """

    FRESH = "fresh"
    EXHAUST = "exhaust"


class IndexPool:
    """Draw local indices for one dataset, honouring reuse and exhaustion policy.

    Args:
        size: Number of samples in the dataset.
        replacement: When ``True``, indices may repeat within an epoch and the
            exhaustion policy is ignored. Defaults to ``False``.
        policy: The :class:`ExhaustionPolicy` for without-replacement draws.
        seed: Base seed; combined with the epoch to make draws reproducible.
        weights: Optional per-index sampling weights of length ``size`` (e.g.
            class-balancing weights). When set, draws are weighted via
            ``torch.multinomial`` and the exhaustion policy is bypassed.

    Raises:
        ValueError: If ``size`` is not positive or ``weights`` has a bad length.

    """

    def __init__(
        self,
        size: int,
        replacement: bool = False,
        policy: ExhaustionPolicy = ExhaustionPolicy.FRESH,
        seed: int = 0,
        weights: Sequence[float] | None = None,
    ) -> None:
        if size <= 0:
            raise ValueError(f"IndexPool size must be positive, got {size}")
        if weights is not None and len(weights) != size:
            raise ValueError(f"weights must have length {size}, got {len(weights)}")
        self.size = size
        self.replacement = replacement
        self.policy = policy
        self.seed = seed
        self.weights = (
            torch.as_tensor(weights, dtype=torch.float32) if weights is not None else None
        )
        self._buffer: list[int] = []
        self._reshuffles = 0

    def _generator(self, epoch: int, salt: int = 0) -> torch.Generator:
        """Build a deterministic generator for an epoch.

        Args:
            epoch: The epoch number.
            salt: Extra offset to decorrelate successive reshuffles.

        Returns:
            A seeded :class:`torch.Generator`.

        """
        gen = torch.Generator()
        gen.manual_seed(self.seed + epoch * 1_000_003 + salt)
        return gen

    def draw(self, count: int, epoch: int) -> list[int]:
        """Return ``count`` local indices to visit this epoch.

        Args:
            count: Number of indices requested. May exceed ``size`` when
                ``replacement`` is ``True`` or under the ``EXHAUST`` policy
                (which wraps across reshuffles).
            epoch: The epoch number, used to seed the draw.

        Returns:
            A list of ``count`` indices in ``[0, size)``.

        """
        if count <= 0:
            return []

        if self.weights is not None:
            # Weighted draw (e.g. class-balanced); with replacement when count
            # exceeds the number of non-zero-weight indices.
            gen = self._generator(epoch)
            replacement = self.replacement or count > int((self.weights > 0).sum().item())
            return torch.multinomial(
                self.weights, count, replacement=replacement, generator=gen
            ).tolist()

        if self.replacement:
            gen = self._generator(epoch)
            return torch.randint(0, self.size, (count,), generator=gen).tolist()

        if self.policy is ExhaustionPolicy.FRESH:
            gen = self._generator(epoch)
            perm = torch.randperm(self.size, generator=gen).tolist()
            if count <= self.size:
                return perm[:count]
            # Requested more than available: tile whole reshuffles then top up.
            out = list(perm)
            while len(out) < count:
                gen = self._generator(epoch, salt=len(out))
                out.extend(torch.randperm(self.size, generator=gen).tolist())
            return out[:count]

        # EXHAUST: consume a persistent buffer, reshuffling only when empty.
        out: list[int] = []
        while len(out) < count:
            if not self._buffer:
                gen = self._generator(self._reshuffles, salt=7)
                self._buffer = torch.randperm(self.size, generator=gen).tolist()
                self._reshuffles += 1
            take = min(count - len(out), len(self._buffer))
            out.extend(self._buffer[:take])
            self._buffer = self._buffer[take:]
        return out
