"""Evaluation regimes, fusion strategies and train/test protocols.

Strategies (Section 3.1 of the paper)
-------------------------------------
``Strategy.FUSED`` (S1)
    All fingers of a subject captured at the same acquisition index are joined
    into a single composite template, so one template represents one subject.

``Strategy.INDEPENDENT`` (S2)
    Every finger sample is enrolled and probed on its own.

Protocols (Section 3.1)
-----------------------
``Protocol.P1`` 70% train / 30% test, ``Protocol.P2`` 50/50, ``Protocol.P3``
90/10.  When a percentage does not divide the number of samples per finger, the
training set is rounded *down* so at least one sample is always available for
testing.

Evaluation regimes (Section 3.1)
--------------------------------
``Evaluation.SESSION_SENSITIVE``
    Only used for the two-session FV-USM dataset: a pair is genuine when both
    the subject *and* the session match.

``Evaluation.SESSION_INDEPENDENT``
    Also FV-USM only: any same-subject pair is genuine.

``Evaluation.ONE_SESSION``
    MMCBNU-6000 and UTFVP, which are treated as single-session collections.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, Sequence, Tuple


class Strategy(Enum):
    """Template-construction strategy."""

    FUSED = "S1"
    INDEPENDENT = "S2"

    @classmethod
    def parse(cls, value: str) -> "Strategy":
        return _parse_enum(cls, value, {"s1": cls.FUSED, "1": cls.FUSED,
                                        "fused": cls.FUSED,
                                        "s2": cls.INDEPENDENT, "2": cls.INDEPENDENT,
                                        "independent": cls.INDEPENDENT})


class Protocol(Enum):
    """Train/test split protocol."""

    P1 = "P1"
    P2 = "P2"
    P3 = "P3"

    @classmethod
    def parse(cls, value: str) -> "Protocol":
        return _parse_enum(cls, value, {"p1": cls.P1, "1": cls.P1,
                                        "p2": cls.P2, "2": cls.P2,
                                        "p3": cls.P3, "3": cls.P3})

    @property
    def train_fraction(self) -> float:
        return {Protocol.P1: 0.7, Protocol.P2: 0.5, Protocol.P3: 0.9}[self]


class Evaluation(Enum):
    """How genuine pairs are defined."""

    SESSION_SENSITIVE = "session-sensitive"
    SESSION_INDEPENDENT = "session-independent"
    ONE_SESSION = "one-session"

    @classmethod
    def parse(cls, value: str) -> "Evaluation":
        key = value.strip().lower().replace("_", "-")
        return _parse_enum(cls, value, {
            "session-sensitive": cls.SESSION_SENSITIVE,
            "sensitive": cls.SESSION_SENSITIVE,
            "session-independent": cls.SESSION_INDEPENDENT,
            "independent": cls.SESSION_INDEPENDENT,
            "one-session": cls.ONE_SESSION,
            "single-session": cls.ONE_SESSION,
        }, key)


def _parse_enum(cls, value: str, aliases: Dict[str, object], key: str | None = None):
    key = key if key is not None else value.strip().lower()
    if key in aliases:
        return aliases[key]
    raise ValueError(
        f"unknown {cls.__name__} {value!r}; expected one of "
        + ", ".join(sorted({str(v.value) for v in cls}))
    )


def identity_fields(evaluation: Evaluation, strategy: Strategy) -> Tuple[str, ...]:
    """Sample attributes that jointly define an identity.

    Two samples are a genuine pair when they agree on all of these fields; a
    probe is classified correctly when its rank-1 candidate shares its identity.

    The defaults follow the definitions given in Section 3.1 of the paper.  Note
    that under both FV-USM regimes the finger is deliberately *not* part of the
    identity, whereas under the one-session regimes each finger of a subject is
    its own class unless the fingers have already been fused by Strategy 1.

    These are the definitions the *article text* gives.  They are not always
    what the experiments used -- see :func:`archive_identity_fields`, which is
    the default in this repository.
    """
    if evaluation is Evaluation.SESSION_SENSITIVE:
        return ("subject", "session")
    if evaluation is Evaluation.SESSION_INDEPENDENT:
        return ("subject",)
    if strategy is Strategy.FUSED:
        return ("subject",)
    return ("subject", "finger")


def archive_identity_fields(
    dataset: str,
    evaluation: Evaluation,
    strategy: Strategy,
    descriptor: str,
    reducer: str,
) -> Tuple[str, ...]:
    """Identity the experiments for this grid cell actually compared.

    This is the repository default, because it is what produced the published
    tables.

    The notebooks derived identities by splitting a label string on ``"_"``, and
    they did not all agree on which tokens to compare.  Reproducing the
    published tables therefore needs the convention of the specific notebook
    family, which this function reconstructs.

    Two of these conventions look unintended.  They are kept because they are
    what ran, and because changing them would silently detach this code from
    the published results:

    * ``mmcbnu_6000`` Strategy 2 -- the MMCBNU finger directories contain an
      underscore (``L_Fore``), so ``label.split("_")[1]`` yields the *hand*
      rather than the finger.  Identity collapses to subject + hand, giving 200
      classes instead of 600.
    * ``fv_usm`` session-sensitive Strategy 2 with raw-pixel 2DPCA -- the label
      begins with the session token, so the comparison of ``parts[0]`` and
      ``parts[-1]`` matches session + finger and never compares the subject.

    Combinations that no archived notebook covers (a single-radius descriptor
    together with a reducer, for instance) fall back to the convention of the
    closest family that does.
    """
    key = dataset.strip().lower().replace("-", "_")
    uses_reducer = reducer.strip().lower() not in {"none", "raw", ""}

    if key == "mmcbnu_6000":
        return ("subject",) if strategy is Strategy.FUSED else ("subject", "hand")

    if key == "utfvp":
        return ("subject",) if strategy is Strategy.FUSED else ("subject", "finger")

    if key != "fv_usm":
        raise ValueError(f"no archived identity convention for dataset {dataset!r}")

    if evaluation is Evaluation.SESSION_INDEPENDENT:
        if strategy is Strategy.FUSED:
            return ("subject",)
        return ("subject", "finger")

    # Session-sensitive FV-USM.
    if strategy is Strategy.FUSED:
        return ("subject", "session")
    if descriptor == "raw" and reducer.strip().lower() == "2dpca":
        return ("session", "finger")
    if not uses_reducer:
        return ("subject", "session")
    return ("subject", "finger", "session")


def split_indices(
    indices: Sequence[int], protocol: Protocol
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """Split acquisition indices into (train, test) according to ``protocol``.

    The first ``floor(fraction * n)`` acquisitions of a finger are used for
    training and the remainder for testing, which reproduces the fixed splits of
    the paper for datasets with a regular number of captures per finger.
    """
    ordered = tuple(indices)
    n_train = int(len(ordered) * protocol.train_fraction)
    n_train = max(1, min(n_train, len(ordered) - 1))
    return ordered[:n_train], ordered[n_train:]
