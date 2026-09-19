"""Dataset readers for FV-USM, MMCBNU-6000 and UTFVP.

Each reader enumerates the ROI files of a dataset as :class:`Sample` records and
knows how to split them into a training and a test set for a given
:class:`~mrflbp.protocols.Protocol`.  Feature extraction and matching are kept
out of this module: readers only describe *which* image belongs to *which*
identity.

Expected directory layouts
--------------------------
FV-USM (ROIs ship with the dataset)::

    <root>/1st_session/extractedvein/vein001_1/01.jpg
    <root>/2nd_session/extractedvein/vein123_4/06.jpg

MMCBNU-6000 (ROIs ship with the dataset)::

    <root>/001/L_Fore/01.bmp ... <root>/100/R_Ring/10.bmp

UTFVP (ROIs must be produced first -- see ``scripts/prepare_utfvp_roi.py`` and
``scripts/augment_utfvp.py``)::

    <root>/0001/0001_1_1.png                  # original ROI
    <root>/0001/0001_1_1_1_Augmented.png      # augmentation 1 of capture 1
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from mrflbp.protocols import Protocol, split_indices

ORIGINAL = "orig"


@dataclass(frozen=True)
class Sample:
    """One ROI image together with the metadata that defines its identity."""

    path: Path
    subject: str
    finger: str
    session: str
    index: str

    def identity(self, fields: Sequence[str]) -> Tuple[str, ...]:
        """Identity key built from the requested metadata fields."""
        return tuple(getattr(self, field) for field in fields)

    @property
    def hand(self) -> str:
        """Hand the finger belongs to, for datasets that encode it.

        MMCBNU-6000 names its finger directories ``L_Fore`` ... ``R_Ring``, so
        the leading token is the hand.  Datasets whose finger identifiers carry
        no hand simply return the finger itself.
        """
        head, separator, _ = self.finger.partition("_")
        return head if separator else self.finger

    @property
    def group(self) -> Tuple[str, str, str]:
        """Key that gathers the fingers fused into one Strategy 1 template."""
        return (self.subject, self.session, self.index)


class FingerVeinDataset:
    """Base class for the three datasets used in the paper."""

    name: str = ""
    #: Canonical ROI size as (height, width) used by every experiment.
    image_size: Tuple[int, int] = (0, 0)
    #: Fingers per subject, in the order used when fusing Strategy 1 templates.
    fingers: Tuple[str, ...] = ()

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        if not self.root.is_dir():
            raise NotADirectoryError(f"dataset root does not exist: {self.root}")

    def split(self, protocol: Protocol) -> Tuple[List[Sample], List[Sample]]:
        """Return the (train, test) samples for ``protocol``."""
        raise NotImplementedError

    # -- helpers -----------------------------------------------------------
    def _subject_dirs(self) -> List[Path]:
        return sorted(p for p in self.root.iterdir() if p.is_dir())

    @staticmethod
    def _keep_existing(samples: Iterable[Sample]) -> List[Sample]:
        return [s for s in samples if s.path.is_file()]


class FVUSM(FingerVeinDataset):
    """FV-USM: 123 subjects x 4 fingers x 6 captures x 2 sessions = 5,904 ROIs."""

    name = "fv_usm"
    image_size = (100, 300)
    fingers = ("1", "2", "3", "4")

    SESSIONS = {"s1": "1st_session", "s2": "2nd_session"}
    SUBJECTS = 123
    CAPTURES = tuple(range(1, 7))

    def _samples(self, indices: Sequence[int]) -> List[Sample]:
        out: List[Sample] = []
        for subject in range(1, self.SUBJECTS + 1):
            subject_id = f"{subject:03d}"
            for session, folder in self.SESSIONS.items():
                for finger in self.fingers:
                    directory = (
                        self.root / folder / "extractedvein" / f"vein{subject_id}_{finger}"
                    )
                    for index in indices:
                        out.append(
                            Sample(
                                path=directory / f"{index:02d}.jpg",
                                subject=subject_id,
                                finger=finger,
                                session=session,
                                index=f"{index:02d}",
                            )
                        )
        return self._keep_existing(out)

    def split(self, protocol: Protocol) -> Tuple[List[Sample], List[Sample]]:
        train_idx, test_idx = split_indices(self.CAPTURES, protocol)
        return self._samples(train_idx), self._samples(test_idx)


class MMCBNU6000(FingerVeinDataset):
    """MMCBNU-6000: 100 subjects x 6 fingers x 10 captures = 6,000 ROIs."""

    name = "mmcbnu_6000"
    image_size = (60, 120)
    fingers = ("L_Fore", "L_Middle", "L_Ring", "R_Fore", "R_Middle", "R_Ring")

    CAPTURES = tuple(range(1, 11))

    def _samples(self, indices: Sequence[int]) -> List[Sample]:
        out: List[Sample] = []
        for subject_dir in self._subject_dirs():
            for finger in self.fingers:
                for index in indices:
                    out.append(
                        Sample(
                            path=subject_dir / finger / f"{index:02d}.bmp",
                            subject=subject_dir.name,
                            finger=finger,
                            session="s1",
                            index=f"{index:02d}",
                        )
                    )
        return self._keep_existing(out)

    def split(self, protocol: Protocol) -> Tuple[List[Sample], List[Sample]]:
        train_idx, test_idx = split_indices(self.CAPTURES, protocol)
        return self._samples(train_idx), self._samples(test_idx)


class UTFVP(FingerVeinDataset):
    """UTFVP: 60 subjects x 6 fingers x 4 captures, expanded to 16 by augmentation.

    The dataset ships without ROIs, so ``root`` must point at the *augmented ROI*
    tree produced by ``scripts/prepare_utfvp_roi.py`` followed by
    ``scripts/augment_utfvp.py`` (1,440 originals + 4,320 augmentations = 5,760
    images).

    Original ROIs are reserved for testing wherever the split allows it, and an
    augmentation of a capture never appears in both sets, so training and test
    images never overlap.
    """

    name = "utfvp"
    image_size = (60, 128)
    fingers = ("1", "2", "3", "4", "5", "6")

    CAPTURES = (1, 2, 3, 4)
    AUGMENTATIONS = (1, 2, 3)

    #: (capture, variant) pairs per protocol; ORIGINAL marks the un-augmented ROI.
    SPLITS: Dict[Protocol, Dict[str, Tuple[Tuple[int, object], ...]]] = {
        Protocol.P1: {
            "train": tuple((c, a) for c in (1, 2, 3) for a in (1, 2, 3))
            + ((4, 1), (4, 2)),
            "test": tuple((c, ORIGINAL) for c in (1, 2, 3, 4)) + ((4, 3),),
        },
        Protocol.P2: {
            "train": tuple((c, a) for c in (1, 2, 3, 4) for a in (1, 2)),
            "test": tuple((c, ORIGINAL) for c in (1, 2, 3, 4))
            + tuple((c, 3) for c in (1, 2, 3, 4)),
        },
        Protocol.P3: {
            "train": ((1, ORIGINAL), (2, ORIGINAL))
            + tuple((c, a) for c in (1, 2, 3, 4) for a in (1, 2, 3)),
            "test": ((3, ORIGINAL), (4, ORIGINAL)),
        },
    }

    def _filename(self, subject: str, finger: str, capture: int, variant: object) -> str:
        if variant == ORIGINAL:
            return f"{subject}_{finger}_{capture}.png"
        return f"{subject}_{finger}_{capture}_{variant}_Augmented.png"

    def _samples(self, spec: Sequence[Tuple[int, object]]) -> List[Sample]:
        out: List[Sample] = []
        for subject_dir in self._subject_dirs():
            subject = subject_dir.name
            for capture, variant in spec:
                for finger in self.fingers:
                    out.append(
                        Sample(
                            path=subject_dir
                            / self._filename(subject, finger, capture, variant),
                            subject=subject,
                            finger=finger,
                            session="s1",
                            index=f"{capture}_{variant}",
                        )
                    )
        return self._keep_existing(out)

    def split(self, protocol: Protocol) -> Tuple[List[Sample], List[Sample]]:
        spec = self.SPLITS[protocol]
        return self._samples(spec["train"]), self._samples(spec["test"])


DATASETS: Dict[str, type] = {
    FVUSM.name: FVUSM,
    MMCBNU6000.name: MMCBNU6000,
    UTFVP.name: UTFVP,
}


def build_dataset(name: str, root: Path | str) -> FingerVeinDataset:
    """Instantiate a dataset reader by name."""
    key = name.strip().lower().replace("-", "_")
    if key not in DATASETS:
        raise ValueError(
            f"unknown dataset {name!r}; expected one of {', '.join(sorted(DATASETS))}"
        )
    return DATASETS[key](root)


def group_templates(
    samples: Iterable[Sample],
) -> Dict[Tuple[str, str, str], List[Sample]]:
    """Group samples into Strategy 1 templates, keeping the finger order stable."""
    groups: Dict[Tuple[str, str, str], List[Sample]] = {}
    for sample in samples:
        groups.setdefault(sample.group, []).append(sample)
    for members in groups.values():
        members.sort(key=lambda s: s.finger)
    return groups
