"""Aligning generated terms to reference terms (IRIs never match, so names must)."""

import re
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from rapidfuzz import fuzz
from rapidfuzz.process import cdist
from scipy.optimize import linear_sum_assignment

from ontoagents.domain import local_name

_CAMEL_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
#: CIDOC-CRM / FRBRoo / DOREMUS style codes: E21_Person, P14_carried_out_by, U13_has_casting.
_CODE_PREFIX_RE = re.compile(r"^[A-Z]{1,3}\d+[a-z]?_")
_PROPERTY_STOPWORDS = frozenset({"has", "is", "of", "by", "the", "a", "an", "to", "in", "was"})


def _lemma(token: str) -> str:
    """Tiny rule-based English lemmatiser (plural nouns only); deterministic and dependency-free."""
    if len(token) <= 3:
        return token
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith(("sses", "xes", "ches", "shes")):
        return token[:-2]
    if token.endswith("s") and not token.endswith(("ss", "us", "is", "os")):
        return token[:-1]
    return token


def normalise(name: str, drop_stopwords: bool = False) -> str:
    """``hasPart`` -> ``has part``; ``E21_Person`` -> ``person``; ``works`` -> ``work``."""
    name = _CODE_PREFIX_RE.sub("", name.strip())
    name = _CAMEL_RE.sub(" ", name)
    tokens = [t for t in _NON_ALNUM_RE.split(name.lower()) if t]
    if drop_stopwords:
        kept = [t for t in tokens if t not in _PROPERTY_STOPWORDS]
        tokens = kept or tokens
    return " ".join(_lemma(t) for t in tokens)


@dataclass(frozen=True)
class Term:
    """A term to align: IRI plus optional label."""

    iri: str
    label: str | None = None

    def variants(self, is_property: bool) -> tuple[str, ...]:
        """Normalised name variants (local name, label, and stop-word-free forms for properties)."""
        raw = [local_name(self.iri)]
        if self.label:
            raw.append(self.label)
        out: list[str] = []
        for r in raw:
            out.append(normalise(r))
            if is_property:
                out.append(normalise(r, drop_stopwords=True))
        return tuple(dict.fromkeys(v for v in out if v))


@dataclass(frozen=True)
class Match:
    """A one-to-one correspondence."""

    generated: str
    reference: str
    score: float


@dataclass(frozen=True)
class Alignment:
    """All candidate matches from the assignment, before thresholding."""

    matches: tuple[Match, ...]

    def at(self, threshold: float) -> tuple[Match, ...]:
        """Matches with ``score >= threshold``."""
        return tuple(m for m in self.matches if m.score >= threshold)

    def mapping(self, threshold: float) -> dict[str, str]:
        """Generated IRI -> reference IRI at ``threshold``."""
        return {m.generated: m.reference for m in self.at(threshold)}

    def inverse(self, threshold: float) -> dict[str, str]:
        """Reference IRI -> generated IRI at ``threshold``."""
        return {m.reference: m.generated for m in self.at(threshold)}


class Aligner(ABC):
    """Strategy for producing one-to-one matches."""

    @abstractmethod
    def align(
        self, generated: Sequence[Term], reference: Sequence[Term], is_property: bool = False
    ) -> Alignment:
        """Align two term lists."""


class LexicalAligner(Aligner):
    """Normalised local names and labels, string similarity, Hungarian assignment.

    Similarity of two terms is the maximum ``fuzz.ratio`` over their name variants, in
    [0, 1]. The assignment maximises total similarity; :meth:`Alignment.at` applies the
    threshold afterwards so sensitivity curves come for free.
    """

    def __init__(self, min_score: float = 0.5) -> None:
        self._min_score = min_score

    def similarity_matrix(
        self, generated: Sequence[Term], reference: Sequence[Term], is_property: bool
    ) -> np.ndarray:
        """Pairwise term similarity in [0, 1]."""
        gen_vars = [t.variants(is_property) for t in generated]
        ref_vars = [t.variants(is_property) for t in reference]
        gen_flat = [v for vs in gen_vars for v in vs]
        ref_flat = [v for vs in ref_vars for v in vs]
        if not gen_flat or not ref_flat:
            return np.zeros((len(generated), len(reference)))
        flat = cdist(gen_flat, ref_flat, scorer=fuzz.ratio, workers=-1) / 100.0
        gen_owner = np.repeat(np.arange(len(generated)), [len(v) for v in gen_vars])
        ref_owner = np.repeat(np.arange(len(reference)), [len(v) for v in ref_vars])
        by_gen = np.full((len(generated), flat.shape[1]), 0.0)
        np.maximum.at(by_gen, gen_owner, flat)
        out = np.full((len(generated), len(reference)), 0.0)
        np.maximum.at(out.T, ref_owner, by_gen.T)
        return out

    def align(
        self, generated: Sequence[Term], reference: Sequence[Term], is_property: bool = False
    ) -> Alignment:
        """Hungarian assignment over the similarity matrix."""
        if not generated or not reference:
            return Alignment(matches=())
        sim = self.similarity_matrix(generated, reference, is_property)
        rows, cols = linear_sum_assignment(sim, maximize=True)
        matches = tuple(
            Match(generated=generated[r].iri, reference=reference[c].iri, score=float(sim[r, c]))
            for r, c in zip(rows, cols, strict=True)
            if sim[r, c] >= self._min_score
        )
        return Alignment(matches=tuple(sorted(matches, key=lambda m: -m.score)))
