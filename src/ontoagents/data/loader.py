"""Loading competency questions from the llm4ke dataset.

The loader is the leakage boundary: it reads ``cqs.yml`` but drops every ``rq`` entry, so
gold SPARQL can never reach the generation loop through a :class:`CompetencyQuestion`.
"""

import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import yaml

from ontoagents.domain import CompetencyQuestion

_PLACEHOLDER_IDS = {"", "<cqid>", "none", "null", "-"}
_WS_RE = re.compile(r"\s+")
_SLOT_RE = re.compile(r"\[\s*([^\]]*?)\s*\]")
_SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([?.,;:!])")


def normalise_question(text: str) -> str:
    """Collapse whitespace, trim spaces inside ``[slots]`` and before punctuation.

    Placeholder content is kept verbatim: ``[ this animal ]`` -> ``[this animal]``.
    """
    text = _WS_RE.sub(" ", text).strip()
    text = _SLOT_RE.sub(lambda m: f"[{m.group(1)}]", text)
    return _SPACE_BEFORE_PUNCT_RE.sub(r"\1", text)


class CQLoader(ABC):
    """Source of competency questions for an ontology."""

    @abstractmethod
    def load(self, ontology: str) -> list[CompetencyQuestion]:
        """Load the CQs of ``ontology`` in dataset order."""


class YamlCQLoader(CQLoader):
    """Reads ``<data_dir>/<ontology>/cqs/cqs.yml``."""

    def __init__(self, data_dir: Path) -> None:
        self._data_dir = data_dir

    def path_for(self, ontology: str) -> Path:
        """Location of the CQ file of ``ontology``."""
        return self._data_dir / ontology / "cqs" / "cqs.yml"

    def load(self, ontology: str) -> list[CompetencyQuestion]:
        """Load CQs, assigning ``<ontology>-<index>`` IDs where missing or duplicated."""
        path = self.path_for(ontology)
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        raw = (payload.get("ontology") or {}).get("cqs") or []
        return self.from_records(ontology, raw)

    @staticmethod
    def from_records(ontology: str, records: list[dict[str, Any]]) -> list[CompetencyQuestion]:
        """Convert raw YAML records into CQs (public for testing)."""
        cqs: list[CompetencyQuestion] = []
        seen: set[str] = set()
        for index, record in enumerate(records, start=1):
            question = normalise_question(str(record.get("question") or ""))
            if not question:
                continue
            raw_id = str(record.get("ID") or "").strip()
            cq_id = raw_id
            if raw_id.lower() in _PLACEHOLDER_IDS or raw_id in seen:
                cq_id = f"{ontology}-{index:03d}"
            seen.add(cq_id)
            group = record.get("group")
            cqs.append(
                CompetencyQuestion(
                    id=cq_id,
                    ontology=ontology,
                    question=question,
                    group=str(group).strip() if group else None,
                )
            )
        return cqs


def subsample(cqs: list[CompetencyQuestion], max_cqs: int | None) -> list[CompetencyQuestion]:
    """Deterministic, evenly spaced subset preserving dataset order (all if ``max_cqs`` is None)."""
    if max_cqs is None or max_cqs >= len(cqs):
        return list(cqs)
    if max_cqs <= 0:
        return []
    step = len(cqs) / max_cqs
    return [cqs[int(i * step)] for i in range(max_cqs)]
