"""Inventory of the ontologies in the dataset (``data/dm-rdf.yml`` plus the folder layout)."""

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict

ONTOLOGY_SUFFIXES = frozenset({".owl", ".ttl", ".rdf", ".n3", ".nt", ".jsonld"})
QUERY_SUFFIXES = frozenset({".rq", ".sparql"})


class CatalogEntry(BaseModel):
    """One ontology project of the dataset."""

    model_config = ConfigDict(frozen=True)

    name: str
    directory: Path
    group: str | None = None
    dm_files: tuple[Path, ...] = ()
    has_cqs: bool = False

    @property
    def has_reference(self) -> bool:
        """Whether a local reference ontology exists (False e.g. for SILKNOW)."""
        return bool(self.dm_files)

    @property
    def description_path(self) -> Path:
        """``description.txt`` location (may not exist)."""
        return self.directory / "description.txt"

    def description(self) -> str | None:
        """The project description, if the dataset has one."""
        path = self.description_path
        if not path.exists():
            return None
        return path.read_text(encoding="utf-8").strip() or None

    @property
    def base_iri(self) -> str:
        """Namespace minted for generated terms; never equal to the reference namespace."""
        slug = "".join(ch if ch.isalnum() else "-" for ch in self.name.lower()).strip("-")
        return f"http://ontoagents.example.org/{slug}#"


class OntologyCatalog:
    """Lists the ontologies under ``data_dir``.

    Folder names are canonical; ``dm-rdf.yml`` contributes the project group, matched
    case-insensitively (the YAML says ``NORIA-O`` while the folder is ``noria-o``).
    """

    def __init__(self, data_dir: Path) -> None:
        self._data_dir = data_dir
        self._entries = self._scan()

    @property
    def data_dir(self) -> Path:
        """Root of the dataset."""
        return self._data_dir

    def _scan(self) -> dict[str, CatalogEntry]:
        groups: dict[str, str] = {}
        index = self._data_dir / "dm-rdf.yml"
        if index.exists():
            payload = yaml.safe_load(index.read_text(encoding="utf-8")) or {}
            for item in payload.get("ontologies") or []:
                groups[str(item["name"]).lower()] = str(item.get("group") or "")
        entries: dict[str, CatalogEntry] = {}
        for directory in sorted(p for p in self._data_dir.iterdir() if p.is_dir()):
            cqs = directory / "cqs" / "cqs.yml"
            dm = directory / "dm"
            if not cqs.exists() and not dm.exists():
                continue
            dm_files = (
                tuple(
                    sorted(
                        p for p in dm.rglob("*") if p.is_file() and p.suffix in ONTOLOGY_SUFFIXES
                    )
                )
                if dm.exists()
                else ()
            )
            entries[directory.name] = CatalogEntry(
                name=directory.name,
                directory=directory,
                group=groups.get(directory.name.lower()) or None,
                dm_files=dm_files,
                has_cqs=cqs.exists(),
            )
        return entries

    def names(self) -> list[str]:
        """All ontology names with competency questions."""
        return [n for n, e in self._entries.items() if e.has_cqs]

    def get(self, name: str) -> CatalogEntry:
        """Look up an entry (case-insensitive)."""
        if name in self._entries:
            return self._entries[name]
        for key, entry in self._entries.items():
            if key.lower() == name.lower():
                return entry
        raise KeyError(f"unknown ontology {name!r}; known: {', '.join(self._entries)}")

    def select(self, names: list[str] | str) -> list[CatalogEntry]:
        """Resolve ``"all"``, ``"with_reference"`` or an explicit list of names."""
        if names == "all":
            return [self._entries[n] for n in self.names()]
        if names == "with_reference":
            return [self._entries[n] for n in self.names() if self._entries[n].has_reference]
        if isinstance(names, str):
            names = [names]
        return [self.get(n) for n in names]

    def shared_files(self) -> tuple[Path, ...]:
        """Files under ``data/shared`` (e.g. ``time.owl``) used to resolve imports."""
        shared = self._data_dir / "shared"
        if not shared.exists():
            return ()
        return tuple(sorted(p for p in shared.iterdir() if p.suffix in ONTOLOGY_SUFFIXES))
