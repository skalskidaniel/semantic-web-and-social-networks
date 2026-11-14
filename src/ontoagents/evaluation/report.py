"""Aggregate result JSONs into a Markdown report."""

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

HEADLINE_METRICS: tuple[tuple[str, str], ...] = (
    ("class_f1", "Class F1"),
    ("class_recall_cq_relevant", "Class R (CQ-rel.)"),
    ("property_f1", "Prop. F1"),
    ("property_recall_cq_relevant", "Prop. R (CQ-rel.)"),
    ("subclass_agreement", "SubClass agr."),
    ("posthoc_covered_rate", "CQ coverage"),
    ("gold_query_pass_rate", "Gold SPARQL"),
    ("final_consistent", "Consistent"),
    ("n_classes", "#Cls"),
    ("n_properties", "#Props"),
    ("llm_calls", "Calls"),
    ("total_tokens", "Tokens"),
)


def load_results(results_dir: Path) -> list[dict[str, Any]]:
    """All per-ontology result files below ``results_dir``."""
    out = []
    for path in sorted(results_dir.glob("*/*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if "ontology" in data:
            data["experiment"] = path.parent.name
            out.append(data)
    return out


def _fmt(value: Any) -> str:
    if value is None:
        return "–"
    if isinstance(value, float):
        return f"{value:.3f}" if value <= 1.0 else f"{value:,.0f}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def render_markdown(results: list[dict[str, Any]]) -> str:
    """Per-experiment tables plus a macro-average comparison."""
    by_exp: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in results:
        by_exp[r["experiment"]].append(r)
    lines = ["# Results", ""]
    header = "| Ontology | " + " | ".join(label for _, label in HEADLINE_METRICS) + " |"
    sep = "|" + "---|" * (len(HEADLINE_METRICS) + 1)
    macro: dict[str, dict[str, float]] = {}
    for exp, rows in sorted(by_exp.items()):
        system = rows[0].get("system", "?")
        lines += [f"## {exp} (`{system}`)", "", header, sep]
        sums: dict[str, list[float]] = defaultdict(list)
        for r in sorted(rows, key=lambda r: r["ontology"]):
            if "error" in r:
                lines.append(f"| {r['ontology']} | ERROR: {r['error'].splitlines()[0][:80]} |")
                continue
            m = r.get("metrics", {})
            lines.append(
                f"| {r['ontology']} | "
                + " | ".join(_fmt(m.get(k)) for k, _ in HEADLINE_METRICS)
                + " |"
            )
            for k, _ in HEADLINE_METRICS:
                if isinstance(m.get(k), int | float):
                    sums[k].append(float(m[k]))
        macro[exp] = {k: sum(v) / len(v) for k, v in sums.items() if v}
        lines.append("")
    if len(macro) > 1:
        lines += [
            "## Macro averages",
            "",
            "| Experiment | " + " | ".join(label for _, label in HEADLINE_METRICS) + " |",
            sep,
        ]
        for exp, avg in sorted(macro.items()):
            lines.append(
                f"| {exp} | " + " | ".join(_fmt(avg.get(k)) for k, _ in HEADLINE_METRICS) + " |"
            )
        lines.append("")
    lines += [
        "Macro averages are over ontologies where the metric applies "
        "(reference metrics skip SILKNOW).",
        "",
    ]
    return "\n".join(lines)
