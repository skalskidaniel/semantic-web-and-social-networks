"""CQ-to-SPARQL translator: sees only the draft's signature, never its axioms."""

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict
from rdflib.plugins.sparql import prepareQuery

from ontoagents.agents.base import LLMAgent, PromptRenderer, extract_json
from ontoagents.domain import CompetencyQuestion, Signature, Usage
from ontoagents.domain.sparql import UNTRANSLATABLE, ensure_prefixes, is_untranslatable
from ontoagents.llm.client import LLMClient, system, user


class TranslationOutput(BaseModel):
    """SPARQL per CQ id (or ``UNTRANSLATABLE``)."""

    model_config = ConfigDict(frozen=True)

    queries: dict[str, str]
    usage: Usage
    parse_retries: int = 0


class CQToSparqlAgent(LLMAgent):
    """Translates a batch of CQs; retries once on unparseable SPARQL."""

    def __init__(
        self,
        llm: LLMClient,
        renderer: PromptRenderer | None = None,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        max_parse_retries: int = 1,
    ) -> None:
        super().__init__(llm, renderer, temperature=temperature, max_tokens=max_tokens)
        self._max_parse_retries = max_parse_retries

    def translate(
        self, cqs: Sequence[CompetencyQuestion], signature: Signature, base_iri: str
    ) -> TranslationOutput:
        """Return one query per CQ. CQs the model omitted are marked UNTRANSLATABLE."""
        sys_msg = system(self._renderer.render("translator_system.j2", base_iri=base_iri))
        sig_text = signature.to_prompt(base_iri)
        queries: dict[str, str] = {}
        usage = Usage()
        pending = list(cqs)
        parse_errors: dict[str, str] = {}
        retries = 0
        for attempt in range(self._max_parse_retries + 1):
            prompt = self._renderer.render(
                "translator_user.j2", signature=sig_text, cqs=pending, parse_errors=parse_errors
            )
            response, call_usage = self._call([sys_msg, user(prompt)])
            usage = usage + call_usage
            parsed = self._parse(response.text, {cq.id for cq in pending})
            parse_errors = {}
            for cq in pending:
                sparql = parsed.get(cq.id)
                if sparql is None:
                    continue
                error = _parse_error(sparql, base_iri)
                if error is None or attempt == self._max_parse_retries:
                    queries[cq.id] = sparql
                else:
                    parse_errors[cq.id] = error
            pending = [cq for cq in pending if cq.id in parse_errors]
            if not pending:
                break
            retries += 1
        for cq in cqs:
            queries.setdefault(cq.id, f"{UNTRANSLATABLE}: no query returned by the translator")
        return TranslationOutput(queries=queries, usage=usage, parse_retries=retries)

    @staticmethod
    def _parse(text: str, wanted: set[str]) -> dict[str, str]:
        try:
            payload = extract_json(text)
        except ValueError:
            return {}
        items = payload.get("queries", []) if isinstance(payload, dict) else payload
        result: dict[str, str] = {}
        if not isinstance(items, list):
            return result
        for item in items:
            if not isinstance(item, dict):
                continue
            cq_id = str(item.get("cq_id", ""))
            sparql = item.get("sparql")
            if cq_id in wanted and isinstance(sparql, str) and sparql.strip():
                result[cq_id] = sparql.strip()
        return result


def _parse_error(sparql: str, base_iri: str) -> str | None:
    if is_untranslatable(sparql):
        return None
    try:
        prepareQuery(ensure_prefixes(sparql, base_iri))
    except Exception as exc:  # pyparsing / rdflib errors
        return str(exc)[:300]
    return None
