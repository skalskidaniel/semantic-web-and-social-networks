import pytest

from ontoagents.llm.cache import CachingLLMClient
from ontoagents.llm.claude_code import _flatten
from ontoagents.llm.client import LLMClient, Message, system, user
from ontoagents.llm.fake import FakeLLMClient
from ontoagents.llm.heuristic import HeuristicLLMClient


def test_fake_replies_in_order_and_records_calls():
    llm = FakeLLMClient(["a", "b"])
    assert [llm.complete([user("x")]).text for _ in range(3)] == ["a", "b", "b"]
    assert len(llm.calls) == 3
    assert isinstance(llm, LLMClient)
    with pytest.raises(ValueError):
        FakeLLMClient()


def test_cache_returns_stored_response(tmp_path):
    inner = FakeLLMClient(["first", "second"])
    cached = CachingLLMClient(inner, tmp_path)
    one = cached.complete([user("q")], temperature=0.0)
    two = cached.complete([user("q")], temperature=0.0)
    assert one.text == two.text == "first"
    assert not one.cached and two.cached
    assert two.input_tokens == one.input_tokens  # true cost preserved
    assert cached.complete([user("q")], temperature=0.5).text == "second"
    assert len(inner.calls) == 2


def test_flatten_for_cli():
    sys_text, prompt = _flatten([system("S"), user("U")])
    assert (sys_text, prompt) == ("S", "U")
    _, multi = _flatten([user("a"), Message(role="assistant", content="b"), user("c")])
    assert "<assistant>\nb\n</assistant>" in multi


def test_heuristic_client_plays_translator():
    llm = HeuristicLLMClient()
    reply = llm.complete(
        [system("You translate competency questions ..."), user("- [c1] Which animals eat plants?")]
    )
    assert '"cq_id": "c1"' in reply.text


def test_langchain_adapter_maps_messages_and_usage():
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from langchain_core.messages import AIMessage

    reply = AIMessage(
        content="hello",
        usage_metadata={"input_tokens": 7, "output_tokens": 3, "total_tokens": 10},
    )
    from ontoagents.llm.langchain_client import LangChainClient

    client = LangChainClient(GenericFakeChatModel(messages=iter([reply])), "fake-lc")
    out = client.complete([system("s"), user("u")])
    assert (out.text, out.input_tokens, out.output_tokens, out.model) == ("hello", 7, 3, "fake-lc")
