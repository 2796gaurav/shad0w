"""llm_teacher against a local OpenAI-compatible mock: structured outputs, plain-text fallback, retries, errors."""
import pytest

import shad0w
from shad0w.llm import PROVIDERS, match_option

from .mock_llm import MockLLM

OPTIONS = {"refund": "wants money back", "lost_card": "card lost or stolen", "balance": "asks balance", "transfer": "move money"}


@pytest.fixture
def mock(request):
    m = MockLLM(getattr(request, "param", "structured"))
    yield m
    m.close()


def test_structured_outputs(mock):
    t = shad0w.llm_teacher(OPTIONS, model="mock-1", base_url=mock.url, api_key="k")
    assert t("someone stole my card, block my card") == "lost_card"
    body = mock.requests[-1]["body"]
    assert body["response_format"]["json_schema"]["schema"]["properties"]["answer"]["enum"] == list(OPTIONS)
    assert body["temperature"] == 0 and body["model"] == "mock-1"
    assert mock.requests[-1]["headers"]["authorization"] == "Bearer k"
    assert "lost_card: card lost or stolen" in body["messages"][0]["content"]


@pytest.mark.parametrize("mock", ["plain"], indirect=True)
def test_falls_back_to_plain_text_once(mock):
    t = shad0w.llm_teacher(OPTIONS, model="mock-1", base_url=mock.url)
    assert t("send money to my friend") == "transfer"
    assert t("i want a refund") == "refund"
    assert "response_format" not in mock.requests[-1]["body"]
    assert len(mock.requests) == 3  # one rejection, then plain text remembered


@pytest.mark.parametrize("mock", ["flaky"], indirect=True)
def test_retries_server_errors(mock):
    t = shad0w.llm_teacher(OPTIONS, model="mock-1", base_url=mock.url, retries=2)
    assert t("what is my balance") == "balance" and len(mock.requests) == 2


def test_client_errors_raise(mock):
    t = shad0w.llm_teacher(OPTIONS, model="mock-1", base_url=mock.url)
    with pytest.raises(shad0w.TeacherError):
        t("boom")


def test_unreachable_server_raises_after_retries():
    t = shad0w.llm_teacher(OPTIONS, model="m", base_url="http://127.0.0.1:9/v1", retries=0, timeout=2)
    with pytest.raises(shad0w.TeacherError):
        t("hi")


def test_presets_and_keys(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "g")
    t = shad0w.llm_teacher(["a", "b"], model="groq/llama-3.1-8b-instant")
    assert t.base_url == PROVIDERS["groq"][0] and t.model == "llama-3.1-8b-instant" and t.api_key == "g"
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        shad0w.llm_teacher(["a", "b"], model="openai/gpt-4o-mini")
    assert shad0w.llm_teacher(["a", "b"], model="ollama/llama3.1").api_key is None
    t = shad0w.llm_teacher(["a", "b"], model="meta-llama/Llama-3-8b", base_url="http://x/v1")
    assert t.model == "meta-llama/Llama-3-8b"


@pytest.mark.parametrize("reply,want", [
    ("lost_card", "lost_card"), (" Lost Card. ", "lost_card"), ('{"answer": "refund"}', "refund"),
    ('```json\n{"intent": "balance"}\n```', "balance"), ("The answer is transfer.", "transfer"),
    ("refund or transfer", None), ("no idea", None), ("", None), ('["refund"]', "refund"),
])
def test_match_option(reply, want):
    assert match_option(reply, list(OPTIONS)) == want


def test_yesno():
    assert match_option("Yes.", ["yes", "no"]) == "yes" and match_option('{"answer": false}', ["yes", "no"]) == "no"


def test_sdk_client_path():
    class Msg:
        content = '{"answer": "refund"}'

    class Choice:
        message = Msg()

    class Resp:
        choices = [Choice()]

    calls = []

    class Completions:
        def create(self, **kw):
            calls.append(kw)
            return Resp()

    class Client:
        class chat:
            completions = Completions()

    t = shad0w.llm_teacher(OPTIONS, model="gpt-4o-mini", client=Client())
    assert t("money back please") == "refund" and calls[0]["model"] == "gpt-4o-mini"


def test_complete_hook_for_any_framework():
    seen = []

    def complete(messages):  # e.g. litellm.completion(...).choices[0].message.content, or langchain llm.invoke(...).content
        seen.append(messages)
        return "I think it is lost_card"

    t = shad0w.llm_teacher(OPTIONS, model="anything", complete=complete)
    assert t("stolen card") == "lost_card" and seen[0][1] == {"role": "user", "content": "stolen card"}
