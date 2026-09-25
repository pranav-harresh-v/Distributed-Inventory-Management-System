"""The Ollama adapter (Phase G) -- the project's only window onto the runtime.

Every test here injects a stub client, so the suite never starts a daemon,
never pulls a model and never opens a socket. That is a requirement rather
than a convenience: a test that needs Phi-3 running would fail on a machine
where it is not installed, and would take minutes when it is.

Two properties are worth more than the happy path and get most of the
attention below:

* **Nothing from the transport escapes.** ``httpx`` and ``ollama`` exceptions
  must never reach a caller, or the gRPC layer above would end up catching
  library types it should know nothing about. Timeouts matter most: the
  ``ollama`` package wraps refused connections but lets ``httpx.ReadTimeout``
  straight through, and a slow CPU generation is the likeliest real failure.
* **The adapter reads no database**, asserted by source inspection in the
  style the Phase F suite already uses for ``.commit()``.
"""

from __future__ import annotations

import ast
import inspect
import socket
from pathlib import Path
from typing import Any

import httpx
import ollama
import pytest

from llm import ollama_client
from llm.ollama_client import (
    DEFAULT_MODEL,
    DEFAULT_OLLAMA_HOST,
    DEFAULT_TEMPERATURE,
    DEFAULT_TIMEOUT_SECONDS,
    OllamaClient,
    OllamaError,
    OllamaGenerationError,
    OllamaTimeoutError,
    OllamaUnavailableError,
)

ANSWER = "Stock of SKU-1 covers about nine days of demand."


class StubClient:
    """Stands in for ``ollama.Client``, recording the call it was given.

    Returns a real :class:`ollama.ChatResponse` rather than a dict, so the
    adapter is exercised against the shape the library actually produces.
    """

    def __init__(
        self, *, content: str | None = ANSWER, raises: BaseException | None = None
    ) -> None:
        self.content = content
        self.raises = raises
        self.calls: list[dict[str, Any]] = []

    def chat(self, **kwargs: Any) -> ollama.ChatResponse:
        self.calls.append(kwargs)
        if self.raises is not None:
            raise self.raises
        return ollama.ChatResponse(
            model=kwargs.get("model", DEFAULT_MODEL),
            message=ollama.Message(role="assistant", content=self.content),
        )

    @property
    def last_call(self) -> dict[str, Any]:
        return self.calls[-1]


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unset the Phase G variables so a developer's shell cannot skew a test."""
    for name in ("OLLAMA_HOST", "OLLAMA_MODEL", "LLM_REQUEST_TIMEOUT_SECONDS"):
        monkeypatch.delenv(name, raising=False)


def build(**kwargs: Any) -> tuple[OllamaClient, StubClient]:
    """An adapter wired to a stub, with the stub returned for assertions."""
    stub = StubClient(**kwargs)
    return OllamaClient(client=stub), stub


# --- configuration ---------------------------------------------------------
def test_defaults_apply_when_environment_is_unset() -> None:
    client = OllamaClient(client=StubClient())

    assert client.host == DEFAULT_OLLAMA_HOST == "http://127.0.0.1:11434"
    assert client.model == DEFAULT_MODEL == "phi3:mini"
    assert client.timeout == DEFAULT_TIMEOUT_SECONDS == 60.0


def test_configuration_is_read_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OLLAMA_HOST", "http://10.0.0.5:11434")
    monkeypatch.setenv("OLLAMA_MODEL", "phi3:medium")
    monkeypatch.setenv("LLM_REQUEST_TIMEOUT_SECONDS", "12.5")

    client = OllamaClient(client=StubClient())

    assert client.host == "http://10.0.0.5:11434"
    assert client.model == "phi3:medium"
    assert client.timeout == 12.5


def test_blank_environment_values_fall_back_to_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OLLAMA_HOST", "   ")
    monkeypatch.setenv("OLLAMA_MODEL", "")

    client = OllamaClient(client=StubClient())

    assert client.host == DEFAULT_OLLAMA_HOST
    assert client.model == DEFAULT_MODEL


def test_explicit_arguments_override_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OLLAMA_MODEL", "phi3:medium")

    client = OllamaClient(model="llama3", timeout=5.0, client=StubClient())

    assert client.model == "llama3"
    assert client.timeout == 5.0


@pytest.mark.parametrize("value", ["not-a-number", "0", "-1"])
def test_an_unusable_timeout_is_rejected_loudly(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    """Failing at construction beats silently reverting to 60 seconds."""
    monkeypatch.setenv("LLM_REQUEST_TIMEOUT_SECONDS", value)

    with pytest.raises(ValueError, match="LLM_REQUEST_TIMEOUT_SECONDS"):
        OllamaClient(client=StubClient())


def test_underlying_client_is_built_with_the_configured_host_and_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The timeout only takes effect because it reaches the real client."""
    captured: dict[str, Any] = {}

    def fake_client_factory(**kwargs: Any) -> StubClient:
        captured.update(kwargs)
        return StubClient()

    monkeypatch.setenv("OLLAMA_HOST", "http://10.0.0.5:11434")
    monkeypatch.setattr(ollama_client.ollama, "Client", fake_client_factory)

    OllamaClient(timeout=30.0)

    assert captured == {"host": "http://10.0.0.5:11434", "timeout": 30.0}


# --- successful generation -------------------------------------------------
def test_generate_returns_only_the_assistant_text() -> None:
    client, _ = build()

    assert client.generate(system_prompt="sys", user_prompt="ask") == ANSWER


def test_generate_sends_both_prompts_in_order() -> None:
    client, stub = build()

    client.generate(system_prompt="You are an inventory analyst.", user_prompt="ask")

    assert stub.last_call["messages"] == [
        {"role": "system", "content": "You are an inventory analyst."},
        {"role": "user", "content": "ask"},
    ]
    assert stub.last_call["model"] == DEFAULT_MODEL


def test_generate_never_streams() -> None:
    """A streaming reply would be an iterator, not the text callers expect."""
    client, stub = build()

    client.generate(system_prompt="sys", user_prompt="ask")

    assert stub.last_call["stream"] is False


def test_generate_uses_a_low_temperature_by_default() -> None:
    client, stub = build()

    client.generate(system_prompt="sys", user_prompt="ask")

    temperature = stub.last_call["options"]["temperature"]
    assert temperature == DEFAULT_TEMPERATURE
    assert 0 < temperature <= 0.2


def test_temperature_can_be_overridden_per_call() -> None:
    client, stub = build()

    client.generate(system_prompt="sys", user_prompt="ask", temperature=0.7)

    assert stub.last_call["options"]["temperature"] == 0.7


def test_structured_output_is_off_unless_requested() -> None:
    """The prompt layer owns output format; the adapter only carries it."""
    client, stub = build(content='{"reorder": 40}')

    client.generate(system_prompt="sys", user_prompt="ask")
    assert stub.last_call["format"] is None

    client.generate(system_prompt="sys", user_prompt="ask", json_output=True)
    assert stub.last_call["format"] == "json"


def test_json_mode_still_returns_raw_text() -> None:
    """Parsing belongs to the caller that asked for JSON, not to the adapter."""
    client, _ = build(content='{"reorder": 40}')

    result = client.generate(system_prompt="sys", user_prompt="ask", json_output=True)

    assert result == '{"reorder": 40}'


# --- the runtime is unreachable --------------------------------------------
def test_refused_connection_becomes_unavailable() -> None:
    """What the ollama package raises once it has discarded the httpx error."""
    client, _ = build(raises=ConnectionError("connection refused"))

    with pytest.raises(OllamaUnavailableError, match="Could not reach"):
        client.generate(system_prompt="sys", user_prompt="ask")


def test_timeout_becomes_unavailable_and_names_the_budget() -> None:
    """The ollama package does not wrap timeouts, so this adapter must."""
    stub = StubClient(raises=httpx.ReadTimeout("timed out"))
    client = OllamaClient(timeout=45.0, client=stub)

    with pytest.raises(OllamaUnavailableError, match="45 seconds"):
        client.generate(system_prompt="sys", user_prompt="ask")


def test_a_timeout_is_distinguishable_from_an_unreachable_runtime() -> None:
    """Lets the gRPC layer report DEADLINE_EXCEEDED rather than UNAVAILABLE.

    A subclass, so a caller that only cares whether an answer arrived can go
    on catching OllamaUnavailableError and still get both.
    """
    client, _ = build(raises=httpx.ReadTimeout("timed out"))

    with pytest.raises(OllamaTimeoutError):
        client.generate(system_prompt="sys", user_prompt="ask")

    assert issubclass(OllamaTimeoutError, OllamaUnavailableError)

    refused, _ = build(raises=ConnectionError("refused"))
    with pytest.raises(OllamaUnavailableError) as exc:
        refused.generate(system_prompt="sys", user_prompt="ask")
    assert not isinstance(exc.value, OllamaTimeoutError)


def test_transport_failure_becomes_unavailable() -> None:
    client, _ = build(raises=httpx.ConnectError("no route to host"))

    with pytest.raises(OllamaUnavailableError):
        client.generate(system_prompt="sys", user_prompt="ask")


# --- the runtime answers with an error -------------------------------------
def test_missing_model_becomes_a_generation_error() -> None:
    client, _ = build(
        raises=ollama.ResponseError('model "phi3:mini" not found', status_code=404)
    )

    with pytest.raises(OllamaGenerationError, match="phi3:mini"):
        client.generate(system_prompt="sys", user_prompt="ask")


def test_malformed_request_becomes_a_generation_error() -> None:
    client, _ = build(raises=ollama.RequestError("bad request"))

    with pytest.raises(OllamaGenerationError):
        client.generate(system_prompt="sys", user_prompt="ask")


@pytest.mark.parametrize("content", ["", "   \n\t ", None])
def test_an_empty_completion_is_a_generation_error(content: str | None) -> None:
    """Returning "" would surface later as a blank answer field, not a fault."""
    client, _ = build(content=content)

    with pytest.raises(OllamaGenerationError, match="empty answer"):
        client.generate(system_prompt="sys", user_prompt="ask")


@pytest.mark.parametrize("user_prompt", ["", "   "])
def test_a_blank_question_is_rejected_before_the_runtime_is_called(
    user_prompt: str,
) -> None:
    client, stub = build()

    with pytest.raises(OllamaGenerationError, match="must not be blank"):
        client.generate(system_prompt="sys", user_prompt=user_prompt)

    assert stub.calls == []


# --- the boundary itself ---------------------------------------------------
@pytest.mark.parametrize(
    "failure",
    [
        ConnectionError("refused"),
        httpx.ReadTimeout("timed out"),
        httpx.ConnectError("no route"),
        ollama.ResponseError("boom", status_code=500),
        ollama.RequestError("bad"),
    ],
)
def test_no_transport_exception_ever_escapes(failure: BaseException) -> None:
    """The whole point of the adapter: callers only ever see OllamaError."""
    client, _ = build(raises=failure)

    with pytest.raises(OllamaError):
        client.generate(system_prompt="sys", user_prompt="ask")


def test_both_failure_types_share_one_base_class() -> None:
    assert issubclass(OllamaUnavailableError, OllamaError)
    assert issubclass(OllamaGenerationError, OllamaError)


def imported_modules(source: str) -> set[str]:
    """Every module a source file imports, read from its syntax tree.

    Parsed rather than grepped: these files *discuss* ``app.storage`` in their
    docstrings precisely because they must not import it, and a substring
    search cannot tell prose from an import.
    """
    names: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            names.add(node.module)
    return names


FORBIDDEN_IN_LLM_PACKAGE = ("app", "sqlalchemy", "sqlite3")


def test_the_suite_cannot_construct_a_real_runtime_client() -> None:
    """The hermeticity guard in conftest fires rather than dialling Ollama.

    Without an injected adapter this call would build a real
    ``ollama.Client``. The suite must pass on a machine with no Ollama and no
    model, so the attempt is a test failure rather than a slow timeout.
    """
    with pytest.raises(AssertionError, match="real ollama.Client"):
        OllamaClient()


def test_the_suite_cannot_reach_ollamas_port() -> None:
    with pytest.raises(AssertionError, match="Ollama runtime"):
        socket.socket().connect(("127.0.0.1", 11434))


def test_the_suite_cannot_reach_the_internet() -> None:
    """No test may depend on a network the grader might not have."""
    with pytest.raises(AssertionError, match="without internet access"):
        socket.socket().connect(("93.184.216.34", 80))


def test_the_adapter_never_reaches_a_database() -> None:
    """The LLM node holds no state; the caller supplies every fact.

    Asserted on the source rather than by behaviour, in the style the Phase F
    suite uses to prove no domain function commits.
    """
    imported = imported_modules(inspect.getsource(ollama_client))

    for module in imported:
        root = module.split(".")[0]
        assert root not in FORBIDDEN_IN_LLM_PACKAGE, module


def test_the_llm_package_imports_nothing_from_the_application() -> None:
    """Holds for the whole package, not just the module under test."""
    package = Path(ollama_client.__file__).parent

    for path in package.glob("*.py"):
        for module in imported_modules(path.read_text(encoding="utf-8")):
            root = module.split(".")[0]
            assert root not in FORBIDDEN_IN_LLM_PACKAGE, f"{path.name}: {module}"
