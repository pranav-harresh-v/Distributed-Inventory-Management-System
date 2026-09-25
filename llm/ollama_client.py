"""Adapter between the LLM service and the local Ollama runtime.

Boundary rule: this module is the only place in the project that knows Ollama
exists. Callers pass two prompts and receive generated text; they never see a
model tag negotiation, an HTTP status, an ``httpx`` exception or an Ollama
response object. Everything that leaks out of here is either a ``str`` or an
:class:`OllamaError`, which is what makes the runtime replaceable without
touching the gRPC layer above.

Two consequences of that rule are worth stating, because both were decided
deliberately rather than inherited:

* **Failures are re-raised as this module's own exceptions.** The ``ollama``
  package translates a refused connection into a builtin ``ConnectionError``
  and an HTTP error into ``ollama.ResponseError``, but it does *not* wrap
  timeouts -- an ``httpx.ReadTimeout`` propagates untouched. On a CPU-only
  Phi-3 with a 60-second budget that is the single most likely failure, so it
  is caught explicitly here rather than allowed to surface as a transport
  detail two layers up.
* **No database access.** Nothing in this package imports ``app.storage``;
  the caller supplies every fact the model is asked to reason about.

Configuration follows the project's existing style (see ``app/server.py``):
environment variables with working defaults, read through small validating
helpers, with no configuration framework in between.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Final

import httpx
import ollama

__all__ = [
    "DEFAULT_MODEL",
    "DEFAULT_OLLAMA_HOST",
    "DEFAULT_TEMPERATURE",
    "DEFAULT_TIMEOUT_SECONDS",
    "HOST_ENV_VAR",
    "MODEL_ENV_VAR",
    "TIMEOUT_ENV_VAR",
    "OllamaClient",
    "OllamaError",
    "OllamaGenerationError",
    "OllamaTimeoutError",
    "OllamaUnavailableError",
    "get_model",
    "get_ollama_host",
    "get_request_timeout",
]

_logger = logging.getLogger(__name__)

#: Read by the official ``ollama`` package itself, which is why it carries no
#: ``INVENTORY_`` prefix: a prefixed copy would only shadow the real variable.
HOST_ENV_VAR: Final[str] = "OLLAMA_HOST"
MODEL_ENV_VAR: Final[str] = "OLLAMA_MODEL"
TIMEOUT_ENV_VAR: Final[str] = "LLM_REQUEST_TIMEOUT_SECONDS"

DEFAULT_OLLAMA_HOST: Final[str] = "http://127.0.0.1:11434"
DEFAULT_MODEL: Final[str] = "phi3:mini"
DEFAULT_TIMEOUT_SECONDS: Final[float] = 60.0

#: Low, but deliberately not zero. Inventory analysis should give the same
#: advice twice for the same figures, which rules out a conversational
#: temperature; greedy decoding at 0.0 makes a small model repeat itself on
#: longer summaries, which is worse than a little sampling noise.
DEFAULT_TEMPERATURE: Final[float] = 0.1


class OllamaError(Exception):
    """Base class for runtime failures, so one ``except`` catches them all.

    Mirrors the role :class:`~app.domain.inventory.CatalogError` plays in the
    domain layer: callers that do not care *why* generation failed catch this,
    and the gRPC layer maps the two subclasses onto different status codes.
    """


class OllamaUnavailableError(OllamaError):
    """The runtime could not be reached.

    Covers a refused connection and a DNS, proxy or network failure: no
    answer is available right now, and retrying later may work.
    """


class OllamaTimeoutError(OllamaUnavailableError):
    """The runtime was reached but did not answer within the timeout.

    A subclass rather than a sibling, because a caller that only cares
    whether an answer arrived should keep catching
    :class:`OllamaUnavailableError` and get both. The distinction exists so
    the gRPC layer can report DEADLINE_EXCEEDED, which tells an operator to
    raise the budget rather than go looking for a dead daemon.
    """


class OllamaGenerationError(OllamaError):
    """The runtime was reached but did not return a usable answer.

    Covers a missing model, a rejected request and an empty completion.
    Retrying is unlikely to help without changing something first.
    """


def _env_text(name: str, default: str) -> str:
    """Read a string setting, treating blank as unset."""
    return os.environ.get(name, "").strip() or default


def _env_float(name: str, default: float) -> float:
    """Read a positive float setting.

    Mirrors ``_env_int`` in ``app/server.py``: an unparseable or nonsensical
    value fails loudly at startup rather than silently reverting to a default
    the operator did not ask for.
    """
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}") from exc
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def get_ollama_host() -> str:
    """Base URL of the Ollama runtime this node generates against."""
    return _env_text(HOST_ENV_VAR, DEFAULT_OLLAMA_HOST)


def get_model() -> str:
    """Model tag to generate with, e.g. ``phi3:mini``."""
    return _env_text(MODEL_ENV_VAR, DEFAULT_MODEL)


def get_request_timeout() -> float:
    """How long to wait for one completion, in seconds."""
    return _env_float(TIMEOUT_ENV_VAR, DEFAULT_TIMEOUT_SECONDS)


class OllamaClient:
    """Generates text with a local Ollama model.

    The underlying client is injectable so tests can exercise this class
    without a running daemon, the same way
    :class:`~app.api.client_service.ClientService` takes a session factory.
    Nothing else about the transport is configurable from outside.
    """

    def __init__(
        self,
        *,
        host: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
        client: ollama.Client | None = None,
    ) -> None:
        self._host = host if host is not None else get_ollama_host()
        self._model = model if model is not None else get_model()
        self._timeout = timeout if timeout is not None else get_request_timeout()
        # Extra keyword arguments reach the underlying httpx.Client, which is
        # how the timeout is applied to every request this client makes.
        self._client = client or ollama.Client(host=self._host, timeout=self._timeout)

    @property
    def host(self) -> str:
        return self._host

    @property
    def model(self) -> str:
        """Configured model tag, for attributing an answer to its model."""
        return self._model

    @property
    def timeout(self) -> float:
        return self._timeout

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        json_output: bool = False,
        temperature: float | None = None,
    ) -> str:
        """Run one non-streaming chat completion and return the reply text.

        ``system_prompt`` sets the model's role and ``user_prompt`` carries the
        question together with the facts to ground it in. Only the assistant's
        text comes back: no token counts, timings or response object, so a
        caller cannot start depending on the runtime's reply shape.

        ``json_output`` switches on Ollama's structured-output mode. It is a
        parameter rather than a policy because the prompt layer owns output
        format -- a reorder recommendation may want parseable JSON while a
        written summary wants prose, and this adapter should not decide which.

        Raises :class:`OllamaUnavailableError` if the runtime cannot be reached
        or overruns ``timeout``, and :class:`OllamaGenerationError` if it
        answers with an error or with nothing.
        """
        if not user_prompt.strip():
            raise OllamaGenerationError("user_prompt must not be blank.")

        options: dict[str, Any] = {
            "temperature": DEFAULT_TEMPERATURE if temperature is None else temperature
        }

        try:
            response = self._client.chat(
                model=self._model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                stream=False,
                format="json" if json_output else None,
                options=options,
            )
        except ConnectionError as exc:
            # What the ollama package raises when the daemon refuses a
            # connection; it has already discarded the httpx exception.
            raise OllamaUnavailableError(
                f"Could not reach the Ollama runtime at {self._host}."
            ) from exc
        except httpx.TimeoutException as exc:
            # Listed before TransportError, which it subclasses.
            raise OllamaTimeoutError(
                f"The Ollama runtime at {self._host} did not answer within "
                f"{self._timeout:g} seconds."
            ) from exc
        except httpx.TransportError as exc:
            raise OllamaUnavailableError(
                f"Could not reach the Ollama runtime at {self._host}."
            ) from exc
        except ollama.ResponseError as exc:
            raise OllamaGenerationError(
                f"The Ollama runtime rejected the request for model "
                f"'{self._model}': {exc}"
            ) from exc
        except ollama.RequestError as exc:
            raise OllamaGenerationError(
                f"The request to model '{self._model}' was malformed: {exc}"
            ) from exc

        text = response.message.content or ""
        if not text.strip():
            raise OllamaGenerationError(
                f"Model '{self._model}' returned an empty answer."
            )

        _logger.debug(
            "generated %d characters with model %s", len(text), self._model
        )
        return text
