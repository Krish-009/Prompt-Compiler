"""Provider interface.

A provider is transport only: it turns (system, user, schema) into a validated object.
What to ask and how to interpret the answer belongs to analyzer/ and optimizer/, so a
second provider can be added without touching the domain logic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TypeVar

from pydantic import BaseModel

SchemaT = TypeVar("SchemaT", bound=BaseModel)

#: Provider messages are long - some end in several sentences of upgrade marketing - but
#: the useful part is at the front.
DETAIL_LIMIT = 300


def detail(exc: Exception) -> str:
    """The provider's own explanation of a failure, condensed onto one line.

    Worth carrying because it is routinely the only thing that says how to fix the problem.
    Found the hard way on the first live run of Phase 9: Gemini answered a 404 with "this
    model is no longer available to new users, use models/gemini-3.6-flash", and Groq
    answered a 413 with the exact token limit and the exact number requested. Both adapters
    reduced those to "API error (HTTP 404)" and "API error (HTTP 413)", throwing away the
    answer and leaving a user to guess.

    Keys never appear here: these messages are response bodies, which do not echo the
    credential, and the contract suite asserts no mapped error carries the key.
    """
    text = " ".join(str(getattr(exc, "message", None) or exc).split())
    if len(text) > DETAIL_LIMIT:
        return text[:DETAIL_LIMIT].rstrip() + "..."
    return text


class Provider(ABC):
    #: Short provider label, e.g. "anthropic".
    name: str
    #: Model identifier this provider instance will call.
    model: str

    @abstractmethod
    def structured(self, *, system: str, user: str, schema: type[SchemaT]) -> SchemaT:
        """Run one call and return an instance of `schema`.

        Raises MissingCredentialsError, ProviderError or InvalidResponseError.
        """
