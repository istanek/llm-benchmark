from __future__ import annotations

from pydantic import BaseModel, Field


class BackendCapabilities(BaseModel):
    """Features and metric provenance advertised by one inference backend."""

    transport: str
    supports_seed: bool = False
    supports_streaming: bool = False
    supports_concurrency: bool = False
    native_metrics: set[str] = Field(default_factory=set)
    metric_limitations: dict[str, str] = Field(default_factory=dict)
