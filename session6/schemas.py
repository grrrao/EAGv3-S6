from __future__ import annotations

from datetime import datetime
from typing import Literal, Self

from pydantic import BaseModel, model_validator


class MemoryItem(BaseModel):
    id: str
    kind: Literal["fact", "preference", "tool_outcome", "scratchpad"]
    keywords: list[str]
    descriptor: str
    value: dict
    artifact_id: str | None = None
    source: str
    run_id: str
    goal_id: str | None = None
    confidence: float = 1.0
    created_at: datetime


class Artifact(BaseModel):
    id: str
    content_type: str
    size_bytes: int
    source: str
    descriptor: str
    created_at: datetime


class Goal(BaseModel):
    id: str
    text: str
    done: bool = False
    attach_artifact_id: str | None = None


class Observation(BaseModel):
    goals: list[Goal]

    @property
    def all_done(self) -> bool:
        return bool(self.goals) and all(g.done for g in self.goals)

    def next_unfinished(self) -> Goal | None:
        for g in self.goals:
            if not g.done:
                return g
        return None


class ToolCall(BaseModel):
    name: str
    arguments: dict


class DecisionOutput(BaseModel):
    answer: str | None = None
    tool_call: ToolCall | None = None

    @model_validator(mode="after")
    def _exactly_one(self) -> Self:
        has_answer = self.answer is not None
        has_tool = self.tool_call is not None
        if has_answer == has_tool:
            raise ValueError("Exactly one of answer or tool_call must be set")
        return self

    @property
    def is_answer(self) -> bool:
        return self.answer is not None


class _PerceivedGoal(BaseModel):
    text: str
    done: bool = False
    attach_artifact_index: int = -1  # -1 means "no attachment"


class PerceptionRaw(BaseModel):
    goals: list[_PerceivedGoal]


class _ClassifiedItem(BaseModel):
    kind: Literal["fact", "preference", "tool_outcome", "scratchpad"]
    keywords: list[str]
    descriptor: str
    value: dict
    confidence: float = 1.0
