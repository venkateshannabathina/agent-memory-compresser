# models.py
# All Pydantic models live here to avoid circular imports.

from pydantic import BaseModel, Field
from typing import Optional

class MemoryTurn(BaseModel):
    role: str
    content: str
    session_id: Optional[str] = None

class Observation(BaseModel):
    memory_dump: list[MemoryTurn]
    task_context: str
    token_budget: int
    task_id: str
    step: int = 0

class Action(BaseModel):
    compressed_memory: str
    removed_items: list[str]
    conflicts_resolved: list[str]
    compression_ratio: float = Field(ge=0.0, le=1.0)

class Reward(BaseModel):
    score: float
    facts_preserved: float
    compression_bonus: float
    conflict_resolution: float
    hallucination_penalty: float
    downstream_qa_score: float
    breakdown: dict

class StepResult(BaseModel):
    observation: Observation
    reward: Reward
    done: bool
    info: dict