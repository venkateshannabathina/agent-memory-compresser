 # env.py
# This is the core environment file.
# It has 2 things: the Pydantic models (data shapes) and the environment class (the logic).

import uuid
from pydantic import BaseModel, Field
from typing import Optional
from app.tasks import TASKS
from app.grader import compute_reward


# ── PYDANTIC MODELS ──────────────────────────────────────────────────────────
# These define the exact shape of every piece of data flowing through your env.
# OpenEnv spec requires typed Observation, Action, and Reward models.

class MemoryTurn(BaseModel):
    role: str                           # "user" or "assistant"
    content: str                        # what was said
    session_id: Optional[str] = None    # only used in the hard task (s1, s2, s3)


class Observation(BaseModel):
    memory_dump: list[MemoryTurn]       # the bloated conversation the agent must compress
    task_context: str                   # what the downstream agent needs to do
    token_budget: int                   # max tokens the compressed output can use
    task_id: str                        # "easy", "medium", or "hard"
    step: int = 0                       # which step we're on in the episode


class Action(BaseModel):
    compressed_memory: str              # the agent's compressed output
    removed_items: list[str]            # what the agent cut and why (one line each)
    conflicts_resolved: list[str]       # any contradictions found and how resolved
    compression_ratio: float = Field(ge=0.0, le=1.0)  # compressed/original size ratio


class Reward(BaseModel):
    score: float                        # final score 0.0 to 1.0
    facts_preserved: float              # how many gold facts made it through
    compression_bonus: float            # reward for compressing aggressively
    conflict_resolution: float          # reward for resolving contradictions
    hallucination_penalty: float        # penalty for making stuff up
    downstream_qa_score: float          # bonus (optional, set to 0.0 for now)
    breakdown: dict                     # full readable breakdown for debugging


class StepResult(BaseModel):
    observation: Observation
    reward: Reward
    done: bool
    info: dict


# ── ENVIRONMENT CLASS ─────────────────────────────────────────────────────────
# This is the actual gym. It manages episodes, hands out observations,
# takes actions, and returns rewards.

class AgentMemoryEnv:
    def __init__(self):
        self.state_store = {}           # stores active episodes by episode_id

    def reset(self, task_id: str) -> dict:
        # Called at the start of every episode.
        # Gives the agent a fresh memory dump to work with.
        episode_id = str(uuid.uuid4())  # unique ID for this episode
        task = TASKS[task_id]

        obs = Observation(
            memory_dump=[MemoryTurn(**turn) for turn in task["memory_dump"]],
            task_context=task["task_context"],
            token_budget=task["token_budget"],
            task_id=task_id,
            step=0
        )

        self.state_store[episode_id] = {
            "task_id": task_id,
            "obs": obs,
            "done": False,
            "step": 0
        }

        return {
            "episode_id": episode_id,
            "observation": obs.model_dump()
        }

    def step(self, episode_id: str, action: Action) -> StepResult:
        # Called when the agent submits its compressed memory.
        # Sends the action to the grader and returns the reward.
        state = self.state_store[episode_id]
        task = TASKS[state["task_id"]]

        reward = compute_reward(action, task["gold"])

        state["done"] = True
        state["step"] += 1

        return StepResult(
            observation=state["obs"],
            reward=reward,
            done=True,
            info={
                "task_id": state["task_id"],
                "step": state["step"]
            }
        )

    def get_state(self, episode_id: str) -> dict:
        # Returns the current state of an episode.
        # Used by the /state endpoint.
        state = self.state_store.get(episode_id, {})
        if not state:
            return {}
        return {
            "task_id": state["task_id"],
            "done": state["done"],
            "step": state["step"],
            "observation": state["obs"].model_dump()
        }