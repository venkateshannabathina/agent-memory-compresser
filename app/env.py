# env.py
# This is the core environment file.
# It has 2 things: the Pydantic models (data shapes) and the environment class (the logic).

import uuid
from app.models import MemoryTurn, Observation, Action, Reward, StepResult
from app.tasks import TASKS
from app.grader import compute_reward

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