# main.py
# This is the FastAPI web server.
# It exposes all the endpoints the hackathon judges will test.
# Every route just calls the environment and returns the result.

from fastapi import FastAPI, HTTPException
from app.env import AgentMemoryEnv, Action
from app.tasks import TASKS

app = FastAPI(
    title="Agent Memory Compressor",
    description="RL environment for training agents to compress LLM conversation memory",
    version="1.0.0"
)

# One global environment instance shared across all requests
env = AgentMemoryEnv()


# ── HEALTH ────────────────────────────────────────────────────────────────────
# Judges hit this first. Must return 200 or you're eliminated in Phase 1.

@app.get("/health")
def health():
    return {"status": "ok"}


# ── RESET ─────────────────────────────────────────────────────────────────────
# Starts a new episode. Returns the memory dump the agent needs to compress.
# Call this before anything else.

@app.post("/reset")
def reset(task_id: str = "easy"):
    if task_id not in TASKS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown task_id: '{task_id}'. Valid options are: easy, medium, hard"
        )
    return env.reset(task_id)


# ── STEP ──────────────────────────────────────────────────────────────────────
# Submit the agent's compressed memory.
# Returns the reward with full scoring breakdown.

@app.post("/step")
def step(episode_id: str, action: Action):
    if episode_id not in env.state_store:
        raise HTTPException(
            status_code=404,
            detail=f"Episode '{episode_id}' not found. Call /reset first."
        )
    result = env.step(episode_id, action)
    return result.model_dump()


# ── STATE ─────────────────────────────────────────────────────────────────────
# Check the current state of any active episode.
# Useful for debugging mid-episode.

@app.get("/state")
def state(episode_id: str):
    s = env.get_state(episode_id)
    if not s:
        raise HTTPException(
            status_code=404,
            detail=f"Episode '{episode_id}' not found."
        )
    return s


# ── TASKS ─────────────────────────────────────────────────────────────────────
# Returns the list of all tasks + the action schema.
# Judges use this to understand what your env expects.

@app.get("/tasks")
def tasks():
    return {
        "tasks": [
            {
                "id": task_id,
                "task_context": task["task_context"],
                "token_budget": task["token_budget"],
                "action_schema": Action.model_json_schema()
            }
            for task_id, task in TASKS.items()
        ]
    }


# ── GRADER ────────────────────────────────────────────────────────────────────
# Returns the status of a completed episode.
# Judges call this after /step to confirm the episode was scored.

@app.post("/grader")
def grader(episode_id: str):
    s = env.get_state(episode_id)
    if not s:
        raise HTTPException(
            status_code=404,
            detail=f"Episode '{episode_id}' not found."
        )
    return {
        "episode_id": episode_id,
        "done": s.get("done"),
        "step": s.get("step"),
        "task_id": s.get("task_id")
    }


# ── BASELINE ──────────────────────────────────────────────────────────────────
# Triggers the baseline script and returns scores for all 3 tasks.
# Judges hit this to verify your env produces real, reproducible scores.

@app.get("/baseline")
def baseline():
    import subprocess
    result = subprocess.run(
        ["python", "baseline.py"],
        capture_output=True,
        text=True
    )
    return {
        "output": result.stdout,
        "errors": result.stderr,
        "returncode": result.returncode
    }