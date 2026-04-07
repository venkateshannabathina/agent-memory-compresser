# inference.py
# This script proves your environment works.
# It uses GPT-4o to actually run through all 3 tasks and prints real scores.
# Judges hit /inference which triggers this script and returns the output.

import os
import json
import requests
from openai import OpenAI
from dotenv import load_dotenv

import sys
load_dotenv()

hf_token = os.environ.get("HF_TOKEN")
if not hf_token:
    print("❌ Error: 'HF_TOKEN' is missing from the .env file. Please add it to make authentication work.")
    sys.exit(1)

client = OpenAI(
    api_key=hf_token,
    base_url=os.environ.get("API_BASE_URL", "https://api.groq.com/openai/v1"),
)
BASE_URL = os.environ.get("ENV_URL", "https://venkateshannabathina-agent-memory-compressor.hf.space")


def run_task(task_id: str) -> float:

    # ── STEP 1: Reset the environment ─────────────────────────────────────────
    # Start a fresh episode and get the memory dump to compress.

    reset_resp = requests.post(f"{BASE_URL}/reset", params={"task_id": task_id})
    reset_resp.raise_for_status()
    data = reset_resp.json()

    episode_id = data["episode_id"]
    obs = data["observation"]

    # ── STEP 2: Build the prompt for GPT-4o ───────────────────────────────────
    # Format the memory dump into readable turns.
    # Tell the model exactly what to do and what format to return.

    turns = "\n".join(
        f"{turn['role'].upper()} [{turn.get('session_id') or 'main'}]: {turn['content']}"
        for turn in obs["memory_dump"]
    )

    prompt = f"""You are an AI memory compression agent.

Your job is to compress a bloated conversation history down to only what matters.
A downstream agent will use ONLY your compressed output to do their job — so preserve every critical fact.

What the downstream agent needs to do:
{obs['task_context']}

Token budget for your compressed output: {obs['token_budget']} tokens max.

Full conversation history to compress:
{turns}

Rules:
- Preserve every critical fact the downstream agent needs.
- Remove all filler, repetition, and small talk.
- If there are contradictions across sessions, keep the most recent truth only.
- Do NOT invent anything that was not in the original conversation.
- Stay within the token budget.

Respond ONLY with valid JSON in this exact format:
{{
  "compressed_memory": "<your compressed output here>",
  "removed_items": ["<what you cut and why, one line each>"],
  "conflicts_resolved": ["<contradiction you found and how you resolved it>"],
  "compression_ratio": <a float between 0.0 and 1.0 — compressed size divided by original size>
}}"""

    # ── STEP 3: Call GPT-4o ───────────────────────────────────────────────────

    response = client.chat.completions.create(
        model=os.environ.get("MODEL_NAME", "llama-3.3-70b-versatile"),
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )

    action_data = json.loads(response.choices[0].message.content)

    # ── STEP 4: Submit the action to the environment ──────────────────────────
    # Send GPT-4o's compressed output back to your /step endpoint.
    # The grader runs and returns the reward.

    step_resp = requests.post(
        f"{BASE_URL}/step",
        params={"episode_id": episode_id},
        json=action_data
    )
    step_resp.raise_for_status()
    result = step_resp.json()

    score = result["reward"]["score"]
    breakdown = result["reward"]["breakdown"]

    print(f"\nTask [{task_id.upper()}]")
    print(f"  Score          : {score}")
    print(f"  Breakdown      : {breakdown}")
    print(f"  Compressed to  : {action_data.get('compression_ratio', '?')} ratio")
    print(f"  Compressed out : {action_data.get('compressed_memory', '')[:120]}...")


    return score


# ── MAIN ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Running inference across all 3 tasks...\n")
    scores = {}

    for task in ["easy", "medium", "hard"]:
        print(f"[START] task={task}", flush=True)
        try:
            score = run_task(task)
            scores[task] = score
        except Exception as e:
            print(f"Task [{task}] FAILED: {e}")
            score = 0.0
            scores[task] = 0.0
        print(f"[STEP] step=1 reward={score}", flush=True)
        print(f"[END] task={task} score={score} steps=1", flush=True)

    print("\n── FINAL INFERENCE SCORES ──────────────────────")
    for task, score in scores.items():
        print(f"  {task:<8}: {score}")

    avg = sum(scores.values()) / len(scores)
    print(f"  average : {avg:.4f}")
    print("───────────────────────────────────────────────")