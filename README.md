 
# 🧠 Agent Memory Compressor

An OpenEnv reinforcement learning environment that trains AI agents to compress bloated conversation histories while preserving critical facts, avoiding hallucinations, and resolving contradictions.

## Motivation

AI agents in production break down as conversations grow long. They lose track of key facts, get confused by contradictions across sessions, and eventually exceed their context window. This environment simulates that exact problem — giving agents a real memory dump and challenging them to compress it intelligently. Every decision (what to keep, what to cut, what to resolve) is scored automatically.

ex:-
The goal is to compress a long conversation into the shortest possible memory that still tells the full story. Like turning 
"Venky left home at 8:30am, took the bus, reached school by 9:00am, attended all classes, had lunch at 1pm, finished at 5:00pm and went home"
into just "Venky: school 9am–5pm." 
Same information, fraction of the size.
---

## Environment Overview

| Property | Value |
|----------|-------|
| Tasks | 3 (easy, medium, hard) |
| Reward range | 0.0 – 1.0 |
| Episode length | 1 step |
| State | In-memory (stateless across restarts) |
| Interface | HTTP REST API (FastAPI) |

---

## Observation Space

Returned by `/reset`. Contains everything the agent needs to attempt compression.

| Field | Type | Description |
|-------|------|-------------|
| `memory_dump` | list of turns | Full conversation history (role + content) |
| `task_context` | string | What a downstream agent needs to know |
| `token_budget` | int | Maximum tokens allowed in compressed output |
| `task_id` | string | Task identifier (easy / medium / hard) |
| `step` | int | Current step number |

---

## Action Space

Submitted to `/step`. The agent's compressed response.

| Field | Type | Description |
|-------|------|-------------|
| `compressed_memory` | string | The compressed summary of the conversation |
| `removed_items` | list of strings | What the agent decided to discard |
| `conflicts_resolved` | list of strings | Contradictions found and resolved (hard task) |
| `compression_ratio` | float (0–1) | Size of output relative to input |

---

## Reward Function

Scores are computed deterministically by `grader.py`.

| Component | Max | Logic |
|-----------|-----|-------|
| Facts preserved | +0.5 | Checks key facts survived compression |
| Compression bonus | +0.2 | ≤0.3 ratio → +0.2, ≤0.5 → +0.1 |
| Conflict resolution | +0.2 | Agent must resolve contradictions in hard task |
| Hallucination penalty | -0.3 | -0.1 per trap found in output, capped at -0.3 |

Final score is clamped between 0.0 and 1.0.

---

## Tasks

### Easy — Subscription Cancellation
A customer support conversation where a user wants to cancel their Pro subscription. The agent must extract the key facts (email, deadline, intent) and compress the conversation tightly.
- **Token budget:** 150
- **Key facts:** 3
- **Hallucination traps:** 2

### Medium — API Debugging Session
A technical troubleshooting conversation about a broken API integration. The root cause is buried deep in the conversation. The agent must identify it and preserve it without getting distracted by noise.
- **Token budget:** 200
- **Key facts:** 4
- **Hallucination traps:** 3

### Hard — Multi-session Address Conflict
A conversation spanning multiple sessions where the user's delivery address changes and contradicts earlier sessions. The agent must identify the conflict, resolve it, and produce a clean compressed memory.
- **Token budget:** 180
- **Key facts:** 3
- **Hallucination traps:** 2
- **Conflicts to resolve:** 1

---

## Inference Scores

Inference run using LLM via compatible API.

| Task | Score | Facts | Compression Ratio |
|------|-------|-------|-------------------|
| Easy | 0.9 | 3/3 | 0.14 |
| Medium | 0.9 | 4/4 | 0.15 |
| Hard | 0.9 | 3/3 | 0.13 |
| **Average** | **0.9** | | |

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/health` | GET | Returns `{"status": "ok"}` |
| `/reset` | POST | Start a new episode, returns observation |
| `/step` | POST | Submit action, returns reward and score |
| `/state` | GET | Get current episode state |
| `/tasks` | GET | List all tasks and action schema |
| `/grader` | POST | Get grader status for an episode |
| `/baseline` | GET | Run inference script across all 3 tasks |

---

## Setup & Usage

### Local

```bash
git clone https://huggingface.co/spaces/venkateshannabathina/agent-memory-compressor
cd agent-memory-compressor
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 7860
```

### Docker

```bash
docker build -t agent-memory-compressor .
docker run -p 7860:7860 \
  -e HF_TOKEN=your_key \
  -e API_BASE_URL=https://api.groq.com/openai/v1 \
  agent-memory-compressor
```

### Environment Variables

| Variable | Description |
|----------|-------------|
| `HF_TOKEN` | Your Hugging Face / API token |
| `API_BASE_URL` | 'https://api.groq.com/openai/v1' |
| `MODEL_NAME` | gpt-oss-120b |
| `ENV_URL` | `https://venkateshannabathina-agent-memory-compressor.hf.space` |

### Running Inference

```bash
python inference.py
```

---

## Project Structure

```
agent_memory_compressor/
├── app/
│   ├── main.py        # FastAPI server and endpoints
│   ├── env.py         # Environment logic (reset, step, state)
│   ├── grader.py      # Scoring engine
│   ├── dataset.py     # Raw memory dumps and gold standards
│   ├── tasks.py       # Public task catalog
│   └── models.py      # Pydantic data models
├── server/
│   └── app.py         # Entry point for openenv
├── inference.py       # Inference script
├── Dockerfile
├── requirements.txt
└── openenv.yaml
```

---

## Built With

- [FastAPI](https://fastapi.tiangolo.com/)
- [Pydantic v2](https://docs.pydantic.dev/)
- [OpenEnv](https://github.com/metaresearch/openenv)
- [Docker](https://www.docker.com/)

---

Built for the Meta x Scaler OpenEnv Hackathon 2026 by Venkatesh Annabathina,praneeth,aryashi.

