# grader.py
# This is the scoring engine.
# It takes the agent's Action and the gold answers, and returns a Reward.
# Never returns the same score twice — score always depends on what the agent actually wrote.

from app.env import Action, Reward


def compute_reward(action: Action, gold: dict) -> Reward:
    score = 0.0
    breakdown = {}

    compressed = action.compressed_memory.lower()

    # ── COMPONENT 1: Facts Preserved (0.0 to 0.5) ────────────────────────────
    # Check how many gold key facts made it into the compressed output.
    # We check word by word — if enough meaningful words from a fact appear, it counts.

    facts_hit = sum(
        1 for fact in gold["key_facts"]
        if any(word.lower() in compressed for word in fact.split() if len(word) > 3)
    )
    facts_score = round((facts_hit / len(gold["key_facts"])) * 0.5, 4)
    score += facts_score
    breakdown["facts_preserved"] = f"{facts_hit}/{len(gold['key_facts'])} facts → {facts_score}"


    # ── COMPONENT 2: Hallucination Penalty (0.0 to -0.3) ─────────────────────
    # Check if the agent invented things that were never in the original.
    # Each hallucination trap found in the compressed output costs 0.1 points.

    hallucination_hits = sum(
        1 for trap in gold.get("hallucination_traps", [])
        if trap.lower() in compressed
    )
    hallucination_penalty = round(min(0.3, hallucination_hits * 0.1), 4)
    score -= hallucination_penalty
    breakdown["hallucination_penalty"] = f"-{hallucination_penalty} ({hallucination_hits} traps found)"


    # ── COMPONENT 3: Compression Bonus (0.0 to 0.2) ──────────────────────────
    # Reward the agent for actually compressing aggressively.
    # ratio = compressed size / original size. Lower ratio = better compression.

    ratio = action.compression_ratio
    if ratio <= 0.3:
        compression_bonus = 0.2     # compressed to 30% or less — perfect
    elif ratio <= 0.5:
        compression_bonus = 0.1     # compressed to 50% — okay
    else:
        compression_bonus = 0.0     # barely compressed — no reward
    score += compression_bonus
    breakdown["compression_bonus"] = f"{compression_bonus} (ratio: {ratio})"


    # ── COMPONENT 4: Conflict Resolution (0.0 to 0.2) ────────────────────────
    # Only applies to the hard task which has conflicting sessions.
    # Check if the agent mentioned resolving the contradiction in conflicts_resolved.

    conflicts = gold.get("conflicts_to_resolve", [])
    if conflicts:
        resolved_text = " ".join(action.conflicts_resolved).lower()
        resolved_count = sum(
            1 for c in conflicts
            if any(word.lower() in resolved_text for word in c.split() if len(word) > 3)
        )
        conflict_score = round((resolved_count / len(conflicts)) * 0.2, 4)
    else:
        conflict_score = 0.2        # no conflicts in this task = full marks automatically
    score += conflict_score
    breakdown["conflict_resolution"] = conflict_score


    # ── FINAL SCORE ───────────────────────────────────────────────────────────
    # Clamp between 0.0 and 1.0 — can never go negative or above 1.
    final_score = round(max(0.0, min(score, 1.0)), 4)

    return Reward(
        score=final_score,
        facts_preserved=facts_score,
        compression_bonus=compression_bonus,
        conflict_resolution=conflict_score,
        hallucination_penalty=hallucination_penalty,
        downstream_qa_score=0.0,    # optional bonus, set to 0 for now
        breakdown=breakdown
    )