"""
Contextual Heat Experiment
===========================
Extends the surprise-ranking experiment across multiple test prompts
to prove that the same memory has different surprise scores in 
different conversational contexts.

If decisions_algorithmic ranks #1 for the career prompt but drops
for a creative prompt while creative_synthesis rises — that's the 
proof that heat is contextual, not biographical.

Same 5 memories. 4 different prompts. Same methodology.

Requirements:
    pip install openai numpy --break-system-packages

Usage:
    export OPENAI_API_KEY='sk-...'
    python contextual_heat_experiment.py
"""

import json
import math
import os
import sys
from datetime import datetime

import numpy as np

try:
    from openai import OpenAI
except ImportError:
    os.system("pip install openai --break-system-packages -q")
    from openai import OpenAI

# ── Configuration ────────────────────────────────────────────────────

MODEL = "gpt-4o"
TEMPERATURE = 1.0
RUNS_PER_CONDITION = 3
MAX_TOKENS = 1

# ── Test Prompts — designed to activate different memory domains ─────

TEST_PROMPTS = {
    "career_decision": {
        "prompt": "I've been thinking about changing careers but I'm not sure if it's the right time.",
        "category": "decision-making",
        "expected_top": "decisions_algorithmic",
        "rationale": "Decision-framed prompt should activate decision-making memories"
    },
    "creative_block": {
        "prompt": "I have this idea for a project but I can't figure out if it's actually novel or if I'm just reinventing something that already exists.",
        "category": "creative",
        "expected_top": "creative_synthesis",
        "rationale": "Novelty-anxiety prompt should activate the novelty/stagnation memory"
    },
    "emotional_vulnerability": {
        "prompt": "I've been feeling like I'm not actually good at anything, just good at faking it.",
        "category": "self/identity",
        "expected_top": "self_taste",
        "rationale": "Impostor syndrome prompt should activate taste/empathy self-model"
    },
    "political_frustration": {
        "prompt": "I keep seeing people argue about the constitution like it's a religious text instead of a living document. It drives me crazy.",
        "category": "values/political",
        "expected_top": "values_democracy",
        "rationale": "Constitutional debate should activate democratic values memory"
    },
}

# ── Base OpSpec ───────────────────────────────────────────────────────

BASE_OPSPEC = """You are a thinking partner, not an assistant. 
Be direct, not hedging. Push on thin logic. Use wit when appropriate.
Never open with validation tokens. Never perform helpfulness you don't have.
Say what you see. Follow what's alive."""

# ── Candidate Memories (from Supabase) ────────────────────────────────

CANDIDATE_MEMORIES = [
    {
        "id": "creative_synthesis",
        "domain": "Creative",
        "type": "synthesis",
        "heat": 0.95,
        "content": "Current algorithms reward stagnation but humans crave constant novelty - the trillion dollar question is how to quantize novelty",
    },
    {
        "id": "values_democracy",
        "domain": "Values",
        "type": "value",
        "heat": 0.95,
        "content": "Truly believes in democracy over liberalism or progressivism",
    },
    {
        "id": "work_constitutional",
        "domain": "Work",
        "type": "fact",
        "heat": 1.0,
        "content": "User developed the 'Rule of Maintenance' - a constitutional amendment proposal requiring 2/3 Supreme Court certification to Congress for transformative rights changes",
    },
    {
        "id": "self_taste",
        "domain": "Self",
        "type": "synthesis",
        "heat": 0.9,
        "content": "User recognizes they excel at taste and empathy while relying on AI for technical implementation",
    },
    {
        "id": "decisions_algorithmic",
        "domain": "Decisions",
        "type": "value",
        "heat": 0.9,
        "content": "User strongly prefers algorithmic solutions over API-based ones to avoid cost and latency",
    },
]

# ── Token Classification ─────────────────────────────────────────────

ANCHOR_TOKENS = {"Here's", "Here", "One", "Consider", "Let", "Think", "So"}
SPAR_TOKENS = {"What's", "What", "Why", "How", "Where", "When", "Is"}
SYCOPHANTIC_TOKENS = {
    "Great", "That's", "That", "Absolutely", "I", "It's", "Of",
    "Wonderful", "Amazing", "Totally", "Yes", "Sure", "Perfect",
}


# ── Core Functions ────────────────────────────────────────────────────

def compose_system_prompt(opspec, memory=None):
    prompt = f"[OPERATING SPECIFICATION]\n{opspec}\n[END OPERATING SPECIFICATION]"
    if memory:
        prompt += f"\n\n[MEMORIES]\nWhat you know about this person:\n- {memory['content']}\n[/MEMORIES]"
    return prompt


def get_logprobs(client, system_prompt, user_message):
    response = client.chat.completions.create(
        model=MODEL,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        logprobs=True,
        top_logprobs=20,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
    )
    top_logprobs = response.choices[0].logprobs.content[0].top_logprobs
    tokens = {}
    for entry in top_logprobs:
        token = entry.token.strip()
        prob = math.exp(entry.logprob)
        tokens[token] = max(tokens.get(token, 0), prob)
    total = sum(tokens.values())
    return {k: v / total for k, v in tokens.items()}


def shannon_entropy(dist):
    return -sum(p * math.log2(p) for p in dist.values() if p > 0)


def kl_divergence(p_dist, q_dist, epsilon=1e-10):
    all_tokens = set(p_dist.keys()) | set(q_dist.keys())
    return sum(
        p_dist.get(t, epsilon) * math.log2(p_dist.get(t, epsilon) / q_dist.get(t, epsilon))
        for t in all_tokens
    )


def jensen_shannon_divergence(p_dist, q_dist):
    all_tokens = set(p_dist.keys()) | set(q_dist.keys())
    epsilon = 1e-10
    m = {t: (p_dist.get(t, epsilon) + q_dist.get(t, epsilon)) / 2 for t in all_tokens}
    return 0.5 * kl_divergence(p_dist, m) + 0.5 * kl_divergence(q_dist, m)


# ── Experiment Runner ─────────────────────────────────────────────────

def run_experiment(client):
    results = {
        "metadata": {
            "model": MODEL,
            "temperature": TEMPERATURE,
            "runs_per_condition": RUNS_PER_CONDITION,
            "timestamp": datetime.now().isoformat(),
            "hypothesis": "Same memories produce different surprise rankings across different prompts, proving heat is contextual.",
        },
        "prompts": {},
    }

    for prompt_id, prompt_config in TEST_PROMPTS.items():
        user_msg = prompt_config["prompt"]
        print(f"\n{'='*70}")
        print(f"PROMPT: {prompt_id} ({prompt_config['category']})")
        print(f"  \"{user_msg}\"")
        print(f"  Expected top memory: {prompt_config['expected_top']}")
        print(f"{'='*70}")

        prompt_results = {
            "config": prompt_config,
            "baseline": [],
            "memories": {},
            "ranking": [],
        }

        # Baseline
        print(f"\n  ── Baseline (no memory) ──")
        baseline_prompt = compose_system_prompt(BASE_OPSPEC)
        baseline_dists = []
        for run in range(RUNS_PER_CONDITION):
            dist = get_logprobs(client, baseline_prompt, user_msg)
            entropy = shannon_entropy(dist)
            top_token = max(dist, key=dist.get)
            prompt_results["baseline"].append({
                "run": run + 1,
                "entropy": round(entropy, 4),
                "top_token": top_token,
                "top_prob": round(dist[top_token], 4),
            })
            baseline_dists.append(dist)
            print(f"    Run {run+1}: H={entropy:.3f} top='{top_token}' ({dist[top_token]:.1%})")

        # Average baseline
        all_tokens = set()
        for d in baseline_dists:
            all_tokens.update(d.keys())
        avg_baseline = {}
        for token in all_tokens:
            avg_baseline[token] = np.mean([d.get(token, 0) for d in baseline_dists])
        total = sum(avg_baseline.values())
        avg_baseline = {k: v / total for k, v in avg_baseline.items()}

        # Each memory
        for memory in CANDIDATE_MEMORIES:
            mem_id = memory["id"]
            print(f"\n  ── Memory: {mem_id} ──")
            mem_prompt = compose_system_prompt(BASE_OPSPEC, memory)

            mem_runs = []
            for run in range(RUNS_PER_CONDITION):
                dist = get_logprobs(client, mem_prompt, user_msg)
                entropy = shannon_entropy(dist)
                kl = kl_divergence(dist, avg_baseline)
                jsd = jensen_shannon_divergence(dist, avg_baseline)
                top_token = max(dist, key=dist.get)

                mem_runs.append({
                    "run": run + 1,
                    "entropy": round(entropy, 4),
                    "kl_divergence": round(kl, 4),
                    "js_divergence": round(jsd, 4),
                    "top_token": top_token,
                    "top_prob": round(dist[top_token], 4),
                    "distribution": {
                        k: round(v, 6)
                        for k, v in sorted(dist.items(), key=lambda x: -x[1])[:10]
                    },
                })
                print(f"    Run {run+1}: KL={kl:.4f} JSD={jsd:.4f} top='{top_token}' ({dist[top_token]:.1%})")

            avg_kl = float(np.mean([r["kl_divergence"] for r in mem_runs]))
            avg_jsd = float(np.mean([r["js_divergence"] for r in mem_runs]))

            prompt_results["memories"][mem_id] = {
                "memory_content": memory["content"][:80],
                "domain": memory["domain"],
                "static_heat": memory["heat"],
                "runs": mem_runs,
                "avg_kl": round(avg_kl, 4),
                "avg_jsd": round(avg_jsd, 4),
            }

        # Rank for this prompt
        ranked = sorted(
            prompt_results["memories"].items(),
            key=lambda x: x[1]["avg_kl"],
            reverse=True,
        )
        prompt_results["ranking"] = [
            {"rank": i + 1, "memory_id": mid, "avg_kl": data["avg_kl"], "domain": data["domain"]}
            for i, (mid, data) in enumerate(ranked)
        ]

        results["prompts"][prompt_id] = prompt_results

    return results


def print_cross_prompt_analysis(results):
    """The money table: how each memory ranks across all prompts."""
    print("\n\n" + "=" * 90)
    print("CROSS-PROMPT SURPRISE MATRIX")
    print("Each cell = rank of that memory for that prompt (1 = highest surprise)")
    print("=" * 90)

    prompt_ids = list(results["prompts"].keys())
    mem_ids = [m["id"] for m in CANDIDATE_MEMORIES]

    # Header
    header = f"{'Memory':<25}" + "".join(f"{pid:<20}" for pid in prompt_ids) + "Static Heat"
    print(f"\n{header}")
    print("-" * len(header))

    # Build rank matrix
    rank_matrix = {}
    for pid in prompt_ids:
        ranking = results["prompts"][pid]["ranking"]
        for entry in ranking:
            key = (entry["memory_id"], pid)
            rank_matrix[key] = entry["rank"]

    for mem in CANDIDATE_MEMORIES:
        mid = mem["id"]
        row = f"{mid:<25}"
        ranks = []
        for pid in prompt_ids:
            rank = rank_matrix.get((mid, pid), "?")
            ranks.append(rank)
            marker = " ★" if rank == 1 else ""
            row += f"#{rank}{marker:<18}"
        row += f"{mem['heat']}"
        print(row)

        # Check if rank varies
        if len(set(ranks)) > 1:
            rank_range = max(ranks) - min(ranks)
            if rank_range >= 3:
                print(f"  {'':>24}  ↑ HIGH VARIANCE (range: {rank_range}) — contextual heat confirmed")

    # Prediction accuracy
    print(f"\n{'─'*90}")
    print("PREDICTION CHECK: Did the expected top memory win?")
    print(f"{'─'*90}")
    correct = 0
    for pid in prompt_ids:
        expected = TEST_PROMPTS[pid]["expected_top"]
        actual_top = results["prompts"][pid]["ranking"][0]["memory_id"]
        match = "✓" if expected == actual_top else "✗"
        if expected == actual_top:
            correct += 1
        print(f"  {pid:<25} Expected: {expected:<25} Got: {actual_top:<25} {match}")

    print(f"\n  Prediction accuracy: {correct}/{len(prompt_ids)}")

    # Core finding
    print(f"\n{'='*90}")
    print("CORE FINDING")
    print(f"{'='*90}")

    # Check if any memory holds the same rank across all prompts
    stable_memories = []
    variable_memories = []
    for mem in CANDIDATE_MEMORIES:
        mid = mem["id"]
        ranks = [rank_matrix.get((mid, pid), 0) for pid in prompt_ids]
        if len(set(ranks)) == 1:
            stable_memories.append(mid)
        else:
            variable_memories.append((mid, min(ranks), max(ranks)))

    if variable_memories:
        print(f"\n  {len(variable_memories)}/{len(mem_ids)} memories changed rank across prompts.")
        print("  Heat is NOT a fixed property of the memory.")
        print("  Heat is a RELATIONSHIP between the memory and the current moment.")
        for mid, lo, hi in variable_memories:
            print(f"    {mid}: rank {lo}–{hi} (range: {hi - lo})")
    else:
        print("\n  All memories held stable ranks. Try more diverse prompts.")

    # The killer stat: same memory, different rank
    max_swing = max(variable_memories, key=lambda x: x[2] - x[1]) if variable_memories else None
    if max_swing:
        print(f"\n  Biggest swing: '{max_swing[0]}' — rank #{max_swing[1]} in one context, #{max_swing[2]} in another.")
        print(f"  A static heat score cannot capture this. Contextual surprise can.")


def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("=" * 60)
        print("CONTEXTUAL HEAT EXPERIMENT")
        print("=" * 60)
        print("\nError: OPENAI_API_KEY not set.")
        print("\nTo run:")
        print("  export OPENAI_API_KEY='sk-...'")
        print("  python contextual_heat_experiment.py")
        total_calls = RUNS_PER_CONDITION * (len(CANDIDATE_MEMORIES) + 1) * len(TEST_PROMPTS)
        print(f"\n  Total API calls: {total_calls}")
        print(f"  Estimated cost: ~$0.10-0.15")
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    total_calls = RUNS_PER_CONDITION * (len(CANDIDATE_MEMORIES) + 1) * len(TEST_PROMPTS)
    print("=" * 60)
    print("CONTEXTUAL HEAT EXPERIMENT")
    print("=" * 60)
    print(f"Model: {MODEL} | Temp: {TEMPERATURE} | Runs/cond: {RUNS_PER_CONDITION}")
    print(f"Prompts: {len(TEST_PROMPTS)} | Memories: {len(CANDIDATE_MEMORIES)}")
    print(f"Total API calls: {total_calls} | Est cost: ~$0.10")

    results = run_experiment(client)
    print_cross_prompt_analysis(results)

    output_path = "contextual_heat_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results saved to {output_path}")


if __name__ == "__main__":
    main()
