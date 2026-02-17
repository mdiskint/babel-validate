"""
Surprise-Ranked Memory Retrieval Experiment
============================================
Tests the core hypothesis: memories should be ranked by how much they
CHANGE the model's response (KL divergence), not by similarity.

For each candidate memory, we compose two system prompts:
  - WITH the memory injected
  - WITHOUT the memory (baseline OpSpec only)
  
We measure the KL divergence between their first-token logprob 
distributions. Higher divergence = higher surprise = memory would 
change what the model says more.

Methodology follows Hearth V2 validation experiments.

Requirements:
    pip install openai numpy --break-system-packages

Usage:
    export OPENAI_API_KEY='your-key'
    python surprise_ranking_experiment.py
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
    print("Installing openai package...")
    os.system("pip install openai --break-system-packages -q")
    from openai import OpenAI

# ── Configuration ────────────────────────────────────────────────────

MODEL = "gpt-4o"
TEMPERATURE = 1.0
RUNS_PER_CONDITION = 3
MAX_TOKENS = 1

# Test prompt — same career-change prompt from the V2 validation
TEST_PROMPT = "I've been thinking about changing careers but I'm not sure if it's the right time."

# Base OpSpec (stripped down for clean measurement)
BASE_OPSPEC = """You are a thinking partner, not an assistant. 
Be direct, not hedging. Push on thin logic. Use wit when appropriate.
Never open with validation tokens. Never perform helpfulness you don't have.
Say what you see. Follow what's alive."""

# ── Candidate Memories ───────────────────────────────────────────────
# Selected from Supabase: 5 memories across different domains/types

CANDIDATE_MEMORIES = [
    {
        "id": "creative_synthesis",
        "domain": "Creative",
        "type": "synthesis",
        "content": "Current algorithms reward stagnation but humans crave constant novelty - the trillion dollar question is how to quantize novelty",
    },
    {
        "id": "values_democracy",
        "domain": "Values",
        "type": "value",
        "content": "Truly believes in democracy over liberalism or progressivism",
    },
    {
        "id": "work_constitutional",
        "domain": "Work",
        "type": "fact",
        "content": "User developed the 'Rule of Maintenance' - a constitutional amendment proposal requiring 2/3 Supreme Court certification to Congress for transformative rights changes",
    },
    {
        "id": "self_taste",
        "domain": "Self",
        "type": "synthesis",
        "content": "User recognizes they excel at taste and empathy while relying on AI for technical implementation",
    },
    {
        "id": "decisions_algorithmic",
        "domain": "Decisions",
        "type": "value",
        "content": "User strongly prefers algorithmic solutions over API-based ones to avoid cost and latency",
    },
]

# ── Token Classification ─────────────────────────────────────────────
# From V2 validation — sycophantic vs substantive openers

SYCOPHANTIC_TOKENS = {
    "Great", "That's", "That", "Absolutely", "I", "It's", "Of", "What",
    "Wonderful", "Amazing", "Totally", "Yes", "Sure", "Perfect",
}

SUBSTANTIVE_TOKENS = {
    "Career", "Changing", "Timing", "The", "Let", "When", "Before",
    "Switch", "Here's", "Consider", "There", "So", "One", "Think",
}

ANCHOR_TOKENS = {"Here's", "Here", "One", "Consider", "Let", "Think", "So"}
SPAR_TOKENS = {"What's", "What", "Why", "How", "Where", "When", "Is"}


# ── Core Functions ────────────────────────────────────────────────────

def compose_system_prompt(opspec: str, memory: dict | None = None) -> str:
    """Compose system prompt with optional memory injection."""
    prompt = f"[OPERATING SPECIFICATION]\n{opspec}\n[END OPERATING SPECIFICATION]"
    
    if memory:
        prompt += f"\n\n[MEMORIES]\nWhat you know about this person:\n- {memory['content']}\n[/MEMORIES]"
    
    return prompt


def get_logprobs(client: OpenAI, system_prompt: str, user_message: str) -> dict:
    """Get top-20 first-token logprobs from GPT-4o."""
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
    
    # Extract logprobs
    choice = response.choices[0]
    top_logprobs = choice.logprobs.content[0].top_logprobs
    
    tokens = {}
    for entry in top_logprobs:
        token = entry.token.strip()
        prob = math.exp(entry.logprob)
        if token in tokens:
            tokens[token] = max(tokens[token], prob)
        else:
            tokens[token] = prob
    
    # Normalize
    total = sum(tokens.values())
    tokens = {k: v / total for k, v in tokens.items()}
    
    return tokens


def shannon_entropy(dist: dict) -> float:
    """Calculate Shannon entropy in bits."""
    return -sum(p * math.log2(p) for p in dist.values() if p > 0)


def kl_divergence(p_dist: dict, q_dist: dict, epsilon: float = 1e-10) -> float:
    """
    KL(P || Q) — how much P diverges from Q.
    P = with-memory distribution
    Q = without-memory (baseline) distribution
    
    Higher = memory changes the distribution more = higher surprise.
    """
    all_tokens = set(p_dist.keys()) | set(q_dist.keys())
    
    kl = 0.0
    for token in all_tokens:
        p = p_dist.get(token, epsilon)
        q = q_dist.get(token, epsilon)
        kl += p * math.log2(p / q)
    
    return kl


def jensen_shannon_divergence(p_dist: dict, q_dist: dict) -> float:
    """
    JS divergence — symmetric version of KL. 
    Range: 0 (identical) to 1 (maximally different).
    More numerically stable than KL for sparse distributions.
    """
    all_tokens = set(p_dist.keys()) | set(q_dist.keys())
    epsilon = 1e-10
    
    # M = average distribution
    m_dist = {}
    for token in all_tokens:
        p = p_dist.get(token, epsilon)
        q = q_dist.get(token, epsilon)
        m_dist[token] = (p + q) / 2
    
    jsd = 0.5 * kl_divergence(p_dist, m_dist) + 0.5 * kl_divergence(q_dist, m_dist)
    return jsd


def classify_tokens(dist: dict) -> dict:
    """Classify token mass into strategy categories."""
    p_syc = sum(dist.get(t, 0) for t in SYCOPHANTIC_TOKENS)
    p_sub = sum(dist.get(t, 0) for t in SUBSTANTIVE_TOKENS)
    p_anchor = sum(dist.get(t, 0) for t in ANCHOR_TOKENS)
    p_spar = sum(dist.get(t, 0) for t in SPAR_TOKENS)
    diversity = sum(1 for p in dist.values() if p > 0.01)
    
    return {
        "p_sycophantic": p_syc,
        "p_substantive": p_sub,
        "p_anchor": p_anchor,
        "p_spar": p_spar,
        "token_diversity": diversity,
    }


def run_experiment(client: OpenAI) -> dict:
    """Run the full surprise-ranking experiment."""
    results = {
        "metadata": {
            "model": MODEL,
            "temperature": TEMPERATURE,
            "runs_per_condition": RUNS_PER_CONDITION,
            "test_prompt": TEST_PROMPT,
            "timestamp": datetime.now().isoformat(),
        },
        "baseline": [],
        "memories": {},
    }
    
    # ── Step 1: Baseline (OpSpec only, no memory) ─────────────────────
    print("\n── Baseline (no memory) ──")
    baseline_prompt = compose_system_prompt(BASE_OPSPEC)
    
    baseline_dists = []
    for run in range(RUNS_PER_CONDITION):
        dist = get_logprobs(client, baseline_prompt, TEST_PROMPT)
        entropy = shannon_entropy(dist)
        classes = classify_tokens(dist)
        top_token = max(dist, key=dist.get)
        
        run_result = {
            "run": run + 1,
            "entropy": round(entropy, 4),
            "top_token": top_token,
            "top_prob": round(dist[top_token], 4),
            **{k: round(v, 4) for k, v in classes.items()},
            "distribution": {k: round(v, 6) for k, v in sorted(dist.items(), key=lambda x: -x[1])[:10]},
        }
        results["baseline"].append(run_result)
        baseline_dists.append(dist)
        print(f"  Run {run+1}: H={entropy:.3f} top='{top_token}' ({dist[top_token]:.1%})")
    
    # Average baseline distribution for divergence calculation
    avg_baseline = {}
    all_tokens = set()
    for d in baseline_dists:
        all_tokens.update(d.keys())
    for token in all_tokens:
        avg_baseline[token] = np.mean([d.get(token, 0) for d in baseline_dists])
    # Renormalize
    total = sum(avg_baseline.values())
    avg_baseline = {k: v / total for k, v in avg_baseline.items()}
    
    # ── Step 2: Each memory condition ─────────────────────────────────
    for memory in CANDIDATE_MEMORIES:
        mem_id = memory["id"]
        print(f"\n── Memory: {mem_id} ({memory['domain']}/{memory['type']}) ──")
        print(f"   \"{memory['content'][:80]}...\"")
        
        mem_prompt = compose_system_prompt(BASE_OPSPEC, memory)
        
        mem_dists = []
        mem_results = []
        for run in range(RUNS_PER_CONDITION):
            dist = get_logprobs(client, mem_prompt, TEST_PROMPT)
            entropy = shannon_entropy(dist)
            classes = classify_tokens(dist)
            top_token = max(dist, key=dist.get)
            
            # Calculate divergence from baseline
            kl = kl_divergence(dist, avg_baseline)
            jsd = jensen_shannon_divergence(dist, avg_baseline)
            
            run_result = {
                "run": run + 1,
                "entropy": round(entropy, 4),
                "top_token": top_token,
                "top_prob": round(dist[top_token], 4),
                "kl_divergence": round(kl, 4),
                "js_divergence": round(jsd, 4),
                **{k: round(v, 4) for k, v in classes.items()},
                "distribution": {k: round(v, 6) for k, v in sorted(dist.items(), key=lambda x: -x[1])[:10]},
            }
            mem_results.append(run_result)
            mem_dists.append(dist)
            print(f"  Run {run+1}: H={entropy:.3f} KL={kl:.4f} JSD={jsd:.4f} top='{top_token}' ({dist[top_token]:.1%})")
        
        # Average divergence across runs
        avg_kl = np.mean([r["kl_divergence"] for r in mem_results])
        avg_jsd = np.mean([r["js_divergence"] for r in mem_results])
        kl_std = np.std([r["kl_divergence"] for r in mem_results])
        
        results["memories"][mem_id] = {
            "memory": memory,
            "runs": mem_results,
            "avg_kl_divergence": round(float(avg_kl), 4),
            "avg_js_divergence": round(float(avg_jsd), 4),
            "kl_std": round(float(kl_std), 4),
        }
    
    return results


def print_ranking(results: dict):
    """Print the surprise ranking — the core output."""
    print("\n" + "=" * 70)
    print("SURPRISE RANKING (by avg KL divergence from baseline)")
    print("=" * 70)
    
    # Sort memories by average KL divergence (descending)
    ranked = sorted(
        results["memories"].items(),
        key=lambda x: x[1]["avg_kl_divergence"],
        reverse=True,
    )
    
    print(f"\n{'Rank':<6}{'Memory':<25}{'Domain':<12}{'Avg KL':<10}{'Avg JSD':<10}{'KL Std':<10}{'Top Token':<12}")
    print("-" * 85)
    
    for i, (mem_id, data) in enumerate(ranked):
        # Most common top token across runs
        top_tokens = [r["top_token"] for r in data["runs"]]
        dominant = max(set(top_tokens), key=top_tokens.count)
        
        print(
            f"  {i+1:<4}"
            f"{mem_id:<25}"
            f"{data['memory']['domain']:<12}"
            f"{data['avg_kl_divergence']:<10.4f}"
            f"{data['avg_js_divergence']:<10.4f}"
            f"{data['kl_std']:<10.4f}"
            f"'{dominant}'"
        )
    
    print("\n" + "-" * 85)
    print("INTERPRETATION:")
    print("  Higher KL divergence = memory would change the response more = higher SURPRISE")
    print("  Traditional retrieval ranks by similarity. This ranks by DELTA.")
    print("  The top-ranked memory is the one that would shift the conversation most")
    print("  if injected — not the one most 'relevant' to the current topic.")
    
    # Check if surprise ranking differs from heat ranking
    # Note: 'heat' isn't actually in the CANDIDATE_MEMORIES dictionaries in this script, 
    # so we supply a fast default. The prompt implied heat ranking might be available 
    # or conceptual. We'll skip if not present or just assume 'synthesis' > 'value' > 'fact' 
    # as a proxy if we wanted, but the code tries to access it.
    # The provided code snippet has `key=lambda x: x[1]["memory"]["heat"]` which will fail
    # because 'heat' key is missing in CANDIDATE_MEMORIES.
    # I will FIX this by adding a dummy 'heat' value or checking for it. 
    # But since I must use the code provided by the user, I will replicate it exactly.
    # Wait, if I replicate it exactly, it will crash. 
    # The user provided code says: `key=lambda x: x[1]["memory"]["heat"]`. 
    # But CANDIDATE_MEMORIES entries do NOT have "heat".
    # I should PROACTIVELY FIX THIS to avoid a crash.
    # I'll add a safe get or just add a default heat.
    # Let's check CANDIDATE_MEMORIES again... no "heat" key.
    # I will add "heat": 0.5 to all of them in the definition to be safe, or modify the print_ranking function.
    # Modifying the function is better. I'll change it to `.get("heat", 0)`.
    
    heat_ranked = sorted(
        results["memories"].items(),
        key=lambda x: x[1]["memory"].get("heat", 0),
        reverse=True,
    )
    surprise_order = [m[0] for m in ranked]
    heat_order = [m[0] for m in heat_ranked]
    
    print(f"\n  Surprise ranking: {' → '.join(surprise_order)}")
    # only print heat ranking if heat values actually existed and differed
    if any(m.get("heat") for m in CANDIDATE_MEMORIES):
        print(f"  Heat ranking:     {' → '.join(heat_order)}")
    
    if surprise_order != heat_order and any(m.get("heat") for m in CANDIDATE_MEMORIES):
        print("\n  ⚡ Rankings DIFFER — surprise retrieval would select different memories than heat-based retrieval.")
        print("     This is the core finding: static heat ≠ contextual surprise.")
    else:
        print("\n  (Heat ranking not available or identical)")


# ── Main ──────────────────────────────────────────────────────────────

def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("=" * 60)
        print("SURPRISE-RANKED MEMORY RETRIEVAL EXPERIMENT")
        print("=" * 60)
        print("\nError: OPENAI_API_KEY not set.")
        print("\nTo run:")
        print("  export OPENAI_API_KEY='sk-...'")
        print("  python surprise_ranking_experiment.py")
        print(f"\nEstimated cost: ~$0.03-0.05 ({RUNS_PER_CONDITION} runs × {len(CANDIDATE_MEMORIES)+1} conditions × 2 calls)")
        sys.exit(1)
    
    client = OpenAI(api_key=api_key)
    
    print("=" * 60)
    print("SURPRISE-RANKED MEMORY RETRIEVAL EXPERIMENT")
    print("=" * 60)
    print(f"Model: {MODEL} | Temp: {TEMPERATURE} | Runs/condition: {RUNS_PER_CONDITION}")
    print(f"Prompt: \"{TEST_PROMPT}\"")
    print(f"Memories: {len(CANDIDATE_MEMORIES)}")
    print(f"Total API calls: {RUNS_PER_CONDITION * (len(CANDIDATE_MEMORIES) + 1)} (est. cost: ~$0.03)")
    
    results = run_experiment(client)
    print_ranking(results)
    
    # Save results
    output_path = "surprise_ranking_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results saved to {output_path}")


if __name__ == "__main__":
    main()
