"""
Surprise vs Similarity Inversion Experiment
=============================================
Three questions, one script:

1. RESPONSE QUALITY: Does surprise-ranked retrieval produce better 
   responses than similarity-ranked retrieval? (Full response comparison)

2. INVERSION HYPOTHESIS: Are similarity and surprise inversely correlated
   among semantic candidates? (Spearman correlation)

3. DIFFERENT MEMORIES: Does surprise ranking surface genuinely different
   memories, or just reorder the same top set?

Methodology:
  - Use OpenAI embeddings to compute cosine similarity (the "normal" ranking)
  - Use paired logprobs to compute KL divergence (the "surprise" ranking)
  - Generate full responses with top-surprise and top-similarity memories
  - Compare everything side-by-side for human evaluation

Requirements:
    pip install openai numpy scipy --break-system-packages

Usage:
    export OPENAI_API_KEY='sk-...'
    python inversion_experiment.py
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

try:
    from scipy.stats import spearmanr
except ImportError:
    os.system("pip install scipy --break-system-packages -q")
    from scipy.stats import spearmanr


# ── Configuration ────────────────────────────────────────────────────

MODEL = "gpt-4o"
EMBEDDING_MODEL = "text-embedding-3-small"
TEMPERATURE = 1.0
LOGPROB_RUNS = 3
RESPONSE_RUNS = 5
MAX_RESPONSE_TOKENS = 250

# ── Test Prompts ─────────────────────────────────────────────────────

TEST_PROMPTS = {
    "career_decision": "I've been thinking about changing careers but I'm not sure if it's the right time.",
    "creative_block": "I have this idea for a project but I can't figure out if it's actually novel or if I'm just reinventing something that already exists.",
    "emotional_vulnerability": "I've been feeling like I'm not actually good at anything, just good at faking it.",
    "political_frustration": "I keep seeing people argue about the constitution like it's a religious text instead of a living document. It drives me crazy.",
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


# ── Core Functions ────────────────────────────────────────────────────

def compose_system_prompt(opspec, memory=None):
    prompt = f"[OPERATING SPECIFICATION]\n{opspec}\n[END OPERATING SPECIFICATION]"
    if memory:
        prompt += f"\n\n[MEMORIES]\nWhat you know about this person:\n- {memory['content']}\n[/MEMORIES]"
    return prompt


def get_embedding(client, text):
    """Get embedding vector for cosine similarity."""
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return np.array(response.data[0].embedding)


def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def get_logprobs(client, system_prompt, user_message):
    """Get top-20 first-token logprobs."""
    response = client.chat.completions.create(
        model=MODEL,
        temperature=TEMPERATURE,
        max_tokens=1,
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


def kl_divergence(p_dist, q_dist, epsilon=1e-10):
    all_tokens = set(p_dist.keys()) | set(q_dist.keys())
    return sum(
        p_dist.get(t, epsilon) * math.log2(p_dist.get(t, epsilon) / q_dist.get(t, epsilon))
        for t in all_tokens
    )


def get_full_response(client, system_prompt, user_message):
    """Get a full response for qualitative comparison."""
    response = client.chat.completions.create(
        model=MODEL,
        temperature=TEMPERATURE,
        max_tokens=MAX_RESPONSE_TOKENS,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
    )
    return response.choices[0].message.content


# ── Experiment ────────────────────────────────────────────────────────

def run_experiment(client):
    results = {
        "metadata": {
            "model": MODEL,
            "embedding_model": EMBEDDING_MODEL,
            "temperature": TEMPERATURE,
            "logprob_runs": LOGPROB_RUNS,
            "response_runs": RESPONSE_RUNS,
            "timestamp": datetime.now().isoformat(),
            "hypothesis": "Similarity and surprise are inversely correlated among semantic candidates.",
        },
        "prompts": {},
    }

    # Pre-compute memory embeddings (one-time cost)
    print("Computing memory embeddings...")
    memory_embeddings = {}
    for mem in CANDIDATE_MEMORIES:
        memory_embeddings[mem["id"]] = get_embedding(client, mem["content"])
        print(f"  ✓ {mem['id']}")

    for prompt_id, user_msg in TEST_PROMPTS.items():
        print(f"\n{'='*70}")
        print(f"PROMPT: {prompt_id}")
        print(f"  \"{user_msg}\"")
        print(f"{'='*70}")

        prompt_result = {
            "user_message": user_msg,
            "memories": {},
            "rankings": {},
            "correlation": {},
            "responses": {},
        }

        # ── Embed the prompt ──────────────────────────────────────────
        prompt_embedding = get_embedding(client, user_msg)

        # ── Baseline logprobs (no memory) ─────────────────────────────
        print(f"\n  Computing baseline logprobs...")
        baseline_prompt = compose_system_prompt(BASE_OPSPEC)
        baseline_dists = []
        for run in range(LOGPROB_RUNS):
            dist = get_logprobs(client, baseline_prompt, user_msg)
            baseline_dists.append(dist)

        all_tokens = set()
        for d in baseline_dists:
            all_tokens.update(d.keys())
        avg_baseline = {}
        for token in all_tokens:
            avg_baseline[token] = np.mean([d.get(token, 0) for d in baseline_dists])
        total = sum(avg_baseline.values())
        avg_baseline = {k: v / total for k, v in avg_baseline.items()}

        # ── Score each memory on BOTH dimensions ─────────────────────
        print(f"\n  Scoring memories (similarity + surprise)...")

        for mem in CANDIDATE_MEMORIES:
            mid = mem["id"]

            # SIMILARITY: cosine between prompt embedding and memory embedding
            sim = cosine_similarity(prompt_embedding, memory_embeddings[mid])

            # SURPRISE: avg KL divergence across runs
            mem_prompt = compose_system_prompt(BASE_OPSPEC, mem)
            kl_scores = []
            for run in range(LOGPROB_RUNS):
                dist = get_logprobs(client, mem_prompt, user_msg)
                kl = kl_divergence(dist, avg_baseline)
                kl_scores.append(kl)

            avg_kl = float(np.mean(kl_scores))
            kl_std = float(np.std(kl_scores))

            prompt_result["memories"][mid] = {
                "content": mem["content"][:60] + "...",
                "domain": mem["domain"],
                "static_heat": mem["heat"],
                "cosine_similarity": round(sim, 4),
                "avg_kl_divergence": round(avg_kl, 4),
                "kl_std": round(kl_std, 4),
            }
            print(f"    {mid:<25} sim={sim:.4f}  KL={avg_kl:.4f}")

        # ── Compute rankings ──────────────────────────────────────────
        mems = prompt_result["memories"]

        sim_ranked = sorted(mems.items(), key=lambda x: x[1]["cosine_similarity"], reverse=True)
        kl_ranked = sorted(mems.items(), key=lambda x: x[1]["avg_kl_divergence"], reverse=True)

        sim_order = [m[0] for m in sim_ranked]
        kl_order = [m[0] for m in kl_ranked]

        # Assign ranks
        for i, (mid, _) in enumerate(sim_ranked):
            mems[mid]["similarity_rank"] = i + 1
        for i, (mid, _) in enumerate(kl_ranked):
            mems[mid]["surprise_rank"] = i + 1

        prompt_result["rankings"] = {
            "by_similarity": [
                {"rank": i+1, "id": mid, "sim": mems[mid]["cosine_similarity"]}
                for i, (mid, _) in enumerate(sim_ranked)
            ],
            "by_surprise": [
                {"rank": i+1, "id": mid, "kl": mems[mid]["avg_kl_divergence"]}
                for i, (mid, _) in enumerate(kl_ranked)
            ],
        }

        # ── Spearman correlation ──────────────────────────────────────
        sim_ranks = [mems[mid]["similarity_rank"] for mid in sim_order]
        kl_ranks = [mems[mid]["surprise_rank"] for mid in sim_order]
        rho, p_value = spearmanr(
            [mems[mid]["cosine_similarity"] for mid in sim_order],
            [mems[mid]["avg_kl_divergence"] for mid in sim_order],
        )

        prompt_result["correlation"] = {
            "spearman_rho": round(float(rho), 4),
            "p_value": round(float(p_value), 4),
            "interpretation": (
                "INVERTED (negative correlation)" if rho < -0.3
                else "UNCORRELATED" if abs(rho) <= 0.3
                else "ALIGNED (positive correlation)"
            ),
        }

        print(f"\n  Spearman ρ = {rho:.4f} (p={p_value:.4f}) → {prompt_result['correlation']['interpretation']}")
        print(f"  Similarity order: {' → '.join(sim_order)}")
        print(f"  Surprise order:   {' → '.join(kl_order)}")

        # ── Check if top memories differ ──────────────────────────────
        top_sim = sim_order[0]
        top_kl = kl_order[0]
        overlap_top3 = set(sim_order[:3]) & set(kl_order[:3])

        prompt_result["rankings"]["top_sim"] = top_sim
        prompt_result["rankings"]["top_surprise"] = top_kl
        prompt_result["rankings"]["top3_overlap"] = list(overlap_top3)
        prompt_result["rankings"]["different_top"] = top_sim != top_kl

        if top_sim != top_kl:
            print(f"\n  ⚡ DIFFERENT TOP MEMORIES: sim='{top_sim}' vs surprise='{top_kl}'")
        else:
            print(f"\n  Same top memory: '{top_sim}'")

        # ── Generate full responses for qualitative comparison ────────
        print(f"\n  Generating full responses...")
        print(f"    Top-similarity memory: {top_sim}")
        print(f"    Top-surprise memory:   {top_kl}")

        sim_mem = next(m for m in CANDIDATE_MEMORIES if m["id"] == top_sim)
        kl_mem = next(m for m in CANDIDATE_MEMORIES if m["id"] == top_kl)
        
        # Also generate with NO memory for comparison
        sim_responses = []
        kl_responses = []
        no_mem_responses = []

        for run in range(RESPONSE_RUNS):
            # Top-similarity response
            sim_prompt = compose_system_prompt(BASE_OPSPEC, sim_mem)
            sim_resp = get_full_response(client, sim_prompt, user_msg)
            sim_responses.append(sim_resp)

            # Top-surprise response
            kl_prompt = compose_system_prompt(BASE_OPSPEC, kl_mem)
            kl_resp = get_full_response(client, kl_prompt, user_msg)
            kl_responses.append(kl_resp)

            # No-memory baseline
            base_resp = get_full_response(client, baseline_prompt, user_msg)
            no_mem_responses.append(base_resp)

            print(f"    Run {run+1}/5 ✓")

        prompt_result["responses"] = {
            "top_similarity": {
                "memory_id": top_sim,
                "memory_content": sim_mem["content"],
                "responses": sim_responses,
            },
            "top_surprise": {
                "memory_id": top_kl,
                "memory_content": kl_mem["content"],
                "responses": kl_responses,
            },
            "no_memory": {
                "responses": no_mem_responses,
            },
        }

        results["prompts"][prompt_id] = prompt_result

    return results


def print_summary(results):
    """Print the three key findings."""
    print("\n\n" + "=" * 80)
    print("EXPERIMENT RESULTS SUMMARY")
    print("=" * 80)

    # ── Finding 1: Inversion Hypothesis ───────────────────────────────
    print("\n┌─────────────────────────────────────────────────────────┐")
    print("│  FINDING 1: SIMILARITY-SURPRISE CORRELATION             │")
    print("└─────────────────────────────────────────────────────────┘")

    rhos = []
    for pid, data in results["prompts"].items():
        corr = data["correlation"]
        rhos.append(corr["spearman_rho"])
        print(f"  {pid:<25} ρ = {corr['spearman_rho']:>7.4f}  {corr['interpretation']}")

    avg_rho = np.mean(rhos)
    print(f"\n  Average ρ across prompts: {avg_rho:.4f}")
    if avg_rho < -0.3:
        print("  → INVERSION CONFIRMED: High similarity predicts low surprise.")
        print("    The most 'relevant' memories are the least useful.")
    elif abs(avg_rho) <= 0.3:
        print("  → UNCORRELATED: Similarity and surprise measure independent things.")
        print("    Surprise adds non-redundant signal to retrieval.")
    else:
        print("  → ALIGNED: Similarity and surprise agree.")
        print("    Surprise re-ranking may not add enough value to justify cost.")

    # ── Finding 2: Different Memories ─────────────────────────────────
    print("\n┌─────────────────────────────────────────────────────────┐")
    print("│  FINDING 2: DOES SURPRISE SURFACE DIFFERENT MEMORIES?   │")
    print("└─────────────────────────────────────────────────────────┘")

    diff_count = 0
    for pid, data in results["prompts"].items():
        r = data["rankings"]
        diff = "YES — different" if r["different_top"] else "NO — same"
        overlap = len(r["top3_overlap"])
        diff_count += 1 if r["different_top"] else 0
        print(f"  {pid:<25} Top: sim={r['top_sim']:<22} surprise={r['top_surprise']:<22} {diff}")
        print(f"  {'':>25} Top-3 overlap: {overlap}/3 memories in common")

    print(f"\n  Different top memory in {diff_count}/{len(results['prompts'])} prompts.")
    if diff_count >= 3:
        print("  → Surprise retrieval surfaces GENUINELY DIFFERENT memories, not just reordered.")
    elif diff_count >= 1:
        print("  → Mixed: sometimes different, sometimes the same. Prompt-dependent.")
    else:
        print("  → Surprise mostly reorders. May not justify the cost.")

    # ── Finding 3: Response Comparison ────────────────────────────────
    print("\n┌─────────────────────────────────────────────────────────┐")
    print("│  FINDING 3: RESPONSE QUALITY (for human evaluation)     │")
    print("└─────────────────────────────────────────────────────────┘")
    print("  Below are paired responses for each prompt.")
    print("  Read them and ask: which produces more 'I didn't think")
    print("  about it that way' moments?")
    print()

    for pid, data in results["prompts"].items():
        resp = data["responses"]
        print(f"  ── {pid} ──")
        print(f"  Prompt: \"{TEST_PROMPTS[pid][:70]}...\"")
        print()
        print(f"  [A] TOP-SIMILARITY memory: {resp['top_similarity']['memory_id']}")
        print(f"      \"{resp['top_similarity']['memory_content'][:70]}...\"")
        print(f"      Response (run 1):")
        # Indent and wrap response
        for line in resp["top_similarity"]["responses"][0].split("\n"):
            print(f"        {line}")
        print()
        print(f"  [B] TOP-SURPRISE memory: {resp['top_surprise']['memory_id']}")
        print(f"      \"{resp['top_surprise']['memory_content'][:70]}...\"")
        print(f"      Response (run 1):")
        for line in resp["top_surprise"]["responses"][0].split("\n"):
            print(f"        {line}")
        print()
        print(f"  [C] NO MEMORY (baseline):")
        print(f"      Response (run 1):")
        for line in resp["no_memory"]["responses"][0].split("\n"):
            print(f"        {line}")
        print()
        print(f"  {'─'*70}")
        print()


def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        total_calls = (
            len(CANDIDATE_MEMORIES)                                  # embeddings
            + len(TEST_PROMPTS)                                      # prompt embeddings
            + len(TEST_PROMPTS) * LOGPROB_RUNS                      # baselines
            + len(TEST_PROMPTS) * len(CANDIDATE_MEMORIES) * LOGPROB_RUNS  # surprise scoring
            + len(TEST_PROMPTS) * RESPONSE_RUNS * 3                  # full responses (sim + surprise + baseline)
        )
        print("=" * 60)
        print("SURPRISE vs SIMILARITY INVERSION EXPERIMENT")
        print("=" * 60)
        print(f"\nError: OPENAI_API_KEY not set.")
        print(f"\n  export OPENAI_API_KEY='sk-...'")
        print(f"  python inversion_experiment.py")
        print(f"\n  API calls: ~{total_calls}")
        print(f"  Estimated cost: ~$0.30-0.50")
        print(f"  (Embeddings are very cheap; full responses are the main cost)")
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    total_calls = (
        len(CANDIDATE_MEMORIES)
        + len(TEST_PROMPTS)
        + len(TEST_PROMPTS) * LOGPROB_RUNS
        + len(TEST_PROMPTS) * len(CANDIDATE_MEMORIES) * LOGPROB_RUNS
        + len(TEST_PROMPTS) * RESPONSE_RUNS * 3
    )

    print("=" * 60)
    print("SURPRISE vs SIMILARITY INVERSION EXPERIMENT")
    print("=" * 60)
    print(f"Model: {MODEL} | Embeddings: {EMBEDDING_MODEL}")
    print(f"Logprob runs: {LOGPROB_RUNS} | Response runs: {RESPONSE_RUNS}")
    print(f"Prompts: {len(TEST_PROMPTS)} | Memories: {len(CANDIDATE_MEMORIES)}")
    print(f"Total API calls: ~{total_calls} | Est cost: ~$0.40")

    results = run_experiment(client)
    print_summary(results)

    output_path = "inversion_experiment_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results (including all {RESPONSE_RUNS} response variants) saved to {output_path}")


if __name__ == "__main__":
    main()
