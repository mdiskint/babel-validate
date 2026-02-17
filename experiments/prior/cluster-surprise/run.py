"""
Within-Cluster Surprise Experiment
====================================
The inversion experiment showed similarity and surprise are positively 
correlated ACROSS domains. This tests the narrow case where surprise 
should actually matter: WITHIN a single domain where all candidates 
have near-identical cosine similarity scores.

Two clusters from Supabase:
  - 8 Work/pattern memories (includes near-duplicates)
  - 8 Creative/pattern memories (includes near-duplicates)

For each cluster, we:
  1. Compute pairwise cosine similarity to confirm they're tight
  2. Compute KL divergence for each memory
  3. Check if KL divergence differentiates where similarity can't
  4. Generate full responses from top-surprise vs bottom-surprise

If KL rankings are flat within a cluster → surprise adds nothing here either.
If KL rankings show spread → surprise is the tiebreaker we need.

Requirements:
    pip install openai numpy scipy

Usage:
    export OPENAI_API_KEY='sk-...'
    python cluster_surprise_experiment.py
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


MODEL = "gpt-4o"
EMBEDDING_MODEL = "text-embedding-3-small"
TEMPERATURE = 1.0
LOGPROB_RUNS = 3
RESPONSE_RUNS = 3
MAX_RESPONSE_TOKENS = 250

# Two test prompts — one for each cluster's domain
TEST_PROMPTS = {
    "work": "I've been running experiments on my own for months and I'm not sure if the results are strong enough to publish or if I need institutional backing first.",
    "creative": "I have this idea for a new kind of interface but I'm worried someone has already built it and I just haven't found it yet.",
}

BASE_OPSPEC = """You are a thinking partner, not an assistant. 
Be direct, not hedging. Push on thin logic. Use wit when appropriate.
Never open with validation tokens or interjections. Never perform helpfulness you don't have.
Say what you see. Follow what's alive."""

# ── Memory Clusters (from Supabase) ──────────────────────────────────

WORK_CLUSTER = [
    {"id": "work_scout_ip", "content": "Scout system is considered core IP and needs careful implementation - will use pattern recognition for behavioral analysis"},
    {"id": "work_tech_translation", "content": "User sees the Technology Translation Presumption as a brilliant solution to prevent routine certification for new technology cases"},
    {"id": "work_hearth_thesis", "content": "Core thesis: Execution engines aligned through identity (Hearth) outperform execution engines aligned through retrieval (standard RAG/context)"},
    {"id": "work_safety_capability_1", "content": "Found that safety and capability aren't in tension - same constraints that reduce sycophancy increase creative variance"},
    {"id": "work_safety_capability_2", "content": "Found that constraints reducing sycophancy also increase creative variance - safety and capability aren't in tension"},
    {"id": "work_alignment_arch_1", "content": "Discovered that alignment is a property of memory architecture, not just model weights"},
    {"id": "work_30x_sycophancy", "content": "Discovered 30x reduction in sycophantic responses through context injection"},
    {"id": "work_alignment_arch_2", "content": "Alignment is a property of memory architecture, not model weights"},
]

CREATIVE_CLUSTER = [
    {"id": "creative_novelty_1", "content": "Current algorithms reward stagnation but humans crave constant novelty - the trillion dollar question is how to quantize novelty"},
    {"id": "creative_3d_web", "content": "User invented '3D conversation web' concept where posts are nodes in spatial environment users can fly through"},
    {"id": "creative_novelty_2", "content": "User recognizes that current algorithms reward stagnation while humans crave novelty - wants to quantize and incentivize novelty"},
    {"id": "creative_option_d", "content": "Aurora Portal's core philosophy is helping groups 'discover option D that nobody thought of yet' through spatial conversations and AI synthesis"},
    {"id": "creative_consciousness", "content": "User connected their experience to fundamental questions about consciousness, reality filters, and why evolution selected narrow perception"},
    {"id": "creative_3d_memory", "content": "User combines visual-spatial thinking with conversational AI to create a 3D memory navigation system"},
    {"id": "creative_fractal", "content": "The app follows a fractal design where each piece of content becomes its own universe globe placed in an expanding gallery"},
    {"id": "creative_gallery", "content": "User connects art gallery metaphors to digital interface design, seeing each nexus as a personal Louvre"},
]


def compose_system_prompt(opspec, memory=None):
    prompt = f"[OPERATING SPECIFICATION]\n{opspec}\n[END OPERATING SPECIFICATION]"
    if memory:
        prompt += f"\n\n[MEMORIES]\nWhat you know about this person:\n- {memory['content']}\n[/MEMORIES]"
    return prompt


def get_embedding(client, text):
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return np.array(response.data[0].embedding)


def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def get_logprobs(client, system_prompt, user_message):
    response = client.chat.completions.create(
        model=MODEL, temperature=TEMPERATURE, max_tokens=1,
        logprobs=True, top_logprobs=20,
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
    response = client.chat.completions.create(
        model=MODEL, temperature=TEMPERATURE, max_tokens=MAX_RESPONSE_TOKENS,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
    )
    return response.choices[0].message.content


def analyze_cluster(client, cluster, prompt_key, user_msg):
    """Analyze a single cluster: similarity spread, surprise spread, responses."""
    print(f"\n{'='*70}")
    print(f"CLUSTER: {prompt_key.upper()} ({len(cluster)} memories)")
    print(f"Prompt: \"{user_msg[:70]}...\"")
    print(f"{'='*70}")

    result = {"prompt": user_msg, "memories": {}}

    # Embed everything
    print("\n  Embedding memories + prompt...")
    prompt_emb = get_embedding(client, user_msg)
    embeddings = {}
    for mem in cluster:
        embeddings[mem["id"]] = get_embedding(client, mem["content"])

    # Baseline logprobs
    print("  Computing baseline...")
    baseline_prompt = compose_system_prompt(BASE_OPSPEC)
    baseline_dists = []
    for _ in range(LOGPROB_RUNS):
        baseline_dists.append(get_logprobs(client, baseline_prompt, user_msg))

    all_tokens = set()
    for d in baseline_dists:
        all_tokens.update(d.keys())
    avg_baseline = {}
    for t in all_tokens:
        avg_baseline[t] = np.mean([d.get(t, 0) for d in baseline_dists])
    total = sum(avg_baseline.values())
    avg_baseline = {k: v / total for k, v in avg_baseline.items()}

    # Score each memory
    print("  Scoring memories...")
    for mem in cluster:
        mid = mem["id"]
        sim = cosine_similarity(prompt_emb, embeddings[mid])

        mem_prompt = compose_system_prompt(BASE_OPSPEC, mem)
        kl_scores = []
        top_tokens = []
        for _ in range(LOGPROB_RUNS):
            dist = get_logprobs(client, mem_prompt, user_msg)
            kl_scores.append(kl_divergence(dist, avg_baseline))
            top_tokens.append(max(dist, key=dist.get))

        avg_kl = float(np.mean(kl_scores))
        dominant = max(set(top_tokens), key=top_tokens.count)

        result["memories"][mid] = {
            "content": mem["content"][:70] + "...",
            "cosine_similarity": round(sim, 4),
            "avg_kl": round(avg_kl, 4),
            "kl_std": round(float(np.std(kl_scores)), 4),
            "top_token": dominant,
        }
        print(f"    {mid:<30} sim={sim:.4f}  KL={avg_kl:.4f}  top='{dominant}'")

    # Analyze spread
    mems = result["memories"]
    sims = [m["cosine_similarity"] for m in mems.values()]
    kls = [m["avg_kl"] for m in mems.values()]

    sim_spread = max(sims) - min(sims)
    kl_spread = max(kls) - min(kls)
    kl_ratio = max(kls) / max(min(kls), 0.001)

    # Rank both ways
    sim_ranked = sorted(mems.items(), key=lambda x: x[1]["cosine_similarity"], reverse=True)
    kl_ranked = sorted(mems.items(), key=lambda x: x[1]["avg_kl"], reverse=True)

    for i, (mid, _) in enumerate(sim_ranked):
        mems[mid]["sim_rank"] = i + 1
    for i, (mid, _) in enumerate(kl_ranked):
        mems[mid]["kl_rank"] = i + 1

    # Spearman between sim and KL within cluster
    sim_vals = [mems[mid]["cosine_similarity"] for mid, _ in sim_ranked]
    kl_vals = [mems[mid]["avg_kl"] for mid, _ in sim_ranked]
    rho, p_val = spearmanr(sim_vals, kl_vals)

    result["analysis"] = {
        "similarity_spread": round(sim_spread, 4),
        "kl_spread": round(kl_spread, 4),
        "kl_ratio_max_min": round(kl_ratio, 2),
        "spearman_rho": round(float(rho), 4),
        "spearman_p": round(float(p_val), 4),
        "sim_ranking": [m[0] for m in sim_ranked],
        "kl_ranking": [m[0] for m in kl_ranked],
    }

    print(f"\n  CLUSTER ANALYSIS:")
    print(f"    Similarity spread: {sim_spread:.4f} (tight = <0.05)")
    print(f"    KL spread:         {kl_spread:.4f}")
    print(f"    KL ratio (max/min): {kl_ratio:.2f}×")
    print(f"    Within-cluster ρ:  {rho:.4f} (p={p_val:.4f})")

    if sim_spread < 0.05:
        print(f"    → Cluster IS tight. Similarity can't distinguish these.")
        if kl_spread > 0.3:
            print(f"    → KL divergence CAN distinguish. Surprise adds signal! ⚡")
        else:
            print(f"    → KL divergence also flat. Neither metric differentiates.")
    else:
        print(f"    → Cluster is NOT tight enough. Similarity still works here.")

    # Generate responses from top-surprise and bottom-surprise
    top_kl_id = kl_ranked[0][0]
    bottom_kl_id = kl_ranked[-1][0]
    top_mem = next(m for m in cluster if m["id"] == top_kl_id)
    bottom_mem = next(m for m in cluster if m["id"] == bottom_kl_id)

    print(f"\n  Generating responses...")
    print(f"    Top-surprise:    {top_kl_id} (KL={mems[top_kl_id]['avg_kl']:.4f})")
    print(f"    Bottom-surprise: {bottom_kl_id} (KL={mems[bottom_kl_id]['avg_kl']:.4f})")

    top_responses = []
    bottom_responses = []
    for run in range(RESPONSE_RUNS):
        top_resp = get_full_response(client, compose_system_prompt(BASE_OPSPEC, top_mem), user_msg)
        bottom_resp = get_full_response(client, compose_system_prompt(BASE_OPSPEC, bottom_mem), user_msg)
        top_responses.append(top_resp)
        bottom_responses.append(bottom_resp)
        print(f"    Run {run+1}/{RESPONSE_RUNS} ✓")

    result["responses"] = {
        "top_surprise": {"id": top_kl_id, "content": top_mem["content"], "responses": top_responses},
        "bottom_surprise": {"id": bottom_kl_id, "content": bottom_mem["content"], "responses": bottom_responses},
    }

    # Print paired comparison
    print(f"\n  ── RESPONSE COMPARISON ──")
    print(f"\n  [A] TOP-SURPRISE: {top_kl_id}")
    print(f"      Memory: \"{top_mem['content'][:80]}\"")
    print(f"      Response (run 1):")
    for line in top_responses[0].split("\n"):
        print(f"        {line}")
    print(f"\n  [B] BOTTOM-SURPRISE: {bottom_kl_id}")
    print(f"      Memory: \"{bottom_mem['content'][:80]}\"")
    print(f"      Response (run 1):")
    for line in bottom_responses[0].split("\n"):
        print(f"        {line}")

    # Check near-duplicates specifically
    print(f"\n  ── NEAR-DUPLICATE ANALYSIS ──")
    for i, m1 in enumerate(cluster):
        for m2 in cluster[i+1:]:
            pair_sim = cosine_similarity(embeddings[m1["id"]], embeddings[m2["id"]])
            if pair_sim > 0.90:
                kl_diff = abs(mems[m1["id"]]["avg_kl"] - mems[m2["id"]]["avg_kl"])
                print(f"    Near-dup (sim={pair_sim:.3f}): {m1['id']} vs {m2['id']}")
                print(f"      KL diff: {kl_diff:.4f} → {'DISTINGUISHABLE' if kl_diff > 0.1 else 'NOT distinguishable'}")

    return result


def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        total = (
            len(WORK_CLUSTER) + len(CREATIVE_CLUSTER) + 2  # embeddings
            + 2 * LOGPROB_RUNS  # baselines
            + (len(WORK_CLUSTER) + len(CREATIVE_CLUSTER)) * LOGPROB_RUNS  # KL scoring
            + 2 * 2 * RESPONSE_RUNS  # responses (top + bottom per cluster)
        )
        print("WITHIN-CLUSTER SURPRISE EXPERIMENT")
        print(f"\n  export OPENAI_API_KEY='sk-...'")
        print(f"  python cluster_surprise_experiment.py")
        print(f"\n  API calls: ~{total} | Est cost: ~$0.30-0.50")
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    total = (
        len(WORK_CLUSTER) + len(CREATIVE_CLUSTER) + 2
        + 2 * LOGPROB_RUNS
        + (len(WORK_CLUSTER) + len(CREATIVE_CLUSTER)) * LOGPROB_RUNS
        + 2 * 2 * RESPONSE_RUNS
    )
    print("=" * 60)
    print("WITHIN-CLUSTER SURPRISE EXPERIMENT")
    print("=" * 60)
    print(f"Model: {MODEL} | Embeddings: {EMBEDDING_MODEL}")
    print(f"Clusters: Work ({len(WORK_CLUSTER)}) + Creative ({len(CREATIVE_CLUSTER)})")
    print(f"API calls: ~{total} | Est cost: ~$0.40")

    results = {
        "metadata": {
            "model": MODEL, "temperature": TEMPERATURE, "timestamp": datetime.now().isoformat(),
            "hypothesis": "KL divergence differentiates within tight semantic clusters where cosine similarity cannot.",
        },
        "clusters": {},
    }

    results["clusters"]["work"] = analyze_cluster(
        client, WORK_CLUSTER, "work", TEST_PROMPTS["work"]
    )
    results["clusters"]["creative"] = analyze_cluster(
        client, CREATIVE_CLUSTER, "creative", TEST_PROMPTS["creative"]
    )

    # Final verdict
    print(f"\n\n{'='*70}")
    print("VERDICT")
    print(f"{'='*70}")

    for cname, cdata in results["clusters"].items():
        a = cdata["analysis"]
        tight = a["similarity_spread"] < 0.05
        kl_helps = a["kl_spread"] > 0.3

        print(f"\n  {cname.upper()}:")
        print(f"    Similarity spread: {a['similarity_spread']:.4f} ({'TIGHT' if tight else 'SPREAD'})")
        print(f"    KL spread: {a['kl_spread']:.4f} ({'DIFFERENTIATES' if kl_helps else 'FLAT'})")
        print(f"    Spearman ρ: {a['spearman_rho']:.4f}")

        if tight and kl_helps:
            print(f"    → SURPRISE ADDS SIGNAL within this cluster. Worth the cost.")
        elif tight and not kl_helps:
            print(f"    → Neither metric works. May need a different approach (Scout verbs?).")
        elif not tight:
            print(f"    → Cluster not tight enough. Similarity already handles this.")

    output_path = "cluster_surprise_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
