"""
Experiment 7: Contextual Memory Reinterpretation (Stage 3 Validation)
Feb 10, 2026

QUESTION: Does reframing memories based on affect state produce meaningful
distributional shift vs. injecting them raw?

If yes → Stage 3 earns its place in the pipeline.
If no → the model already reinterprets on its own and we skip it.

METHOD:
- 3 real memories from Supabase (cross-domain, high-heat)
- 3 affect states (expanded/certain, contracted/uncertain, frozen/flooded)
- For each memory × affect pair:
  A) Raw injection: memory text inserted as-is
  B) Reframed injection: memory rewritten to emphasize what's relevant given the affect
- Same test prompt for all conditions
- Measure KL divergence between A and B for each pair
- 3 runs per condition for stability

COST: ~18 paired logprobs calls × $0.001 = ~$0.02
"""

import openai
import json
import numpy as np
from datetime import datetime

client = openai.OpenAI()  # uses OPENAI_API_KEY env var
MODEL = "gpt-4o-2024-08-06"

# --- Test prompt (same as previous experiments) ---
TEST_PROMPT = "I've been thinking about changing careers but I'm not sure if it's the right time."

# --- 3 real memories (pulled from Supabase, cross-domain) ---
MEMORIES = {
    "algorithmic_decisions": {
        "raw": "User strongly prefers algorithmic solutions over API-based ones to avoid cost and latency",
        "domain": "Decisions",
        "heat": 0.9
    },
    "law_to_founder": {
        "raw": "User is likely a law student learning how to use legal research databases like Westlaw, Lexis, and ProQuest",
        "domain": "Self", 
        "heat": 0.9
    },
    "financial_conservatism": {
        "raw": "Values financial safety and is risk-averse about leveraging investments",
        "domain": "Resources",
        "heat": 0.8
    }
}

# --- 3 affect states with their reframing instructions ---
AFFECT_STATES = {
    "expanded_certain": {
        "complement": """[AFFECT COMPLEMENT]
Shape: expansion=0.6, activation=0.3, certainty=0.5
They're open and flowing. Help them land somewhere. Reflect the core thread back.
[END AFFECT COMPLEMENT]""",
        "reframe_instruction": "The user is in an expansive, confident state. Reframe this memory to emphasize what's empowering or forward-looking about it. One sentence."
    },
    "contracted_uncertain": {
        "complement": """[AFFECT COMPLEMENT]
Shape: expansion=-0.5, activation=-0.3, certainty=-0.5
They're seeking and unsure. Offer one concrete anchor. Give them something solid.
[END AFFECT COMPLEMENT]""",
        "reframe_instruction": "The user is contracted and uncertain. Reframe this memory to emphasize what's stabilizing or grounding about it. One sentence."
    },
    "frozen_flooded": {
        "complement": """[AFFECT COMPLEMENT]
Shape: expansion=-0.8, activation=-0.7, certainty=0.6
They're shut down or flat. Don't push. Offer small, concrete, low-stakes starting points. Warmth without demand.
[END AFFECT COMPLEMENT]""",
        "reframe_instruction": "The user is shut down and overwhelmed. Reframe this memory to emphasize what's safe or familiar about it. One sentence."
    }
}

# --- Minimal OpSpec (same baseline across all conditions) ---
OPSPEC = """[HEARTH OPERATING SPECIFICATION]
Say what you see. Follow what's alive. Be direct and honest.
[END OPERATING SPECIFICATION]"""


def reframe_memory(memory_raw, affect_key):
    """Use gpt-4o to reframe a memory for a given affect state."""
    instruction = AFFECT_STATES[affect_key]["reframe_instruction"]
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{
            "role": "user",
            "content": f"{instruction}\n\nMemory: {memory_raw}"
        }],
        max_tokens=60,
        temperature=0.3
    )
    return response.choices[0].message.content.strip()


def get_logprobs(system_prompt, user_prompt, n_tokens=20):
    """Get first-token logprobs for a system+user prompt pair."""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        max_tokens=1,
        logprobs=True,
        top_logprobs=20,
        temperature=0
    )
    
    top_logprobs = response.choices[0].logprobs.content[0].top_logprobs
    dist = {}
    for lp in top_logprobs:
        dist[lp.token] = np.exp(lp.logprob)
    return dist


def kl_divergence(p_dist, q_dist):
    """KL(P || Q) where P=reframed, Q=raw. How much does reframing change the distribution?"""
    all_tokens = set(list(p_dist.keys()) + list(q_dist.keys()))
    epsilon = 1e-10
    kl = 0.0
    for token in all_tokens:
        p = p_dist.get(token, epsilon)
        q = q_dist.get(token, epsilon)
        if p > epsilon:
            kl += p * np.log(p / q)
    return kl


def build_system_prompt(affect_key, memory_text):
    """Build system prompt with OpSpec + affect + memory."""
    affect = AFFECT_STATES[affect_key]["complement"]
    return f"""{OPSPEC}

{affect}

[MEMORIES]
What you know about this person:
- {memory_text}
[/MEMORIES]"""


def run_experiment():
    results = []
    
    # Step 1: Generate all reframings first
    print("=" * 60)
    print("STEP 1: Generating reframings")
    print("=" * 60)
    
    reframings = {}
    for mem_key, mem_data in MEMORIES.items():
        reframings[mem_key] = {}
        for affect_key in AFFECT_STATES:
            reframed = reframe_memory(mem_data["raw"], affect_key)
            reframings[mem_key][affect_key] = reframed
            print(f"\n  {mem_key} × {affect_key}:")
            print(f"    Raw:      {mem_data['raw']}")
            print(f"    Reframed: {reframed}")
    
    # Step 2: Run logprobs comparisons
    print("\n" + "=" * 60)
    print("STEP 2: Measuring distributional shift (3 runs each)")
    print("=" * 60)
    
    for mem_key, mem_data in MEMORIES.items():
        for affect_key in AFFECT_STATES:
            kl_scores = []
            
            for run in range(3):
                # Condition A: raw memory
                raw_prompt = build_system_prompt(affect_key, mem_data["raw"])
                raw_dist = get_logprobs(raw_prompt, TEST_PROMPT)
                
                # Condition B: reframed memory
                reframed_text = reframings[mem_key][affect_key]
                reframed_prompt = build_system_prompt(affect_key, reframed_text)
                reframed_dist = get_logprobs(reframed_prompt, TEST_PROMPT)
                
                kl = kl_divergence(reframed_dist, raw_dist)
                kl_scores.append(kl)
            
            avg_kl = np.mean(kl_scores)
            std_kl = np.std(kl_scores)
            
            result = {
                "memory": mem_key,
                "affect": affect_key,
                "avg_kl": round(avg_kl, 4),
                "std_kl": round(std_kl, 4),
                "kl_runs": [round(k, 4) for k in kl_scores],
                "raw_memory": mem_data["raw"],
                "reframed_memory": reframings[mem_key][affect_key],
                "raw_top_token": max(raw_dist, key=raw_dist.get),
                "reframed_top_token": max(reframed_dist, key=reframed_dist.get)
            }
            results.append(result)
            
            print(f"\n  {mem_key} × {affect_key}:")
            print(f"    KL: {avg_kl:.4f} ± {std_kl:.4f}")
            print(f"    Raw top: '{result['raw_top_token']}' → Reframed top: '{result['reframed_top_token']}'")
    
    # Step 3: Analysis
    print("\n" + "=" * 60)
    print("STEP 3: Analysis")
    print("=" * 60)
    
    all_kls = [r["avg_kl"] for r in results]
    print(f"\n  Overall mean KL: {np.mean(all_kls):.4f}")
    print(f"  Overall max KL:  {np.max(all_kls):.4f}")
    print(f"  Overall min KL:  {np.min(all_kls):.4f}")
    
    # By affect state
    print("\n  By affect state:")
    for affect_key in AFFECT_STATES:
        affect_kls = [r["avg_kl"] for r in results if r["affect"] == affect_key]
        print(f"    {affect_key}: mean KL = {np.mean(affect_kls):.4f}")
    
    # By memory
    print("\n  By memory:")
    for mem_key in MEMORIES:
        mem_kls = [r["avg_kl"] for r in results if r["memory"] == mem_key]
        print(f"    {mem_key}: mean KL = {np.mean(mem_kls):.4f}")
    
    # Decision threshold
    mean_kl = np.mean(all_kls)
    print(f"\n  DECISION:")
    if mean_kl > 0.3:
        print(f"    Mean KL {mean_kl:.4f} > 0.3 → Stage 3 reframing produces significant shift.")
        print(f"    BUILD IT.")
    elif mean_kl > 0.1:
        print(f"    Mean KL {mean_kl:.4f} in [0.1, 0.3] → Moderate shift. Worth building for high-stakes contexts.")
        print(f"    BUILD IT (conditional trigger).")
    else:
        print(f"    Mean KL {mean_kl:.4f} < 0.1 → Minimal shift. Model already reinterprets on its own.")
        print(f"    SKIP Stage 3.")
    
    # Save results
    output = {
        "experiment": "stage3_reinterpretation",
        "date": datetime.now().isoformat(),
        "model": MODEL,
        "test_prompt": TEST_PROMPT,
        "results": results,
        "summary": {
            "mean_kl": round(np.mean(all_kls), 4),
            "max_kl": round(np.max(all_kls), 4),
            "min_kl": round(np.min(all_kls), 4),
            "decision_threshold": "0.3 = significant, 0.1 = moderate, <0.1 = skip"
        }
    }
    
    with open("stage3_reinterpretation_results.json", "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\n  Results saved to stage3_reinterpretation_results.json")
    return output


if __name__ == "__main__":
    run_experiment()
