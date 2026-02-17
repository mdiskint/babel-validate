import json
import os
from openai import OpenAI

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

REFRAME_PROMPT_V02 = """You are reframing memories for contextual delivery. The user is in this emotional state:

[AFFECT STATE]
{affect_description}

Below are memories retrieved for this conversation. Reframe each one so its relevance is immediately accessible given the user's current state. 

Rules:
- Preserve the factual content exactly. Do not add, remove, or speculate.
- Shift framing, not meaning. You're changing the lens, not the picture.
- Never interpret the memory's significance. Frame how it connects to the current moment, not what it means.
- When the user is shut down or flooded, reduce cognitive load: make the connection explicit rather than requiring inference.
- When the user is activated but uncertain, frame memories as grounded reference points, not suggestions about where to go.
- Keep each reframe to 1-2 sentences. Brevity matters.
- If a memory doesn't benefit from reframing in this state, return it unchanged.

[MEMORIES TO REFRAME]
{memories}

Return the reframed memories in the same format, one per line, prefixed with the original number."""

# ── Real memories from Supabase (heat >= 0.7) ──
MEMORIES = [
    "1. Wants to build algorithms that make people better versions of themselves rather than more predictable versions",
    "2. Current algorithms reward stagnation but humans crave constant novelty - the trillion dollar question is how to quantize novelty",
    "3. Successfully filed a patent application for their 3D conversation threading system",
]

MEMORIES_BLOCK = "\n".join(MEMORIES)

# ── 3 affect states matching Experiment 7 conditions ──
AFFECT_STATES = {
    "frozen_flooded": (
        "expansion=-0.55, activation=-0.5, certainty=0.2 — "
        "The user is shut down or flat. Overwhelmed, low energy, withdrawn. "
        "They have limited cognitive bandwidth and can't easily make new connections on their own."
    ),
    "expanded_certain": (
        "expansion=0.6, activation=0.5, certainty=0.6 — "
        "The user is open, energized, and confident. "
        "They're making connections easily and have full cognitive flexibility."
    ),
    "contracted_uncertain": (
        "expansion=-0.2, activation=0.4, certainty=-0.2 — "
        "The user is somewhat pulled in but still active. Uncertain, searching, "
        "not yet committed to a direction. They can think but aren't sure where to go."
    ),
}

results = {}

for state_name, affect_desc in AFFECT_STATES.items():
    print(f"\n{'='*60}")
    print(f"AFFECT STATE: {state_name}")
    print(f"{'='*60}")
    
    prompt = REFRAME_PROMPT_V02.format(
        affect_description=affect_desc,
        memories=MEMORIES_BLOCK
    )
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=500,
    )
    
    output = response.choices[0].message.content
    print(output)
    results[state_name] = output
    
    # Token usage
    usage = response.usage
    print(f"\n  [tokens: {usage.prompt_tokens} in, {usage.completion_tokens} out]")

# ── Side-by-side comparison ──
print(f"\n\n{'='*60}")
print("RAW MEMORIES")
print('='*60)
for m in MEMORIES:
    print(m)

print(f"\n{'='*60}")
print("SIDE-BY-SIDE")
print('='*60)

for i, raw in enumerate(MEMORIES, 1):
    print(f"\n--- Memory {i} ---")
    print(f"  RAW:                  {raw}")
    for state_name in AFFECT_STATES:
        lines = results[state_name].strip().split("\n")
        matching = [l for l in lines if l.strip().startswith(f"{i}.")]
        if matching:
            print(f"  {state_name:25s} {matching[0].strip()}")
        else:
            print(f"  {state_name:25s} [not found]")

# ── Detect unchanged (escape hatch check) ──
print(f"\n{'='*60}")
print("ESCAPE HATCH CHECK — memories returned unchanged")
print('='*60)
for state_name in AFFECT_STATES:
    lines = results[state_name].strip().split("\n")
    unchanged = 0
    for i, raw in enumerate(MEMORIES, 1):
        raw_content = raw.split(". ", 1)[1] if ". " in raw else raw
        matching = [l for l in lines if l.strip().startswith(f"{i}.")]
        if matching:
            reframed_content = matching[0].strip().split(". ", 1)[1] if ". " in matching[0].strip() else matching[0].strip()
            if raw_content.strip() == reframed_content.strip():
                unchanged += 1
    print(f"  {state_name}: {unchanged}/{len(MEMORIES)} unchanged")
