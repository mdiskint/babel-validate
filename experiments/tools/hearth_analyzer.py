"""
Hearth Conversation Analyzer v0.3
─────────────────────────────────
Changes from v0.2:
  - Exponential decay: recent messages matter more, but a single decisive
    message can't overwrite five turns of exploration. Configurable half-life.
  - Momentum: tracks the DIRECTION things are moving, not just current state.
    "Certainty rising" and "certainty high" are different situations.
  - Trajectory blending: final values blend current-message reading with
    windowed trajectory. Ratio is tunable (default 40% current / 60% trajectory).
  - Affect complement text generation now includes behavioral instructions
    matched to the computed values (the conditional blocks from the real OpSpec).
"""

import re
import math
from dataclasses import dataclass, field
from typing import Optional


# ── Word Lists (unchanged from v0.2) ─────────────────────────────

HEDGE_WORDS = {
    "maybe", "perhaps", "possibly", "might", "could be", "i think",
    "i guess", "not sure", "kind of", "sort of", "i wonder",
    "probably", "seems like", "feels like", "idk", "dunno",
    "honestly", "tbh", "i mean"
}

CERTAINTY_WORDS = {
    "definitely", "absolutely", "exactly", "clearly", "obviously",
    "certainly", "for sure", "no doubt", "let's do it", "let's go",
    "yes", "right", "correct", "precisely", "totally",
    "lets do it", "lets go", "do it", "go for it", "ship it"
}

EXPANSION_WORDS = {
    "what if", "what about", "could we", "imagine", "explore",
    "interesting", "curious", "idea", "wonder", "possibility",
    "another way", "alternative", "new", "different", "expand",
    "how about", "have you considered", "tangent", "rabbit hole"
}

CONTRACTION_WORDS = {
    "just", "only", "but", "however", "can't", "won't", "don't",
    "never", "stop", "enough", "too much", "overwhelmed", "confused",
    "lost", "stuck", "frustrated", "tired", "forget it", "nvm",
    "nevermind"
}

CONCRETE_TOPIC_WORDS = {
    "code", "script", "python", "javascript", "api", "database",
    "server", "file", "function", "replit", "github", "terminal",
    "command", "json", "html", "css", "deploy", "docker", "sql"
}

ABSTRACT_TOPIC_WORDS = {
    "concept", "theory", "philosophy", "meaning", "framework",
    "paradigm", "vision", "principle", "fundamental", "essence",
    "nature", "consciousness", "emergence", "alignment", "coherence",
    "ontology", "epistemology", "metaphor"
}

CONCRETE_MODE_WORDS = {
    "step", "first", "then", "next", "specific", "exactly",
    "how do we", "let's", "lets", "build", "create", "make",
    "test", "run", "try", "implement", "add", "fix", "change",
    "update", "show me", "give me"
}

ABSTRACT_MODE_WORDS = {
    "think about", "consider", "approach", "strategy", "in general",
    "broadly", "conceptually", "theoretically", "philosophically",
    "what does it mean", "why does", "how come", "underlying"
}

ENGAGEMENT_POSITIVE = {
    "yes", "yeah", "yep", "right", "exactly", "good", "great",
    "nice", "cool", "awesome", "perfect", "love it", "love that",
    "brilliant", "oh", "ooh", "aha", "interesting", "wow",
    "that makes sense", "keep going", "more", "tell me more",
    "go on", "and then"
}

ENGAGEMENT_NEGATIVE = {
    "ok", "okay", "sure", "fine", "whatever", "i guess", "meh",
    "hmm", "hm", "uh", "um", "idk", "dunno", "nvm", "nevermind",
    "forget it", "anyway", "moving on"
}


# ── Utilities ─────────────────────────────────────────────────────

def clamp(value: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def count_matches(text: str, word_set: set) -> int:
    lower = text.lower()
    return sum(1 for w in word_set if w in lower)


# ── v0.3: Decay & Trajectory Engine ──────────────────────────────

def exponential_weights(n: int, half_life: float = 3.0) -> list[float]:
    """
    Generate weights for n items where most recent = highest weight.
    half_life controls how fast older messages fade.
    
    half_life=3 means a message 3 turns ago has half the weight of current.
    half_life=1 would be very aggressive recency bias.
    half_life=10 would weight history almost equally.
    
    Think of it like memory — you remember yesterday clearly, last week
    vaguely, last month barely. half_life=3 is roughly that curve for
    a conversation.
    """
    if n == 0:
        return []
    decay = math.log(2) / half_life
    raw = [math.exp(-decay * (n - 1 - i)) for i in range(n)]
    total = sum(raw)
    return [w / total for w in raw]


def weighted_average(values: list[float], weights: list[float]) -> float:
    """Compute weighted average, handling length mismatches."""
    n = min(len(values), len(weights))
    if n == 0:
        return 0.0
    return sum(v * w for v, w in zip(values[-n:], weights[-n:]))


def compute_momentum(values: list[float], window: int = 4) -> float:
    """
    Momentum: is the value RISING or FALLING?
    
    Returns roughly -1 to 1.
    Positive = trending up. Negative = trending down. ~0 = stable.
    
    Uses linear regression slope over the window, normalized.
    """
    recent = values[-window:]
    n = len(recent)
    if n < 2:
        return 0.0

    # Simple linear regression slope
    x_mean = (n - 1) / 2
    y_mean = sum(recent) / n
    numerator = sum((i - x_mean) * (y - y_mean) for i, y in enumerate(recent))
    denominator = sum((i - x_mean) ** 2 for i in range(n))

    if denominator == 0:
        return 0.0

    slope = numerator / denominator
    # Normalize: slope of 0.3 per turn is pretty steep
    return clamp(slope / 0.3, -1.0, 1.0)


def blend(current: float, trajectory: float, ratio: float = 0.4) -> float:
    """
    Blend current-message reading with trajectory.
    ratio = weight given to current message (0-1).
    
    Default 0.4 means: 40% what this message says, 60% where the
    conversation has been heading. This is the v0.2→v0.3 fix.
    
    The ratio should probably be dynamic eventually — a VERY decisive
    message should get more weight than a neutral one. But static 
    blending is the right v0.3 step.
    """
    return round(current * ratio + trajectory * (1 - ratio), 2)


# ── Signal Extraction (same as v0.2) ─────────────────────────────

def extract_signals(message: str) -> dict:
    lower = message.lower().strip()
    words = lower.split()
    word_count = len(words)

    if word_count == 0:
        return {"empty": True, "word_count": 0}

    sentences = re.split(r'[.!?]+', message)
    sentences = [s.strip() for s in sentences if s.strip()]

    return {
        "empty": False,
        "word_count": word_count,
        "char_count": len(message),
        "sentence_count": max(len(sentences), 1),
        "avg_sentence_length": word_count / max(len(sentences), 1),
        "question_marks": message.count("?"),
        "exclamation_marks": message.count("!"),
        "ellipsis_count": message.count("..."),
        "caps_ratio": sum(1 for c in message if c.isupper()) / max(len(message), 1),
        "hedge_count": count_matches(lower, HEDGE_WORDS),
        "certainty_count": count_matches(lower, CERTAINTY_WORDS),
        "expansion_count": count_matches(lower, EXPANSION_WORDS),
        "contraction_count": count_matches(lower, CONTRACTION_WORDS),
        "concrete_topic": count_matches(lower, CONCRETE_TOPIC_WORDS),
        "abstract_topic": count_matches(lower, ABSTRACT_TOPIC_WORDS),
        "concrete_mode": count_matches(lower, CONCRETE_MODE_WORDS),
        "abstract_mode": count_matches(lower, ABSTRACT_MODE_WORDS),
        "engagement_positive": count_matches(lower, ENGAGEMENT_POSITIVE),
        "engagement_negative": count_matches(lower, ENGAGEMENT_NEGATIVE),
        "has_sequential_markers": any(m in lower for m in ["first", "then", "next", "after that", "step"]),
        "has_reference_back": any(m in lower for m in ["we talked", "you said", "earlier", "before", "that thing", "the thing", "we discussed"]),
        "starts_with_connector": lower[:10].startswith(("so", "and", "but", "also", "plus", "right")),
    }


def classify_brevity(current: dict, previous: list[dict]) -> str:
    """Same as v0.2 — classify WHY a message is short."""
    if current.get("empty") or current["word_count"] > 8:
        return "not_brief"

    if current["certainty_count"] > 0 or current["engagement_positive"] > 0:
        return "decisive"

    if current["engagement_negative"] > 0 or current["hedge_count"] > 0:
        return "disengaged"

    if len(previous) >= 2:
        prev_lengths = [p.get("word_count", 0) for p in previous[-3:] if not p.get("empty")]
        if len(prev_lengths) >= 2 and all(prev_lengths[i] >= prev_lengths[i+1] for i in range(len(prev_lengths)-1)):
            return "disengaged"

    if len(previous) >= 1:
        prev_wcs = [p.get("word_count", 0) for p in previous[-3:] if not p.get("empty")]
        if prev_wcs:
            prev_avg = sum(prev_wcs) / len(prev_wcs)
            if prev_avg > 25 and current["word_count"] < 8:
                return "decisive"

    return "ambiguous"


# ── Per-Message Scorers ──────────────────────────────────────────
# These compute a raw score for ONE message. The trajectory engine
# then blends across the conversation.

def score_expansion_single(sig: dict, prev: list[dict]) -> float:
    """Raw expansion score for one message."""
    if sig.get("empty"):
        return -0.3

    score = 0.0

    exp = sig["expansion_count"]
    con = sig["contraction_count"]
    if exp + con > 0:
        score += (exp - con) / (exp + con) * 0.4

    if sig["question_marks"] > 0:
        score += min(sig["question_marks"] * 0.15, 0.3)

    brevity = classify_brevity(sig, prev)
    if brevity == "decisive":
        score -= 0.1
    elif brevity == "disengaged":
        score -= 0.35

    score -= sig.get("ellipsis_count", 0) * 0.05

    if sig.get("starts_with_connector"):
        score += 0.05
    if sig.get("has_reference_back"):
        score += 0.1

    return clamp(score)


def score_activation_single(sig: dict, prev: list[dict]) -> float:
    """Raw activation score for one message."""
    if sig.get("empty"):
        return -0.2

    score = 0.0

    score += min(sig["exclamation_marks"] * 0.15, 0.3)

    if sig["caps_ratio"] > 0.1:
        score += 0.2

    pos = sig["engagement_positive"]
    neg = sig["engagement_negative"]
    if pos + neg > 0:
        score += (pos - neg) / (pos + neg) * 0.25

    brevity = classify_brevity(sig, prev)
    if brevity == "decisive":
        score += 0.15
    elif brevity == "disengaged":
        score -= 0.2

    if brevity == "not_brief":
        if sig["word_count"] > 50:
            score += 0.15
        elif sig["word_count"] > 30:
            score += 0.05

    return clamp(score)


def score_certainty_single(sig: dict, prev: list[dict]) -> float:
    """Raw certainty score for one message."""
    if sig.get("empty"):
        return -0.2

    score = 0.0

    h = sig["hedge_count"]
    c = sig["certainty_count"]
    if h + c > 0:
        score += (c - h) / (h + c) * 0.5

    if sig["question_marks"] > 0:
        score -= min(sig["question_marks"] * 0.15, 0.4)

    brevity = classify_brevity(sig, prev)
    if brevity == "decisive":
        score += 0.4
    elif brevity == "disengaged":
        score -= 0.2
    elif brevity == "ambiguous" and sig["word_count"] <= 5:
        score -= 0.1
    elif sig["word_count"] <= 5 and sig["question_marks"] == 0:
        score += 0.2

    if sig.get("has_sequential_markers"):
        score += 0.1

    return clamp(score)


def score_openness_single(sig: dict, prev: list[dict]) -> float:
    """Raw openness score for one message (0-1 scale)."""
    if sig.get("empty"):
        return 0.5

    score = 0.5

    score += min(sig["question_marks"] * 0.1, 0.2)
    score += sig["expansion_count"] * 0.04

    brevity = classify_brevity(sig, prev)
    if brevity == "decisive":
        score -= 0.05
    elif brevity == "disengaged":
        score -= 0.15

    if sig["abstract_mode"] > 0:
        score += 0.1

    return clamp(score, 0.0, 1.0)


def score_materiality_single(sig: dict) -> float:
    """Raw materiality score for one message (0-1 scale)."""
    if sig.get("empty"):
        return 0.5

    mode_concrete = sig.get("concrete_mode", 0) * 2 + sig.get("concrete_topic", 0) * 0.5
    mode_abstract = sig.get("abstract_mode", 0) * 2 + sig.get("abstract_topic", 0) * 0.5

    if sig.get("has_sequential_markers"):
        mode_concrete += 1.5

    if sig.get("question_marks", 0) > 0 and sig.get("abstract_mode", 0) > 0:
        mode_abstract += 1

    total = mode_concrete + mode_abstract
    if total == 0:
        return 0.5

    raw = mode_concrete / total
    return clamp(0.2 + raw * 0.7, 0.0, 1.0)


# ── v0.3: Trajectory-Aware Complements ───────────────────────────

HALF_LIFE = 3.0        # Messages until weight halves
BLEND_CURRENT = 0.4    # Weight for current message vs trajectory
MOMENTUM_WINDOW = 4    # Messages for momentum calculation


def compute_trajectory_complement(
    signals_history: list[dict],
    scorer,
    value_range: tuple = (-1.0, 1.0),
    half_life: float = HALF_LIFE,
    blend_ratio: float = BLEND_CURRENT,
) -> tuple[float, float]:
    """
    Generic trajectory-aware complement computation.
    
    1. Score each message individually
    2. Apply exponential decay weights
    3. Compute weighted trajectory
    4. Compute momentum (rising/falling)
    5. Blend current message with trajectory
    
    Returns (blended_value, momentum)
    """
    if not signals_history:
        return (0.0, 0.0)

    # Score each message
    scores = []
    for i, sig in enumerate(signals_history):
        prev = signals_history[:i]
        if callable(scorer):
            # Check if scorer takes 1 or 2 args
            try:
                s = scorer(sig, prev)
            except TypeError:
                s = scorer(sig)
        scores.append(s)

    # Weighted trajectory
    weights = exponential_weights(len(scores), half_life)
    trajectory = weighted_average(scores, weights)

    # Current message score
    current = scores[-1]

    # Momentum
    momentum = compute_momentum(scores, MOMENTUM_WINDOW)

    # Blend
    blended = blend(current, trajectory, blend_ratio)

    lo, hi = value_range
    return (round(clamp(blended, lo, hi), 2), round(momentum, 2))


def detect_phase(signals_history: list[dict]) -> str:
    """v0.3: Phase detection now uses trajectory scores."""
    if len(signals_history) < 3:
        return "EXPLORING"

    recent = signals_history[-5:]

    total_questions = sum(s.get("question_marks", 0) for s in recent)
    total_concrete_mode = sum(s.get("concrete_mode", 0) for s in recent)
    total_expansion = sum(s.get("expansion_count", 0) for s in recent)
    total_certainty = sum(s.get("certainty_count", 0) for s in recent)

    recent_brevity = [classify_brevity(s, signals_history[:i])
                      for i, s in enumerate(recent)]
    decisive_count = recent_brevity.count("decisive")
    has_sequence = any(s.get("has_sequential_markers") for s in recent[-3:])

    if total_concrete_mode > 4 and has_sequence and total_questions <= 2:
        return "REFINING"
    if (total_certainty > 1 or decisive_count >= 2) and total_concrete_mode > 2:
        return "CONVERGING"
    if total_questions > 4 and total_expansion > 2 and total_concrete_mode < 2:
        return "EXPLORING"
    return "DIVERGING"


# ── Complement Instruction Generator ─────────────────────────────

def generate_affect_instructions(expansion: float, activation: float, certainty: float) -> str:
    """
    Generate the behavioral instruction blocks that match the computed values.
    These are the conditional instructions from the real Hearth OpSpec.
    """
    instructions = []

    # Expansion-based instructions
    if expansion < -0.3:
        instructions.append(
            "Open gently. Offer possibilities without demanding choice. "
            'Use "what if" and "I wonder" language. Don\'t push — create space '
            "they can step into. Longer sentences, softer framing. "
            "Avoid bullet points and lists — they compress."
        )
    elif expansion > 0.3:
        instructions.append(
            "They're open and flowing. Help them land somewhere. "
            'Reflect the core thread back. Ask "which of these feels most alive?" '
            "Don't add more — help crystallize what's already there."
        )

    # Activation-based instructions
    if activation < -0.3:
        instructions.append(
            "They're shut down or flat. Don't ask big questions. Don't analyze. "
            'Offer small, concrete, low-stakes starting points. "We could just..." '
            "Permission-giving language. Warmth without demand. "
            "Match their pace, then very gradually lift."
        )
    elif activation > 0.3:
        instructions.append(
            "They're overwhelmed or accelerating. Slow the pace. Short, grounded "
            "sentences. Name one thing at a time. Don't match their energy — be "
            'the steady surface. "Let\'s pause on just this one piece." '
            "Create breathing room between ideas."
        )

    # Certainty-based instructions
    if certainty < -0.3:
        instructions.append(
            "They're seeking and unsure. Don't pile on more options. Offer one "
            'concrete frame or anchor. "Here\'s one way to think about this." '
            "Name what seems true from what they've said. Give them something "
            "solid to push against."
        )
    elif certainty > 0.3:
        instructions.append(
            "They know what they want. Don't second-guess or over-explain. "
            "Execute. Match their directness. If they're wrong, say so directly "
            "— they can handle it. No hedging, no preamble."
        )

    return "\n\n".join(instructions) if instructions else (
        "Neutral state. Read the room and respond naturally."
    )


def generate_forge_instructions(openness: float, materiality: float, phase: str) -> str:
    """Generate forge complement behavioral instructions."""
    phase_instructions = {
        "EXPLORING": "Keep the space wide open. Ask questions. Don't solve yet.",
        "DIVERGING": "Keep the space open. Introduce unexpected material. Resist convergence.",
        "CONVERGING": "Help them narrow. Reflect back what's emerging. Start naming decisions.",
        "REFINING": "Be a mirror. Show them what they made, not what you'd make.",
    }
    return phase_instructions.get(phase, "Respond naturally to the creative moment.")


# ── Main Analyzer ─────────────────────────────────────────────────

@dataclass
class AffectComplement:
    expansion: float
    activation: float
    certainty: float
    # v0.3: momentum tracks direction of change
    expansion_momentum: float = 0.0
    activation_momentum: float = 0.0
    certainty_momentum: float = 0.0

@dataclass
class ForgeComplement:
    openness: float
    materiality: float
    phase: str
    openness_momentum: float = 0.0
    materiality_momentum: float = 0.0

@dataclass
class DebugInfo:
    brevity_classification: str
    recent_word_counts: list[int]
    per_message_scores: dict = field(default_factory=dict)
    decay_weights: list[float] = field(default_factory=list)

@dataclass
class HearthAnalysis:
    affect: AffectComplement
    forge: ForgeComplement
    debug: DebugInfo
    message_count: int

    def to_context_block(self) -> str:
        """Generate full context injection with values AND instructions."""
        affect_instructions = generate_affect_instructions(
            self.affect.expansion, self.affect.activation, self.affect.certainty
        )
        forge_instructions = generate_forge_instructions(
            self.forge.openness, self.forge.materiality, self.forge.phase
        )

        momentum_note = ""
        # Flag notable momentum shifts
        shifts = []
        if abs(self.affect.certainty_momentum) > 0.3:
            direction = "rising" if self.affect.certainty_momentum > 0 else "falling"
            shifts.append(f"certainty {direction}")
        if abs(self.affect.expansion_momentum) > 0.3:
            direction = "expanding" if self.affect.expansion_momentum > 0 else "contracting"
            shifts.append(f"{direction}")
        if shifts:
            momentum_note = f"\nMomentum: {', '.join(shifts)}"

        return f"""[AFFECT COMPLEMENT]
Shape: expansion={self.affect.expansion:.2f}, activation={self.affect.activation:.2f}, certainty={self.affect.certainty:.2f}{momentum_note}

{affect_instructions}
[END AFFECT COMPLEMENT]

[FORGE COMPLEMENT]
Shape: openness={self.forge.openness:.2f}, materiality={self.forge.materiality:.2f}
Phase: {self.forge.phase}

{forge_instructions}
[END FORGE COMPLEMENT]"""


def analyze_conversation(messages: list[dict]) -> HearthAnalysis:
    user_messages = [m for m in messages if m.get("role") == "user"]

    if not user_messages:
        return HearthAnalysis(
            affect=AffectComplement(0.0, 0.0, 0.0),
            forge=ForgeComplement(0.5, 0.5, "EXPLORING"),
            debug=DebugInfo("none", []),
            message_count=0,
        )

    signals_history = [extract_signals(m["content"]) for m in user_messages]
    current = signals_history[-1]
    brevity = classify_brevity(current, signals_history[:-1])

    # v0.3: Trajectory-aware computation
    exp_val, exp_mom = compute_trajectory_complement(
        signals_history, score_expansion_single)

    act_val, act_mom = compute_trajectory_complement(
        signals_history, score_activation_single)

    cert_val, cert_mom = compute_trajectory_complement(
        signals_history, score_certainty_single)

    open_val, open_mom = compute_trajectory_complement(
        signals_history, score_openness_single, value_range=(0.0, 1.0))

    mat_val, mat_mom = compute_trajectory_complement(
        signals_history, score_materiality_single, value_range=(0.0, 1.0))

    phase = detect_phase(signals_history)

    # Debug: show per-message expansion scores for validation
    exp_scores = []
    for i, sig in enumerate(signals_history):
        exp_scores.append(round(score_expansion_single(sig, signals_history[:i]), 2))

    weights = exponential_weights(len(signals_history), HALF_LIFE)

    affect = AffectComplement(
        expansion=exp_val, activation=act_val, certainty=cert_val,
        expansion_momentum=exp_mom, activation_momentum=act_mom,
        certainty_momentum=cert_mom,
    )

    forge = ForgeComplement(
        openness=open_val, materiality=mat_val, phase=phase,
        openness_momentum=open_mom, materiality_momentum=mat_mom,
    )

    debug = DebugInfo(
        brevity_classification=brevity,
        recent_word_counts=[s.get("word_count", 0) for s in signals_history[-5:]],
        per_message_scores={"expansion": exp_scores[-5:]},
        decay_weights=[round(w, 3) for w in weights[-5:]],
    )

    return HearthAnalysis(
        affect=affect, forge=forge, debug=debug,
        message_count=len(user_messages),
    )


# ── Tests ─────────────────────────────────────────────────────────

def run_test(name, emoji, expected, messages, actual_injected=None):
    result = analyze_conversation(messages)
    print(f"\n{emoji} Scenario: {name}")
    print(f"   Expected:  {expected}")
    print(f"   Affect:    exp={result.affect.expansion}, act={result.affect.activation}, cert={result.affect.certainty}")
    print(f"   Momentum:  exp_m={result.affect.expansion_momentum}, act_m={result.affect.activation_momentum}, cert_m={result.affect.certainty_momentum}")
    print(f"   Forge:     open={result.forge.openness}, mat={result.forge.materiality}, phase={result.forge.phase}")
    print(f"   Debug:     brevity={result.debug.brevity_classification}, wc={result.debug.recent_word_counts}")
    print(f"   Scores:    {result.debug.per_message_scores}")
    print(f"   Weights:   {result.debug.decay_weights}")
    if actual_injected:
        print(f"   Actual:    {actual_injected}")
    print(f"\n   ── Generated Context Block ──")
    print(f"{result.to_context_block()}")
    return result


def test_scenarios():
    print("=" * 64)
    print("  HEARTH ANALYZER v0.3 — TRAJECTORY-AWARE VALIDATION")
    print("=" * 64)

    # THE CRITICAL TEST: "Lets do it" after exploration
    # v0.2 gave certainty=0.9. Actual was 0.2. Target: ~0.2-0.4
    run_test("THIS CONVERSATION (the v0.2 bug)", "🪞",
        "certainty should be ~0.2-0.4, NOT 0.9 (the v0.2 over-index bug)",
        [
            {"role": "user", "content": "Context is Not Compute: Destroying the ICL 'World Model' Myth... let me know when your ready to begin"},
            {"role": "assistant", "content": "Ready. Let's get into it."},
            {"role": "user", "content": "So, hearth is great... its not asking the system to deal with novel anything... just steering along the well defined grooves of the system... but something that could actually reason from the POV of the OpSpec could be even more powerful. We talked about how chain of thought sort of helps the frontier models, but its not enough. What about that thing where the model creates a python script and pokes at the information"},
            {"role": "assistant", "content": "You're describing code-as-reasoning."},
            {"role": "user", "content": "Would the dynamic parts be better or worse for coding?"},
            {"role": "assistant", "content": "Counterintuitively — better."},
            {"role": "user", "content": "what advantage would this technique give us?"},
            {"role": "assistant", "content": "The model stops being asked to do two jobs at once."},
            {"role": "user", "content": "We have to try this... can we create a new branch of Hearth or something so we don't break anything?"},
            {"role": "assistant", "content": "You don't even need to branch."},
            {"role": "user", "content": "Lets do it"},
        ],
        actual_injected="expansion=-0.55, activation=0, certainty=0.2 | openness=0.55, materiality=0.67, phase=DIVERGING"
    )

    print(f"\n{'─' * 64}")
    print(f"  Brevity Disambiguation (should still work from v0.2)")
    print(f"{'─' * 64}")

    run_test("DECISIVE after exploration", "✅",
        "positive certainty, positive activation",
        [
            {"role": "user", "content": "I've been going back and forth on this for a while. The architecture could go either way. But I think the simpler approach is better because it lets us iterate faster."},
            {"role": "assistant", "content": "Makes sense. Want to start?"},
            {"role": "user", "content": "Yes. Let's go."},
        ])

    run_test("DISENGAGED declining", "🔻",
        "negative certainty, negative activation",
        [
            {"role": "user", "content": "This is getting complicated..."},
            {"role": "assistant", "content": "Let me simplify."},
            {"role": "user", "content": "sure"},
            {"role": "assistant", "content": "Here's a cleaner version."},
            {"role": "user", "content": "ok"},
        ])

    print(f"\n{'─' * 64}")
    print(f"  v0.3 Momentum Tests")
    print(f"{'─' * 64}")

    run_test("CERTAINTY RISING (hedge → decide)", "📈",
        "cert_momentum should be strongly positive",
        [
            {"role": "user", "content": "I'm not sure about this... maybe we should think more about it?"},
            {"role": "assistant", "content": "Take your time."},
            {"role": "user", "content": "I think there might be something here actually..."},
            {"role": "assistant", "content": "What are you seeing?"},
            {"role": "user", "content": "Yeah, I'm pretty sure this is the right approach."},
            {"role": "assistant", "content": "What convinced you?"},
            {"role": "user", "content": "Let's definitely go with this."},
        ])

    run_test("EXPANSION COLLAPSING (explore → shut down)", "📉",
        "exp_momentum should be negative",
        [
            {"role": "user", "content": "What if we tried a totally different approach? Like what about using WebSockets instead? Or maybe a different framework entirely?"},
            {"role": "assistant", "content": "Lots of options. What draws you to each?"},
            {"role": "user", "content": "Well I guess WebSockets could work but it's complicated..."},
            {"role": "assistant", "content": "What about the framework idea?"},
            {"role": "user", "content": "nvm, just stick with what we have"},
        ])

    print("\n" + "=" * 64)
    print("  v0.3 key checks:")
    print("  1. Certainty for meta-test should be ~0.2-0.4, not 0.9")
    print("  2. Momentum correctly tracks direction of change")
    print("  3. Brevity disambiguation still works")
    print("  4. Context block includes behavioral instructions")
    print("=" * 64)


if __name__ == "__main__":
    test_scenarios()
