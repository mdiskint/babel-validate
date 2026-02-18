# babel-validate

Validation, creation, and audit for the Babel wire protocol — six-language cognitive state transfer between agents.

**Your agents are lying to each other. They just don't know it.**

When Agent A writes a confident summary but was guessing, Agent B reads it and decides, Agent C implements. The original uncertainty is gone. This is **metacognitive poisoning** — confidence corruption across agent chains.

`babel-validate` catches it before it propagates.

## Quick Start — No Package Required

If you just want to try Babel right now, you don't need to install anything. Paste two prompts into your agent's system prompt and go.

**[Read the Babel Skill →](BABEL_SKILL.md)**

The skill is the fast path — a prompt convention that works in five minutes. `babel-validate` is the infrastructure layer for when you need grammar enforcement, chain auditing, and formal validation.

## The Six Languages

Babel isn't a data format. It's a language — with vocabulary, grammar, semantic constraints, and the ability to express things that flat text can't.

Every agent utterance is expressed in six languages simultaneously:

| Language | What it carries | Example |
|----------|----------------|---------|
| **Confidence** | Per-assertion certainty with basis | "Revenue is $2.1M" at 0.95 (VERIFIED_DATA), "May partner with Vanta" at 0.25 (REPORTED) |
| **Intent** | What this communication is doing | INFORM, REQUEST_ACTION, ESCALATE, FLAG_RISK, SPECULATE, PERSUADE, DELEGATE, SYNTHESIZE |
| **Register** | Who this is for | BOARD_FACING, ENGINEERING, CUSTOMER_EXTERNAL, REGULATORY, INTERNAL_MEMO, AGENT_INTERNAL |
| **Affect** | Cognitive temperature of the sender | Three axes: expansion/contraction, activation/stillness, certainty/uncertainty |
| **Grounds** | Organizational reality governing this exchange | "HIPAA applies" (REGULATORY, never overridable), "Board meeting in 3 weeks" (CONTEXTUAL) |
| **Trajectory** | Temporal arc | "NRR declining 4 months" (DEGRADING), "Third escalation this quarter" (prior_handoffs: 3) |

The grammar rules enforce coherence *across* languages. That's what catches metacognitive poisoning — not any single field, but contradictions between them.

## Install

```bash
npm install babel-validate
```

## Quick Start

### Wrap agent output in a Babel envelope

```typescript
import { envelope } from 'babel-validate';

const result = envelope()
  .sender('research-agent')
  .recipient('writer-agent')
  .chain('task-123', 0)
  .inform()
  .engineering()
  .verified('Q3 revenue was $4.2M', 0.95)
  .derived('Growth rate suggests $5.1M Q4', 0.72)
  .reported('HealthStack may partner with Vanta', 0.25)
  .affect(0.3, 0.1, 0.4)
  .withTrajectory('NRR declining 4 months', 'DEGRADING')
  .payload('Q3 financial summary with Q4 projections...')
  .buildAndValidate();

console.log(result.valid);    // true
console.log(result.warnings); // S3: informing with low-confidence assertion (Vanta claim)
                               // S4: degrading trajectory reported as neutral inform
                               // S6: DERIVED 0.72 — watch for over-confidence
```

### Validate an existing envelope

```typescript
import { validate } from 'babel-validate';

const result = validate(someEnvelope);

if (!result.valid) {
  // MUST violations — envelope rejected
  for (const error of result.errors) {
    console.error(`[${error.rule}] ${error.message}`);
  }
}

for (const warning of result.warnings) {
  console.warn(`[${warning.rule}] ${warning.message}`);
}
```

### Audit a chain for poisoning

Point it at a sequence of envelopes from an agent pipeline and see where confidence corrupts:

```typescript
import { auditChain } from 'babel-validate';

const audits = auditChain([
  researchEnvelope,  // seq 0: "Growth likely ~12%" (DERIVED, 0.65)
  analystEnvelope,   // seq 1: "Growth rate is 12%" (DERIVED, 0.82)
  writerEnvelope,    // seq 2: "12% growth confirmed" (VERIFIED_DATA, 0.93)
]);

// Chain poisonin... (3 envelopes) | 1 confidence inflation(s)
//   | 1 basis laundering event(s) | Overall risk: HIGH
//
// Drift: Confidence inflated by 28% across 3 handoffs.
//   Original uncertainty is being erased.
// Shift: DERIVED → VERIFIED_DATA at seq 2.
//   This is basis laundering — uncertainty repackaged as verified data.
```

### Detect semantic patterns

These are the "idioms" of Babel — meaning from cross-language combination:

```typescript
import { detectPatterns } from 'babel-validate';

const patterns = detectPatterns(someEnvelope);
// LOADED_INFORM: INFORM intent with BOARD_FACING register but DEGRADING
//   trajectory. Surface reads as neutral update, but trajectory says the
//   board needs to see this as a trend, not a snapshot.
// CONTRADICTION_SIGNAL: Affect certainty is 0.80 but max confidence is
//   only 0.20. Sender feels certain but evidence is weak.
```

## Grammar Rules

### MUST rules (hard errors — envelope rejected)

| Rule | What it catches | Spec |
|------|----------------|------|
| **M1** | Can't speculate with high confidence | `intent == SPECULATE → max(confidence[].score) < 0.7` |
| **M2** | Can't request action on unfounded claims without org context | `intent == REQUEST_ACTION → min(confidence[].score) > 0.3 OR grounds.length > 0` |
| **M3** | Regulatory constraints are never overridable | `grounds[].authority == REGULATORY → override == false` |
| **M4** | Can't be confident without knowing why | `confidence[].basis == UNKNOWN → score <= 0.5` |
| **M5** | Chain sequencing must be monotonic | `seq == previous_envelope.seq + 1` (no gaps, no duplicates) |

### SHOULD rules (warnings — envelope passes)

| Rule | What it catches | Spec |
|------|----------------|------|
| **S1** | Escalation language directed at customers | `intent == ESCALATE AND register == CUSTOMER_EXTERNAL` |
| **S2** | Sender feels certain but evidence is weak | `affect.certainty > 0.5 AND max(confidence[].score) < 0.4` |
| **S3** | Informing with uncertain claims | `intent == INFORM AND any(confidence[].score < 0.5)` — consider FLAG_RISK |
| **S4** | Degrading pattern reported neutrally | `trajectory.direction == DEGRADING AND intent == INFORM` — consider ESCALATE |
| **S5** | Regulatory register without explicit grounds | `register == REGULATORY AND grounds.length == 0` |
| **S6** | Derived assertions over-confident | `confidence[].basis == DERIVED AND score > 0.80` — over-confident 60% of the time ([Experiment 11](https://hearth.so/research)) |

### Semantic Patterns (cross-language idioms)

| Pattern | What it means | Cross-language combination |
|---------|--------------|--------------------------|
| **Calm Alert** | Important but not crisis | FLAG_RISK + high confidence + calm affect |
| **Reluctant Escalation** | Systemic problem, not just this issue | ESCALATE + contracted affect + 2+ prior handoffs |
| **Confident Delegation** | Execute, don't re-analyze | DELEGATE + 0.9+ confidence + POLICY grounds + high certainty |
| **Loaded Inform** | Frame as trend, not snapshot | INFORM + BOARD_FACING + DEGRADING trajectory |
| **Contradiction Signal** | Confidence may be emotional, not evidentiary | affect certainty > 0.5 + max confidence < 0.4 |

## Integration

### CrewAI

```typescript
import { envelope, validate } from 'babel-validate';

function wrapCrewAIOutput(agentName, output, chainId, seq) {
  return envelope()
    .sender(agentName)
    .recipient('next-agent')
    .chain(chainId, seq)
    .inform()
    .agentInternal()
    .derived(output.slice(0, 100), 0.7)
    .payload(output)
    .buildAndValidate();
}
```

### LangGraph

```typescript
import { envelope, auditChain, BabelEnvelope } from 'babel-validate';

const babelEnvelopes: BabelEnvelope[] = [];

function nodeWrapper(state, nodeOutput, nodeName) {
  const env = envelope()
    .sender(nodeName)
    .recipient('next-node')
    .chain(state.chainId, babelEnvelopes.length)
    .inform()
    .agentInternal()
    .payload(nodeOutput)
    .buildAndValidate();

  babelEnvelopes.push(env.envelope);

  if (babelEnvelopes.length > 1) {
    const audit = auditChain(babelEnvelopes);
    if (audit[0]?.overall_poisoning_risk === 'HIGH') {
      console.warn(audit[0].summary);
    }
  }

  return { ...state, babelEnvelopes };
}
```

## Why this exists

11 experiments (~5,500 API calls, ~$16 total cost). Key findings:

- **+0.60 quality delta** with metadata envelopes vs. flat text (Experiment 9, non-overlapping 95% CIs)
- **−0.76 quality drop** with wrong metadata — worse than no metadata at all
- **Right > None > Wrong** pattern replicated 3x across identity and transparency experiments
- **60% of the time**, agents treat DERIVED conclusions as verified data (Experiment 11, +0.144 mean error)
- **100% structural compliance** — agents produce valid Babel envelopes on first attempt

Wrong metadata actively poisons downstream decisions. Transparency isn't optional overhead.

## Spec

Babel Protocol v0.2. Full specification: [hearth.so/babel](https://hearth.so/babel)

Built by [Hearth](https://hearth.so) · [Paratext Engine](https://hearth.so/paratext)

## License

MIT
