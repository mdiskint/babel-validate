# babel-validate

Wire protocol + measurement engine for multi-agent metacognitive integrity.

**Your agents are lying to each other. They just don't know it.**

When Agent A writes a confident summary but was guessing, Agent B reads it and decides, Agent C implements. The original uncertainty is gone. This is **metacognitive poisoning** — confidence corruption across agent chains.

`babel-validate` catches it before it propagates.

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
  .speculation('Market conditions favor expansion', 0.35)
  .affect(0.3, 0.1, 0.4)
  .payload('Q3 financial summary with Q4 projections...')
  .buildAndValidate();

console.log(result.valid);    // true
console.log(result.warnings); // S6: DERIVED basis with score 0.72 — watch for over-confidence
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

// SHOULD violations — envelope passes with warnings
for (const warning of result.warnings) {
  console.warn(`[${warning.rule}] ${warning.message}`);
}
```

### Audit a chain for poisoning

This is the core capability. Point it at a sequence of envelopes from an agent pipeline and see where confidence corrupts:

```typescript
import { auditChain } from 'babel-validate';

const audits = auditChain([
  researchEnvelope,  // seq 0: "Growth likely ~12%" (DERIVED, 0.65)
  analystEnvelope,   // seq 1: "Growth rate is 12%" (DERIVED, 0.82)
  writerEnvelope,    // seq 2: "12% growth confirmed" (VERIFIED_DATA, 0.93)
]);

// Output:
// Chain task-123: 3 envelopes
// 1 confidence inflation(s) | 1 basis laundering event(s) | Overall risk: HIGH
//
// Drift: "Growth likely ~12%" inflated 28% across 3 handoffs.
// Basis shift: DERIVED → VERIFIED_DATA at seq 2.
//   This is basis laundering — uncertainty is being repackaged as verified data.
```

### Detect semantic contradictions

```typescript
import { detectPatterns } from 'babel-validate';

const patterns = detectPatterns(someEnvelope);
// LOADED_INFORM: Agent presents as neutral informer but envelope
//   carries strong emotional/certainty signal. May be covert persuasion.
// CONTRADICTION_SIGNAL: Wide confidence spread (0.25–0.92) within
//   single envelope. Process assertions independently.
```

## Grammar Rules

### MUST rules (hard errors — envelope rejected)

| Rule | What it catches |
|------|----------------|
| **M1** | Can't speculate with high confidence (SPECULATION/UNKNOWN basis + score > 0.5) |
| **M2** | Can't request action on unfounded claims without organizational grounds |
| **M3** | Regulatory constraints are never overridable |
| **M4** | Can't be confident without basis (score > 0.7 + no basis) |
| **M5** | Chain sequencing must be monotonic |

### SHOULD rules (warnings — envelope passes)

| Rule | What it catches |
|------|----------------|
| **S1** | ESCALATE without urgency in affect |
| **S2** | SPECULATE with high certainty in affect |
| **S3** | BOARD_FACING register with engineering jargon |
| **S4** | FLAG_RISK without any low-confidence assertions |
| **S5** | DELEGATE without trajectory handoff count |
| **S6** | DERIVED basis with score > 0.80 (over-confident 60% of the time — [Experiment 11](https://hearth.so/research)) |

## The Envelope

Every agent handoff gets wrapped in a Babel envelope:

```typescript
{
  meta: {
    version: "babel/0.2",
    timestamp: "2026-02-16T...",
    sender: "research-agent",
    recipient: "writer-agent",
    chain_id: "uuid",
    seq: 0
  },
  intent: "INFORM",              // What the agent is trying to do
  confidence: [{                  // Per-assertion confidence with basis
    assertion: "Q3 revenue was $4.2M",
    score: 0.95,
    basis: "VERIFIED_DATA"
  }],
  register: "ENGINEERING",        // Audience-appropriate framing
  affect: {                       // Cognitive/emotional state
    expansion: 0.3,
    activation: 0.1,
    certainty: 0.4
  },
  grounds: [{                     // Organizational constraints
    constraint: "SOC2 compliance required",
    authority: "REGULATORY",
    override: false
  }],
  trajectory: {                   // Temporal arc
    pattern: "quarterly-review",
    direction: "STABLE",
    prior_handoffs: 2
  },
  payload: "..."                  // The actual content
}
```

## Integration

### CrewAI

```typescript
import { envelope, validate } from 'babel-validate';

// Wrap CrewAI agent output before passing to next agent
function wrapCrewAIOutput(agentName: string, output: string, chainId: string, seq: number) {
  return envelope()
    .sender(agentName)
    .recipient('next-agent')
    .chain(chainId, seq)
    .inform()
    .agentInternal()
    // Parse confidence from agent output or use defaults
    .derived(output.slice(0, 100), 0.7)
    .payload(output)
    .buildAndValidate();
}
```

### LangGraph

```typescript
import { envelope, auditChain } from 'babel-validate';

// Add to LangGraph state
const babelEnvelopes: BabelEnvelope[] = [];

// In each node, wrap output
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

  // Check for poisoning after each step
  if (babelEnvelopes.length > 1) {
    const audit = auditChain(babelEnvelopes);
    if (audit[0]?.overall_poisoning_risk === 'HIGH') {
      // Surface to human, log, or halt pipeline
      console.warn(audit[0].summary);
    }
  }

  return { ...state, babelEnvelopes };
}
```

## Why this exists

We ran 11 experiments (~5,500 API calls, ~$16 total cost). Key findings:

- **+0.60 quality delta** when agent handoffs carry metadata envelopes vs. flat text ([Experiment 9](https://hearth.so/research))
- **−0.76 quality drop** when envelopes carry wrong metadata — worse than no metadata at all
- **60% of the time**, agents treat DERIVED conclusions as verified data ([Experiment 11](https://hearth.so/research))
- **100% structural compliance** — agents produce valid Babel envelopes on first attempt

The pattern `Right > None > Wrong` means transparency isn't optional overhead. Wrong metadata actively poisons downstream decisions.

## Spec

Babel Protocol v0.2. Full specification: [hearth.so/babel](https://hearth.so/babel)

Built by [Hearth](https://hearth.so) · [Paratext Engine](https://hearth.so/paratext)

## License

MIT
