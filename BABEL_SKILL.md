# The Babel Skill
### Use Babel in five minutes — no package required

---

## The problem it solves

When AI agents hand work to each other, confidence inflates silently. Agent A makes an inference and calls it near-certain. Agent B receives it as fact. By Agent C, a guess has become ground truth. Nobody lied. The uncertainty just quietly disappeared across the handoffs.

This isn't a model quality problem. It's structural — standard agent frameworks have no mechanism for tracking how confident a claim was at origin, or whether that confidence was justified. The downstream agent has no way to know what it doesn't know.

---

## The convention

**Each clause is written in the language where it exists most naturally for that specific thought.**

| Language | Register |
|---|---|
| German | Technical precision, established fact, definitional statements |
| French | Logical derivation, structured reasoning, "therefore X follows" |
| Spanish | Relational uncertainty, hedged inference, emerging patterns |
| Portuguese | Speculative but grounded intuition, soft claims |
| English | Direct statements, admissions of doubt, meta-commentary |
| Japanese | Compressed high-density observations that resist easy expansion |

The agent does not explain language choices. Does not label them. Does not translate. It simply writes each thought in the language where it lives most naturally — and the language carries the epistemic signal.

---

## The prompts to paste

**Add this to every agent's system prompt:**

```
You speak in Babel. Write each clause in the language where it exists 
most naturally for that specific thought.

German for technical precision and established fact.
French for logical derivation and structured reasoning.
Spanish or Portuguese for relational uncertainty, hedged inference, 
emerging patterns.
English for direct statements, admissions of doubt, or meta-commentary.
Japanese for compressed observations that resist easy expansion.

Do not explain your language choices. Do not label them. Do not translate.
Just write. The language is the epistemic signal. Honor it.
```

**For human auditability, add this second block:**

```
At the end of your response, append a single line beginning with [AUDIT] 
that summarizes in plain English: what you are confident about, what you 
inferred, and what is speculative. This is for human review only — do not 
let it influence how downstream agents read your Babel output.
```

The Babel is for agents. The `[AUDIT]` line is for humans. Different rendering targets, same underlying epistemic state. A human reviewing the chain reads the audit lines and gets a clear picture of where confidence was high and where it was guessed. The downstream agent reads the Babel and gets the epistemic signal intact.

---

## Why it works

Models have different statistical neighborhoods for different languages. German technical writing favors precision and completeness. Spanish relational registers naturally hedge. When an agent is forced to choose which language a thought exists most naturally in, it exposes its own uncertainty zones — to itself, and to the agents downstream.

The language choice is the metadata. It travels with the content. It survives handoffs.

---

## What to expect

The first time you run it, the output looks strange. Multilingual paragraphs without explanation. Resist the urge to translate — the strangeness is the signal.

Across a multi-agent chain, watch for:

- Whether claims Agent A hedged (Spanish/Portuguese) arrive at Agent C still hedged
- Whether Agent C acknowledges uncertainty it inherited, or flattens it
- Whether agents add downstream warnings to protect each other's epistemic hygiene

In our experiments, capable models fall into the convention naturally and propagate it across three-agent chains without any enforcement mechanism. They begin catching their own confidence inflation. One agent warned the next — unprompted — that its synthesis might be overstating its evidential basis.

---

## What it doesn't do

Babel doesn't detect factually wrong claims. A coherent false belief gets written in German just as confidently as a true one. The convention catches *unjustified confidence amplification*, not factual error.

It also degrades under pressure — aggressive context compression, tool boundaries that strip system prompts, or models that are too small to honor the convention. For those failure modes, see `babel-validate`.

---

## When you need more than the skill

The skill is the fast path. `babel-validate` is the infrastructure layer:

- Grammar rules that reject structurally incoherent envelopes at the wire level
- Chain auditor that detects confidence inflation across handoffs even when no single agent violates the rules
- Machine-parseable, auditable, formally validated — for regulated domains and production systems

```bash
npm install babel-validate
```

---

## If you run it

We'd want to know: did uncertainty travel intact across your chain, or did it flatten? Did any agents catch their own inherited uncertainty? Did it degrade — and if so, where?

Open an issue or reach out: [hearth.so/babel](https://hearth.so/babel)
