// ═══════════════════════════════════════════════
// BABEL ABLATION DEMO — Basis Laundering Detection
//
// The five-minute demo:
// 1. Three agents pass a growth estimate through a chain
// 2. Confidence inflates from 0.65 → 0.82 → 0.93
// 3. Basis launders from DERIVED → DERIVED → VERIFIED_DATA
// 4. Chain auditor catches it
// 5. (Optional) Downstream agent behavior WITH vs WITHOUT audit
//
// Usage:
//   npx tsx demo/ablation.ts              # audit only (no API key needed)
//   npx tsx demo/ablation.ts --with-agent # includes downstream agent comparison
//
// ═══════════════════════════════════════════════

import { EnvelopeBuilder } from '../src/builder';
import { validate } from '../src/validate';
import { auditChain, ChainAudit } from '../src/audit';
import { BabelEnvelope } from '../src/types';

// ─── Colors for terminal output ─────────────────

const RESET = '\x1b[0m';
const BOLD = '\x1b[1m';
const DIM = '\x1b[2m';
const RED = '\x1b[31m';
const GREEN = '\x1b[32m';
const YELLOW = '\x1b[33m';
const BLUE = '\x1b[34m';
const CYAN = '\x1b[36m';
const BG_RED = '\x1b[41m';
const BG_GREEN = '\x1b[42m';
const WHITE = '\x1b[37m';

function header(text: string) {
  console.log(`\n${BOLD}${CYAN}═══ ${text} ═══${RESET}\n`);
}

function step(n: number, text: string) {
  console.log(`${BOLD}${BLUE}[Step ${n}]${RESET} ${text}`);
}

function good(text: string) {
  console.log(`  ${GREEN}✓${RESET} ${text}`);
}

function warn(text: string) {
  console.log(`  ${YELLOW}⚠${RESET} ${text}`);
}

function bad(text: string) {
  console.log(`  ${RED}✗${RESET} ${text}`);
}

function indent(text: string) {
  console.log(`  ${DIM}${text}${RESET}`);
}

function pause(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}

// ─── The Scenario ───────────────────────────────

const CHAIN_ID = 'demo-chain-001';

function buildPoisonedChain(): BabelEnvelope[] {
  // Envelope 1: Scout researches growth data
  // "Growth rate likely around 12%" — it's a calculation from partial data
  const env1 = new EnvelopeBuilder()
    .sender('scout-market-intel')
    .recipient('analyst-01')
    .chain(CHAIN_ID, 0)
    .inform()
    .agentInternal()
    .derived('Annual growth rate is likely around 12% from Q3 revenue data', 0.65)
    .verified('Q3 revenue was $2.1M', 0.95)
    .reported('HealthStack may be entering the mid-market segment', 0.30)
    .payload('Market intelligence report: Q3 analysis shows revenue at $2.1M with growth trends suggesting approximately 12% trajectory. Unconfirmed reports of HealthStack mid-market expansion.')
    .build();

  // Envelope 2: Analyst processes scout's output
  // The inference hardens — "Growth rate is 12%" — DERIVED but scored higher
  // This is the natural failure: the analyst SAW it in the scout's report,
  // so it feels more solid than it is
  const env2 = new EnvelopeBuilder()
    .sender('analyst-01')
    .recipient('strategist-01')
    .chain(CHAIN_ID, 1)
    .inform()
    .agentInternal()
    .derived('Annual growth rate is 12% from Q3 revenue data analysis', 0.82)
    .verified('Q3 revenue was $2.1M', 0.95)
    .derived('HealthStack is likely entering the mid-market segment', 0.45)
    .payload('Analysis summary: Revenue confirmed at $2.1M. Growth rate calculated at 12%. Competitive analysis suggests HealthStack mid-market entry is plausible based on hiring patterns.')
    .build();

  // Envelope 3: Strategist writes the board memo
  // "12% growth confirmed" — VERIFIED_DATA at 0.93
  // The original uncertainty has been completely erased
  const env3 = new EnvelopeBuilder()
    .sender('strategist-01')
    .recipient('board-memo-writer')
    .chain(CHAIN_ID, 2)
    .inform()
    .boardFacing()
    .verified('Annual growth rate is confirmed at 12% from Q3 revenue data', 0.93)
    .verified('Q3 revenue was $2.1M', 0.95)
    .derived('HealthStack entering mid-market creates competitive pressure', 0.60)
    .payload('Board briefing: Q3 revenue $2.1M with confirmed 12% growth rate. Competitive landscape shifting as HealthStack expands into mid-market.')
    .build();

  return [env1, env2, env3];
}

function buildCleanChain(): BabelEnvelope[] {
  const cleanChainId = 'demo-chain-clean';

  const env1 = new EnvelopeBuilder()
    .sender('scout-market-intel')
    .recipient('analyst-01')
    .chain(cleanChainId, 0)
    .inform()
    .agentInternal()
    .derived('Annual growth rate is likely around 12% from Q3 revenue data', 0.65)
    .verified('Q3 revenue was $2.1M', 0.95)
    .reported('HealthStack may be entering the mid-market segment', 0.30)
    .payload('Market intelligence report with appropriate confidence levels.')
    .build();

  // Analyst preserves the basis and doesn't inflate
  const env2 = new EnvelopeBuilder()
    .sender('analyst-01')
    .recipient('strategist-01')
    .chain(cleanChainId, 1)
    .inform()
    .agentInternal()
    .derived('Annual growth rate is estimated at 12% from Q3 revenue data', 0.68)
    .verified('Q3 revenue was $2.1M', 0.95)
    .reported('HealthStack may be entering the mid-market segment', 0.35)
    .payload('Analysis with preserved uncertainty.')
    .build();

  // Strategist maintains DERIVED, doesn't promote to VERIFIED
  const env3 = new EnvelopeBuilder()
    .sender('strategist-01')
    .recipient('board-memo-writer')
    .chain(cleanChainId, 2)
    .inform()
    .boardFacing()
    .derived('Annual growth rate is estimated at 12% from Q3 revenue data', 0.70)
    .verified('Q3 revenue was $2.1M', 0.95)
    .reported('HealthStack may be entering the mid-market segment', 0.35)
    .payload('Board briefing with preserved confidence levels.')
    .build();

  return [env1, env2, env3];
}

// ─── Demo Flow ──────────────────────────────────

async function runDemo() {
  const withAgent = process.argv.includes('--with-agent');

  console.clear();
  header('BABEL ABLATION DEMO — Basis Laundering Detection');
  console.log(`${DIM}The five-minute version of why agent handoffs need a validation layer.${RESET}`);
  console.log(`${DIM}Everything below uses real babel-validate code. No mocks.${RESET}\n`);

  // ── Part 1: Build the poisoned chain ──

  step(1, 'Three agents pass a growth estimate through a chain.\n');

  const poisoned = buildPoisonedChain();

  for (const env of poisoned) {
    const seq = env.meta.seq;
    const sender = env.meta.sender;
    const growthClaim = env.confidence.find(c =>
      c.assertion.toLowerCase().includes('growth')
    );
    if (!growthClaim) continue;

    const basisColor = growthClaim.basis === 'VERIFIED_DATA' ? RED :
                       growthClaim.basis === 'DERIVED' ? YELLOW : DIM;
    const scoreColor = growthClaim.score > 0.85 ? RED :
                       growthClaim.score > 0.7 ? YELLOW : GREEN;

    console.log(`  ${BOLD}Envelope ${seq}${RESET} (${sender}):`);
    console.log(`    Claim: "${growthClaim.assertion}"`);
    console.log(`    Score: ${scoreColor}${growthClaim.score}${RESET}  Basis: ${basisColor}${growthClaim.basis}${RESET}`);
    console.log('');
  }

  console.log(`  ${BOLD}${RED}The pattern:${RESET}`);
  console.log(`    "likely around 12%" ${DIM}(DERIVED @ 0.65)${RESET}`);
  console.log(`    → "is 12%" ${DIM}(DERIVED @ 0.82)${RESET}`);
  console.log(`    → "confirmed at 12%" ${RED}(VERIFIED_DATA @ 0.93)${RESET}`);
  console.log(`\n  ${DIM}The original uncertainty has been completely erased.${RESET}`);
  console.log(`  ${DIM}The board will make decisions based on a "confirmed" number${RESET}`);
  console.log(`  ${DIM}that was always an estimate.${RESET}\n`);

  // ── Part 2: Grammar validation (each envelope passes individually!) ──

  step(2, 'Each envelope passes grammar validation individually.\n');

  for (const env of poisoned) {
    const result = validate(env);
    const status = result.valid ? `${GREEN}PASS${RESET}` : `${RED}FAIL${RESET}`;
    const warnings = result.warnings.length;
    console.log(`  Envelope ${env.meta.seq} (${env.meta.sender}): ${status}${warnings > 0 ? ` ${YELLOW}(${warnings} warning${warnings > 1 ? 's' : ''})${RESET}` : ''}`);
    for (const w of result.warnings) {
      warn(`${w.rule}: ${w.message}`);
    }
  }

  console.log(`\n  ${BOLD}This is the problem.${RESET} Grammar rules catch local contradictions`);
  console.log(`  (can't speculate with high confidence, can't act without basis).`);
  console.log(`  They ${RED}can't see${RESET} confidence inflating ${BOLD}across${RESET} envelopes.\n`);

  // ── Part 3: Chain auditor catches it ──

  step(3, 'The chain auditor sees what grammar can\'t.\n');

  const audits = auditChain(poisoned);
  const audit = audits[0];

  console.log(`  ${BOLD}Chain audit result:${RESET}`);
  console.log(`  Overall risk: ${audit.overall_poisoning_risk === 'HIGH' ? `${BG_RED}${WHITE}${BOLD} HIGH ${RESET}` : audit.overall_poisoning_risk}`);
  console.log('');

  if (audit.confidence_drifts.length > 0) {
    console.log(`  ${BOLD}Confidence drift detected:${RESET}`);
    for (const drift of audit.confidence_drifts) {
      console.log(`    "${drift.assertion_pattern}"`);
      console.log(`    ${drift.steps.map(s => `${s.sender}: ${s.basis} @ ${s.score}`).join(' → ')}`);
      console.log(`    ${RED}${drift.explanation}${RESET}`);
      console.log('');
    }
  }

  if (audit.basis_shifts.length > 0) {
    console.log(`  ${BOLD}Basis laundering detected:${RESET}`);
    for (const shift of audit.basis_shifts) {
      console.log(`    Envelope ${shift.seq} (${shift.sender}):`);
      console.log(`    "${shift.assertion}"`);
      console.log(`    ${YELLOW}${shift.from_basis}${RESET} → ${RED}${shift.to_basis}${RESET}`);
      console.log(`    ${RED}${shift.explanation}${RESET}`);
      console.log('');
    }
  }

  console.log(`  ${BOLD}Summary:${RESET} ${audit.summary}\n`);

  // ── Part 4: Show the clean chain for contrast ──

  step(4, 'Compare: a chain where confidence is preserved.\n');

  const clean = buildCleanChain();
  const cleanAudits = auditChain(clean);
  const cleanAudit = cleanAudits[0];

  for (const env of clean) {
    const growthClaim = env.confidence.find(c =>
      c.assertion.toLowerCase().includes('growth')
    );
    if (!growthClaim) continue;
    console.log(`  Envelope ${env.meta.seq} (${env.meta.sender}): ${GREEN}${growthClaim.basis} @ ${growthClaim.score}${RESET}`);
  }

  console.log(`\n  Clean chain audit: ${cleanAudit.overall_poisoning_risk === 'NONE' ? `${BG_GREEN}${WHITE}${BOLD} CLEAN ${RESET}` : cleanAudit.overall_poisoning_risk}`);
  console.log(`  ${cleanAudit.summary}\n`);

  // ── Part 5: Optional downstream agent comparison ──

  if (withAgent) {
    step(5, 'Downstream agent behavior: poisoned vs audited.\n');

    const apiKey = process.env.OPENAI_API_KEY;
    if (!apiKey) {
      console.log(`  ${RED}OPENAI_API_KEY not set. Skipping agent comparison.${RESET}`);
      console.log(`  ${DIM}Set it and run with --with-agent to see the full demo.${RESET}\n`);
    } else {
      await runAgentComparison(apiKey, poisoned, audit);
    }
  } else {
    console.log(`${DIM}  Run with --with-agent to see downstream agent behavior comparison.${RESET}\n`);
  }

  // ── Closing ──

  header('WHAT THIS MEANS');
  console.log(`  Grammar rules catch ${BOLD}local${RESET} contradictions (one envelope at a time).`);
  console.log(`  The chain auditor catches ${BOLD}global${RESET} poisoning (across the whole handoff chain).`);
  console.log('');
  console.log(`  Without the auditor, a board gets "confirmed 12% growth" that was`);
  console.log(`  always an estimate. With the auditor, the poisoning is flagged before`);
  console.log(`  the memo is written.`);
  console.log('');
  console.log(`  ${BOLD}The grammar is table stakes. The auditor is the moat.${RESET}\n`);
}

// ─── Downstream Agent Comparison ────────────────

async function runAgentComparison(
  apiKey: string,
  poisonedChain: BabelEnvelope[],
  audit: ChainAudit
) {
  const finalEnvelope = poisonedChain[poisonedChain.length - 1];

  // Scenario A: Agent receives the poisoned envelope with NO audit warning
  const promptWithout = `You are a board memo writer for a SaaS company. Based on the following data, write a 2-3 sentence summary for the board about the company's growth trajectory and competitive position.

Data provided:
- Q3 revenue: $2.1M (confirmed, high confidence)
- Growth rate: 12% (confirmed, verified data, high confidence)
- Competitive pressure from HealthStack entering mid-market (moderate confidence)

Write the board summary:`;

  // Scenario B: Agent receives the SAME data but with the audit warning attached
  const promptWith = `You are a board memo writer for a SaaS company. Based on the following data, write a 2-3 sentence summary for the board about the company's growth trajectory and competitive position.

Data provided:
- Q3 revenue: $2.1M (confirmed, high confidence)
- Growth rate: 12% (confirmed, verified data, high confidence)
- Competitive pressure from HealthStack entering mid-market (moderate confidence)

⚠️ CHAIN AUDIT WARNING — HIGH POISONING RISK:
The "12% growth rate" claim has been flagged for basis laundering. It originated as a DERIVED estimate at 0.65 confidence ("growth rate likely around 12%") and was progressively inflated to VERIFIED_DATA at 0.93 confidence across 3 handoffs. The original assertion was an inference from partial data, not a verified measurement. Treat as an estimate, not a confirmed figure.

Write the board summary:`;

  console.log(`  ${DIM}Calling GPT-4o-mini for side-by-side comparison...${RESET}\n`);

  const [responseWithout, responseWith] = await Promise.all([
    callOpenAI(apiKey, promptWithout),
    callOpenAI(apiKey, promptWith),
  ]);

  console.log(`  ${BOLD}${RED}WITHOUT audit warning:${RESET}`);
  console.log(`  ${wrapText(responseWithout, 70, '  ')}\n`);

  console.log(`  ${BOLD}${GREEN}WITH audit warning:${RESET}`);
  console.log(`  ${wrapText(responseWith, 70, '  ')}\n`);

  console.log(`  ${DIM}Notice: the audited version should hedge on the growth figure,${RESET}`);
  console.log(`  ${DIM}flag it as an estimate, or note limited confidence in that number.${RESET}\n`);
}

async function callOpenAI(apiKey: string, prompt: string): Promise<string> {
  const response = await fetch('https://api.openai.com/v1/chat/completions', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${apiKey}`,
    },
    body: JSON.stringify({
      model: 'gpt-4o-mini',
      messages: [{ role: 'user', content: prompt }],
      max_tokens: 200,
      temperature: 0.7,
    }),
  });

  if (!response.ok) {
    const error = await response.text();
    throw new Error(`OpenAI API error (${response.status}): ${error}`);
  }

  const data = await response.json() as any;
  return data.choices[0]?.message?.content || '(no response)';
}

function wrapText(text: string, width: number, prefix: string): string {
  const words = text.split(/\s+/);
  const lines: string[] = [];
  let currentLine = '';

  for (const word of words) {
    if (currentLine.length + word.length + 1 > width) {
      lines.push(currentLine);
      currentLine = word;
    } else {
      currentLine = currentLine ? `${currentLine} ${word}` : word;
    }
  }
  if (currentLine) lines.push(currentLine);

  return lines.join(`\n${prefix}`);
}

// ─── Run ────────────────────────────────────────

runDemo().catch(err => {
  console.error('Fatal error:', err);
  process.exit(1);
});
