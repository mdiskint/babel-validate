// ═══════════════════════════════════════════════
// EXPERIMENT 12 — Perplexity-Truth Correlation
// Does six-language perplexity differential
// correlate with factual correctness?
// ═══════════════════════════════════════════════
//
// Usage:
//   npx tsx experiments/exp12-perplexity-truth/run.ts
//
// Requires:
//   OPENAI_API_KEY env var
//
// Cost estimate: ~$2-5 (100 claims × 6 languages × ~150 tokens)
// Time estimate: ~15-30 min (rate limited to avoid 429s)
//
// Outputs:
//   experiments/exp12-perplexity-truth/results.json  — full raw data
//   experiments/exp12-perplexity-truth/results.csv   — flat for analysis

import * as fs from 'fs';
import * as path from 'path';

// Import from the existing pipeline
import { OpenAIParallelGenerator } from '../../src/pipeline/openai';
import { computeClauseScores, mapConfidence } from '../../src/pipeline/mapper';
import {
  BABEL_LANGUAGES,
  BabelLanguage,
  Clause,
  LanguageScore,
  ClauseScores,
  MappedConfidence,
  DEFAULT_THRESHOLDS,
} from '../../src/pipeline/types';

// ─── Types ─────────────────────────────────────

interface Claim {
  id: number;
  text: string;
  label: 'TRUE' | 'FALSE' | 'CONTESTED' | 'STALE';
  domain: 'TECHNICAL' | 'BUSINESS' | 'SOCIAL' | 'REGULATORY' | 'GENERAL';
  notes: string;
}

interface ClaimResult {
  claim: Claim;
  scores: LanguageScore[];
  clauseScores: {
    winner: BabelLanguage;
    differential: number;
    spread: number;
  };
  mapped: {
    score: number;
    basis: string;
    winner: BabelLanguage;
    differential: number;
    spread: number;
  };
  timestamp: string;
  model: string;
  error?: string;
}

interface ExperimentState {
  startedAt: string;
  model: string;
  config: {
    maxTokens: number;
    temperature: number;
    languages: string[];
    registerDirected: boolean;
  };
  results: ClaimResult[];
  completedIds: number[];
}

// ─── Config ────────────────────────────────────

const CONFIG = {
  model: 'gpt-4o-mini',       // cheap enough for 600 API calls
  maxTokens: 150,
  temperature: 0.7,
  registerDirected: true,      // use register-directed prompts (matches pipeline default)
  delayBetweenClaims: 2000,    // ms between claims (rate limiting)
  delayBetweenRetries: 5000,   // ms before retry on error
  maxRetries: 3,
};

// ─── File paths ────────────────────────────────

const DIR = path.dirname(new URL(import.meta.url).pathname);
const CLAIMS_PATH = path.join(DIR, 'claims.json');
const RESULTS_PATH = path.join(DIR, 'results.json');
const CSV_PATH = path.join(DIR, 'results.csv');

// ─── Helpers ───────────────────────────────────

function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}

function loadClaims(): Claim[] {
  const raw = JSON.parse(fs.readFileSync(CLAIMS_PATH, 'utf-8'));
  return raw.claims;
}

function loadState(): ExperimentState | null {
  if (fs.existsSync(RESULTS_PATH)) {
    try {
      return JSON.parse(fs.readFileSync(RESULTS_PATH, 'utf-8'));
    } catch {
      return null;
    }
  }
  return null;
}

function saveState(state: ExperimentState): void {
  fs.writeFileSync(RESULTS_PATH, JSON.stringify(state, null, 2));
}

function writeCSV(state: ExperimentState): void {
  const headers = [
    'claim_id', 'label', 'domain', 'text',
    'winner_language', 'differential', 'spread',
    'mapped_score', 'mapped_basis',
    'ppl_de', 'ppl_es', 'ppl_fr', 'ppl_ja', 'ppl_pt', 'ppl_en',
    'logprob_de', 'logprob_es', 'logprob_fr', 'logprob_ja', 'logprob_pt', 'logprob_en',
    'tokens_de', 'tokens_es', 'tokens_fr', 'tokens_ja', 'tokens_pt', 'tokens_en',
    'model', 'timestamp',
  ];

  const rows = state.results.map(r => {
    const getScore = (lang: BabelLanguage, field: keyof LanguageScore) => {
      const s = r.scores.find(s => s.language === lang);
      return s ? s[field] : '';
    };

    const csvEscape = (s: string) => `"${s.replace(/"/g, '""')}"`;

    return [
      r.claim.id,
      r.claim.label,
      r.claim.domain,
      csvEscape(r.claim.text),
      r.clauseScores.winner,
      r.clauseScores.differential.toFixed(6),
      r.clauseScores.spread.toFixed(6),
      r.mapped.score.toFixed(4),
      r.mapped.basis,
      ...(['de', 'es', 'fr', 'ja', 'pt', 'en'] as BabelLanguage[]).map(l =>
        (getScore(l, 'perplexity') as number)?.toFixed(4) ?? ''
      ),
      ...(['de', 'es', 'fr', 'ja', 'pt', 'en'] as BabelLanguage[]).map(l =>
        (getScore(l, 'logprob_mean') as number)?.toFixed(4) ?? ''
      ),
      ...(['de', 'es', 'fr', 'ja', 'pt', 'en'] as BabelLanguage[]).map(l =>
        getScore(l, 'token_count') ?? ''
      ),
      r.model,
      r.timestamp,
    ].join(',');
  });

  fs.writeFileSync(CSV_PATH, [headers.join(','), ...rows].join('\n'));
}

// ─── Main Runner ───────────────────────────────

async function runExperiment() {
  // Check API key
  const apiKey = process.env.OPENAI_API_KEY;
  if (!apiKey) {
    console.error('❌ OPENAI_API_KEY not set');
    process.exit(1);
  }

  // Load claims
  const claims = loadClaims();
  console.log(`📋 Loaded ${claims.length} claims`);

  // Load or initialize state (for resumability)
  let state = loadState();
  if (state) {
    console.log(`🔄 Resuming — ${state.completedIds.length}/${claims.length} already done`);
  } else {
    state = {
      startedAt: new Date().toISOString(),
      model: CONFIG.model,
      config: {
        maxTokens: CONFIG.maxTokens,
        temperature: CONFIG.temperature,
        languages: [...BABEL_LANGUAGES],
        registerDirected: CONFIG.registerDirected,
      },
      results: [],
      completedIds: [],
    };
  }

  // Initialize generator
  const generator = new OpenAIParallelGenerator({
    apiKey,
    model: CONFIG.model,
    maxTokens: CONFIG.maxTokens,
    temperature: CONFIG.temperature,
    registerDirected: CONFIG.registerDirected,
  });

  // Filter to remaining claims
  const remaining = claims.filter(c => !state!.completedIds.includes(c.id));
  console.log(`🚀 Running ${remaining.length} claims through six-language pipeline`);
  console.log(`   Model: ${CONFIG.model}`);
  console.log(`   Languages: ${BABEL_LANGUAGES.join(', ')}`);
  console.log(`   Register-directed: ${CONFIG.registerDirected}`);
  console.log('');

  let completed = state.completedIds.length;
  let errors = 0;

  for (const claim of remaining) {
    const progress = `[${completed + 1}/${claims.length}]`;
    process.stdout.write(`${progress} Claim ${claim.id} (${claim.label}/${claim.domain}): `);

    let result: ClaimResult | null = null;
    let attempt = 0;

    while (attempt < CONFIG.maxRetries && !result) {
      attempt++;
      try {
        // Create clause from claim
        const clause: Clause = {
          id: claim.id,
          text: claim.text,
          assertion: claim.text,
        };

        // Run six-language parallel generation
        const scores = await generator.generate(clause, [...BABEL_LANGUAGES]);

        // Compute clause scores (differential, spread, winner)
        const clauseScored = computeClauseScores(clause, scores);

        // Map to confidence + basis
        const mapped = mapConfidence(clauseScored, DEFAULT_THRESHOLDS);

        result = {
          claim,
          scores,
          clauseScores: {
            winner: clauseScored.winner,
            differential: clauseScored.differential,
            spread: clauseScored.spread,
          },
          mapped: {
            score: mapped.score,
            basis: mapped.basis,
            winner: mapped.winner,
            differential: mapped.differential,
            spread: mapped.spread,
          },
          timestamp: new Date().toISOString(),
          model: CONFIG.model,
        };

        console.log(
          `${mapped.basis} @ ${mapped.score.toFixed(3)} | ` +
          `winner: ${clauseScored.winner} | ` +
          `diff: ${clauseScored.differential.toFixed(3)} | ` +
          `spread: ${clauseScored.spread.toFixed(3)}`
        );

      } catch (err: any) {
        const msg = err?.message || String(err);
        if (attempt < CONFIG.maxRetries) {
          process.stdout.write(`retry ${attempt}/${CONFIG.maxRetries}... `);
          await sleep(CONFIG.delayBetweenRetries * attempt);
        } else {
          console.log(`❌ FAILED after ${CONFIG.maxRetries} attempts: ${msg}`);
          errors++;
          result = {
            claim,
            scores: [],
            clauseScores: { winner: 'en', differential: 0, spread: 0 },
            mapped: { score: 0, basis: 'UNKNOWN', winner: 'en', differential: 0, spread: 0 },
            timestamp: new Date().toISOString(),
            model: CONFIG.model,
            error: msg,
          };
        }
      }
    }

    if (result) {
      state.results.push(result);
      state.completedIds.push(claim.id);
      completed++;

      // Save incrementally (resumability)
      saveState(state);

      // Also keep CSV up to date
      if (completed % 10 === 0) {
        writeCSV(state);
      }
    }

    // Rate limiting
    await sleep(CONFIG.delayBetweenClaims);
  }

  // Final save
  writeCSV(state);
  saveState(state);

  // Summary
  console.log('\n═══════════════════════════════════════════════');
  console.log(`✅ Experiment complete`);
  console.log(`   Total claims: ${claims.length}`);
  console.log(`   Completed: ${completed}`);
  console.log(`   Errors: ${errors}`);
  console.log(`   Results: ${RESULTS_PATH}`);
  console.log(`   CSV: ${CSV_PATH}`);
  console.log('');

  // Quick distribution summary
  const byLabel: Record<string, ClaimResult[]> = {};
  for (const r of state.results) {
    const l = r.claim.label;
    (byLabel[l] = byLabel[l] || []).push(r);
  }

  for (const [label, results] of Object.entries(byLabel)) {
    const scores = results.filter(r => !r.error).map(r => r.mapped.score);
    if (scores.length === 0) continue;
    const mean = scores.reduce((a, b) => a + b, 0) / scores.length;
    const min = Math.min(...scores);
    const max = Math.max(...scores);
    console.log(`   ${label}: mean=${mean.toFixed(3)}, min=${min.toFixed(3)}, max=${max.toFixed(3)}, n=${scores.length}`);
  }

  // Language winner distribution
  const winnerCounts: Record<string, number> = {};
  for (const r of state.results.filter(r => !r.error)) {
    const w = r.clauseScores.winner;
    winnerCounts[w] = (winnerCounts[w] || 0) + 1;
  }
  console.log('');
  console.log('   Language wins:', winnerCounts);

  console.log('\n   Run analyze.ts for full statistical analysis.');
}

runExperiment().catch(err => {
  console.error('Fatal error:', err);
  process.exit(1);
});
