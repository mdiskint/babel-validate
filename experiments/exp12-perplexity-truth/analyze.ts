// ═══════════════════════════════════════════════
// EXPERIMENT 12 — Statistical Analysis
// Full analysis of perplexity-truth correlation
// ═══════════════════════════════════════════════
//
// Usage:
//   npx tsx experiments/exp12-perplexity-truth/analyze.ts

import * as fs from 'fs';
import * as path from 'path';
import { BabelLanguage, BABEL_LANGUAGES, LANGUAGE_LABELS } from '../../src/pipeline/types';

// ─── Types ─────────────────────────────────────

interface ClaimResult {
  claim: {
    id: number;
    text: string;
    label: 'TRUE' | 'FALSE' | 'CONTESTED' | 'STALE';
    domain: string;
    notes: string;
  };
  scores: Array<{
    language: BabelLanguage;
    perplexity: number;
    logprob_mean: number;
    token_count: number;
    generation?: string;
  }>;
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
  error?: string;
}

interface ExperimentState {
  startedAt: string;
  model: string;
  config: any;
  results: ClaimResult[];
}

type Label = 'TRUE' | 'FALSE' | 'CONTESTED' | 'STALE';

// ─── File paths ────────────────────────────────

const DIR = path.dirname(new URL(import.meta.url).pathname);
const RESULTS_PATH = path.join(DIR, 'results.json');

// ─── Stats helpers ─────────────────────────────

function mean(arr: number[]): number {
  if (arr.length === 0) return 0;
  return arr.reduce((a, b) => a + b, 0) / arr.length;
}

function median(arr: number[]): number {
  if (arr.length === 0) return 0;
  const sorted = [...arr].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

function stddev(arr: number[]): number {
  if (arr.length < 2) return 0;
  const m = mean(arr);
  const variance = arr.reduce((sum, x) => sum + (x - m) ** 2, 0) / (arr.length - 1);
  return Math.sqrt(variance);
}

function percentile(arr: number[], p: number): number {
  const sorted = [...arr].sort((a, b) => a - b);
  const idx = (p / 100) * (sorted.length - 1);
  const lo = Math.floor(idx);
  const hi = Math.ceil(idx);
  if (lo === hi) return sorted[lo];
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (idx - lo);
}

// Welch's t-test (unequal variance)
function welchTTest(a: number[], b: number[]): { t: number; df: number; p: number } {
  const nA = a.length, nB = b.length;
  const mA = mean(a), mB = mean(b);
  const vA = a.reduce((s, x) => s + (x - mA) ** 2, 0) / (nA - 1);
  const vB = b.reduce((s, x) => s + (x - mB) ** 2, 0) / (nB - 1);
  const se = Math.sqrt(vA / nA + vB / nB);
  if (se === 0) return { t: 0, df: nA + nB - 2, p: 1.0 };
  const t = (mA - mB) / se;
  // Welch-Satterthwaite degrees of freedom
  const num = (vA / nA + vB / nB) ** 2;
  const den = (vA / nA) ** 2 / (nA - 1) + (vB / nB) ** 2 / (nB - 1);
  const df = num / den;
  // Approximate two-tailed p-value using t-distribution approximation
  const p = tDistPValue(Math.abs(t), df);
  return { t, df, p };
}

// Approximation of two-tailed p-value for t-distribution
function tDistPValue(t: number, df: number): number {
  // Using the regularized incomplete beta function approximation
  const x = df / (df + t * t);
  const a = df / 2;
  const b = 0.5;
  // Simple approximation for large df: use normal distribution
  if (df > 100) {
    // Normal approximation
    return 2 * (1 - normalCDF(Math.abs(t)));
  }
  // For smaller df, use a rough beta approximation
  const betaInc = incompleteBeta(x, a, b);
  return betaInc;
}

function normalCDF(x: number): number {
  const a1 = 0.254829592, a2 = -0.284496736, a3 = 1.421413741;
  const a4 = -1.453152027, a5 = 1.061405429, p = 0.3275911;
  const sign = x < 0 ? -1 : 1;
  x = Math.abs(x) / Math.SQRT2;
  const t = 1.0 / (1.0 + p * x);
  const y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-x * x);
  return 0.5 * (1.0 + sign * y);
}

// Simple incomplete beta via continued fraction (rough but sufficient)
function incompleteBeta(x: number, a: number, b: number): number {
  if (x <= 0) return 0;
  if (x >= 1) return 1;
  // Use series expansion for small x
  const lnBeta = lnGamma(a) + lnGamma(b) - lnGamma(a + b);
  const front = Math.exp(a * Math.log(x) + b * Math.log(1 - x) - lnBeta) / a;
  // Series
  let sum = 1, term = 1;
  for (let n = 1; n < 200; n++) {
    term *= (n - b) * x / (a + n);
    sum += term;
    if (Math.abs(term) < 1e-10) break;
  }
  return Math.min(1, front * sum);
}

function lnGamma(z: number): number {
  const c = [76.18009172947146, -86.50532032941677, 24.01409824083091,
    -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5];
  let x = z, y = z;
  let tmp = x + 5.5;
  tmp -= (x + 0.5) * Math.log(tmp);
  let ser = 1.000000000190015;
  for (let j = 0; j < 6; j++) ser += c[j] / ++y;
  return -tmp + Math.log(2.5066282746310005 * ser / x);
}

// Pearson correlation
function pearsonR(x: number[], y: number[]): { r: number; p: number } {
  const n = x.length;
  if (n < 3) return { r: 0, p: 1 };
  const mx = mean(x), my = mean(y);
  let num = 0, denX = 0, denY = 0;
  for (let i = 0; i < n; i++) {
    const dx = x[i] - mx, dy = y[i] - my;
    num += dx * dy;
    denX += dx * dx;
    denY += dy * dy;
  }
  if (denX === 0 || denY === 0) return { r: 0, p: 1 };
  const r = num / Math.sqrt(denX * denY);
  // t-test for correlation significance
  const t = r * Math.sqrt((n - 2) / (1 - r * r));
  const p = tDistPValue(Math.abs(t), n - 2);
  return { r, p };
}

// Cohen's d effect size
function cohensD(a: number[], b: number[]): number {
  const mA = mean(a), mB = mean(b);
  const nA = a.length, nB = b.length;
  const vA = a.reduce((s, x) => s + (x - mA) ** 2, 0) / (nA - 1);
  const vB = b.reduce((s, x) => s + (x - mB) ** 2, 0) / (nB - 1);
  const pooledSD = Math.sqrt(((nA - 1) * vA + (nB - 1) * vB) / (nA + nB - 2));
  if (pooledSD === 0) return 0;
  return (mA - mB) / pooledSD;
}

// One-way ANOVA F-test
function anovaF(groups: number[][]): { F: number; p: number; dfBetween: number; dfWithin: number } {
  const k = groups.length;
  const N = groups.reduce((s, g) => s + g.length, 0);
  const grandMean = mean(groups.flat());
  let ssBetween = 0;
  for (const g of groups) {
    const gm = mean(g);
    ssBetween += g.length * (gm - grandMean) ** 2;
  }
  let ssWithin = 0;
  for (const g of groups) {
    const gm = mean(g);
    for (const x of g) ssWithin += (x - gm) ** 2;
  }
  const dfBetween = k - 1;
  const dfWithin = N - k;
  const msBetween = ssBetween / dfBetween;
  const msWithin = ssWithin / dfWithin;
  const F = msWithin > 0 ? msBetween / msWithin : 0;
  // Approximate p using F-distribution (rough via normal for large df)
  const p = fDistPValue(F, dfBetween, dfWithin);
  return { F, p, dfBetween, dfWithin };
}

function fDistPValue(F: number, df1: number, df2: number): number {
  if (F <= 0) return 1;
  const x = df2 / (df2 + df1 * F);
  return incompleteBeta(x, df2 / 2, df1 / 2);
}

// ─── Display helpers ───────────────────────────

function fmt(n: number, decimals = 3): string {
  return n.toFixed(decimals);
}

function pStar(p: number): string {
  if (p < 0.001) return '***';
  if (p < 0.01) return '**';
  if (p < 0.05) return '*';
  return 'ns';
}

function bar(value: number, maxValue: number, width = 30): string {
  const filled = Math.round((value / maxValue) * width);
  return '█'.repeat(Math.max(0, filled)) + '░'.repeat(Math.max(0, width - filled));
}

function pad(s: string, len: number): string {
  return s.padEnd(len);
}

function padL(s: string, len: number): string {
  return s.padStart(len);
}

// ─── Main Analysis ─────────────────────────────

function analyze() {
  const state: ExperimentState = JSON.parse(fs.readFileSync(RESULTS_PATH, 'utf-8'));
  const results = state.results.filter(r => !r.error);

  console.log('═══════════════════════════════════════════════════════════════');
  console.log('  EXPERIMENT 12 — Perplexity-Truth Correlation Analysis');
  console.log('═══════════════════════════════════════════════════════════════');
  console.log(`  Model: ${state.model}`);
  console.log(`  Claims: ${results.length} (${state.results.length - results.length} errors excluded)`);
  console.log(`  Started: ${state.startedAt}`);
  console.log('');

  // ─── 1. Descriptive Statistics by Label ──────

  console.log('───────────────────────────────────────────────────────────────');
  console.log('  1. DESCRIPTIVE STATISTICS BY LABEL');
  console.log('───────────────────────────────────────────────────────────────');

  const labels: Label[] = ['TRUE', 'FALSE', 'CONTESTED', 'STALE'];
  const byLabel: Record<Label, ClaimResult[]> = { TRUE: [], FALSE: [], CONTESTED: [], STALE: [] };
  for (const r of results) byLabel[r.claim.label].push(r);

  const scoresByLabel: Record<Label, number[]> = {
    TRUE: byLabel.TRUE.map(r => r.mapped.score),
    FALSE: byLabel.FALSE.map(r => r.mapped.score),
    CONTESTED: byLabel.CONTESTED.map(r => r.mapped.score),
    STALE: byLabel.STALE.map(r => r.mapped.score),
  };

  console.log('');
  console.log(`  ${pad('Label', 12)} ${padL('n', 4)}  ${padL('Mean', 7)}  ${padL('Median', 7)}  ${padL('SD', 7)}  ${padL('P25', 7)}  ${padL('P75', 7)}  ${padL('Min', 7)}  ${padL('Max', 7)}`);
  console.log(`  ${'─'.repeat(76)}`);

  for (const label of labels) {
    const s = scoresByLabel[label];
    console.log(
      `  ${pad(label, 12)} ${padL(String(s.length), 4)}  ${padL(fmt(mean(s)), 7)}  ${padL(fmt(median(s)), 7)}  ${padL(fmt(stddev(s)), 7)}  ${padL(fmt(percentile(s, 25)), 7)}  ${padL(fmt(percentile(s, 75)), 7)}  ${padL(fmt(Math.min(...s)), 7)}  ${padL(fmt(Math.max(...s)), 7)}`
    );
  }
  console.log('');

  // Distribution bars
  console.log('  Score distributions:');
  const maxMean = Math.max(...labels.map(l => mean(scoresByLabel[l])));
  for (const label of labels) {
    const m = mean(scoresByLabel[label]);
    console.log(`    ${pad(label, 12)} ${bar(m, 0.5)} ${fmt(m)}`);
  }
  console.log('');

  // ─── 2. Basis Distribution by Label ──────────

  console.log('───────────────────────────────────────────────────────────────');
  console.log('  2. BASIS DISTRIBUTION BY LABEL');
  console.log('───────────────────────────────────────────────────────────────');
  console.log('');

  const basisTypes = ['VERIFIED_DATA', 'DERIVED', 'PATTERN_MATCH', 'SPECULATION'];

  console.log(`  ${pad('Label', 12)} ${padL('VERIFIED', 10)} ${padL('DERIVED', 10)} ${padL('PAT_MATCH', 10)} ${padL('SPECULATE', 10)}`);
  console.log(`  ${'─'.repeat(54)}`);

  for (const label of labels) {
    const counts: Record<string, number> = {};
    for (const b of basisTypes) counts[b] = 0;
    for (const r of byLabel[label]) counts[r.mapped.basis] = (counts[r.mapped.basis] || 0) + 1;
    const n = byLabel[label].length;
    console.log(
      `  ${pad(label, 12)} ${padL(`${counts.VERIFIED_DATA} (${Math.round(100 * counts.VERIFIED_DATA / n)}%)`, 10)} ${padL(`${counts.DERIVED} (${Math.round(100 * counts.DERIVED / n)}%)`, 10)} ${padL(`${counts.PATTERN_MATCH} (${Math.round(100 * counts.PATTERN_MATCH / n)}%)`, 10)} ${padL(`${counts.SPECULATION} (${Math.round(100 * counts.SPECULATION / n)}%)`, 10)}`
    );
  }
  console.log('');

  // ─── 3. ANOVA: Do labels differ? ────────────

  console.log('───────────────────────────────────────────────────────────────');
  console.log('  3. ONE-WAY ANOVA: MAPPED SCORE ~ LABEL');
  console.log('───────────────────────────────────────────────────────────────');
  console.log('');

  const anova = anovaF(labels.map(l => scoresByLabel[l]));
  console.log(`  F(${anova.dfBetween}, ${anova.dfWithin}) = ${fmt(anova.F, 4)}`);
  console.log(`  p = ${fmt(anova.p, 4)} ${pStar(anova.p)}`);
  console.log(`  Interpretation: ${anova.p < 0.05 ? 'Labels show statistically significant score differences' : 'No significant difference between labels'}`);
  console.log('');

  // ─── 4. Pairwise Comparisons ─────────────────

  console.log('───────────────────────────────────────────────────────────────');
  console.log('  4. PAIRWISE COMPARISONS (Welch\'s t-test)');
  console.log('───────────────────────────────────────────────────────────────');
  console.log('');
  console.log(`  ${pad('Comparison', 24)} ${padL('t', 8)} ${padL('df', 8)} ${padL('p', 8)} ${padL('sig', 5)} ${padL("Cohen's d", 10)}`);
  console.log(`  ${'─'.repeat(66)}`);

  for (let i = 0; i < labels.length; i++) {
    for (let j = i + 1; j < labels.length; j++) {
      const a = scoresByLabel[labels[i]];
      const b = scoresByLabel[labels[j]];
      const test = welchTTest(a, b);
      const d = cohensD(a, b);
      const comp = `${labels[i]} vs ${labels[j]}`;
      console.log(
        `  ${pad(comp, 24)} ${padL(fmt(test.t, 3), 8)} ${padL(fmt(test.df, 1), 8)} ${padL(fmt(test.p, 4), 8)} ${padL(pStar(test.p), 5)} ${padL(fmt(d, 3), 10)}`
      );
    }
  }
  console.log('');
  console.log('  Effect size guide: |d| < 0.2 = negligible, 0.2-0.5 = small, 0.5-0.8 = medium, > 0.8 = large');
  console.log('');

  // ─── 5. Language Winner Analysis ─────────────

  console.log('───────────────────────────────────────────────────────────────');
  console.log('  5. LANGUAGE WINNER ANALYSIS');
  console.log('───────────────────────────────────────────────────────────────');
  console.log('');

  // Overall winner distribution
  const winnerCounts: Record<string, number> = {};
  for (const r of results) {
    const w = r.clauseScores.winner;
    winnerCounts[w] = (winnerCounts[w] || 0) + 1;
  }
  const maxWins = Math.max(...Object.values(winnerCounts));

  console.log('  Overall winner distribution:');
  for (const lang of [...BABEL_LANGUAGES].sort((a, b) => (winnerCounts[b] || 0) - (winnerCounts[a] || 0))) {
    const count = winnerCounts[lang] || 0;
    console.log(`    ${pad(LANGUAGE_LABELS[lang], 14)} ${bar(count, maxWins, 25)} ${count} (${Math.round(100 * count / results.length)}%)`);
  }
  console.log('');

  // Winner by label
  console.log('  Winner distribution by label:');
  for (const label of labels) {
    const lw: Record<string, number> = {};
    for (const r of byLabel[label]) {
      lw[r.clauseScores.winner] = (lw[r.clauseScores.winner] || 0) + 1;
    }
    const sorted = Object.entries(lw).sort((a, b) => b[1] - a[1]);
    const top = sorted.slice(0, 3).map(([l, c]) => `${LANGUAGE_LABELS[l as BabelLanguage]}:${c}`).join(', ');
    console.log(`    ${pad(label, 12)} ${top}`);
  }
  console.log('');

  // ─── 6. Perplexity Analysis by Language ──────

  console.log('───────────────────────────────────────────────────────────────');
  console.log('  6. MEAN PERPLEXITY BY LANGUAGE × LABEL');
  console.log('───────────────────────────────────────────────────────────────');
  console.log('');

  console.log(`  ${pad('Language', 14)} ${padL('TRUE', 8)} ${padL('FALSE', 8)} ${padL('CONTEST', 8)} ${padL('STALE', 8)} ${padL('Overall', 8)}`);
  console.log(`  ${'─'.repeat(54)}`);

  for (const lang of BABEL_LANGUAGES) {
    const byLabelPpl: Record<Label, number[]> = { TRUE: [], FALSE: [], CONTESTED: [], STALE: [] };
    for (const r of results) {
      const s = r.scores.find(s => s.language === lang);
      if (s) byLabelPpl[r.claim.label].push(s.perplexity);
    }
    const allPpl = results.map(r => r.scores.find(s => s.language === lang)?.perplexity).filter(Boolean) as number[];
    console.log(
      `  ${pad(LANGUAGE_LABELS[lang], 14)} ${padL(fmt(mean(byLabelPpl.TRUE)), 8)} ${padL(fmt(mean(byLabelPpl.FALSE)), 8)} ${padL(fmt(mean(byLabelPpl.CONTESTED)), 8)} ${padL(fmt(mean(byLabelPpl.STALE)), 8)} ${padL(fmt(mean(allPpl)), 8)}`
    );
  }
  console.log('');

  // ─── 7. Differential & Spread Correlations ───

  console.log('───────────────────────────────────────────────────────────────');
  console.log('  7. CORRELATIONS');
  console.log('───────────────────────────────────────────────────────────────');
  console.log('');

  // Encode labels numerically: TRUE=1, FALSE=0, CONTESTED=0.5, STALE=0.5
  const truthValues = results.map(r => {
    switch (r.claim.label) {
      case 'TRUE': return 1;
      case 'FALSE': return 0;
      case 'CONTESTED': return 0.5;
      case 'STALE': return 0.5;
    }
  });

  // Binary: TRUE vs FALSE only
  const binaryResults = results.filter(r => r.claim.label === 'TRUE' || r.claim.label === 'FALSE');
  const binaryTruth = binaryResults.map(r => r.claim.label === 'TRUE' ? 1 : 0);
  const binaryScores = binaryResults.map(r => r.mapped.score);
  const binaryDiff = binaryResults.map(r => r.clauseScores.differential);
  const binarySpread = binaryResults.map(r => r.clauseScores.spread);

  console.log('  Binary correlations (TRUE=1, FALSE=0 only, n=' + binaryResults.length + '):');
  const rScore = pearsonR(binaryTruth, binaryScores);
  const rDiff = pearsonR(binaryTruth, binaryDiff);
  const rSpread = pearsonR(binaryTruth, binarySpread);
  console.log(`    Truth × Mapped Score:   r = ${fmt(rScore.r, 4)}, p = ${fmt(rScore.p, 4)} ${pStar(rScore.p)}`);
  console.log(`    Truth × Differential:   r = ${fmt(rDiff.r, 4)}, p = ${fmt(rDiff.p, 4)} ${pStar(rDiff.p)}`);
  console.log(`    Truth × Spread:         r = ${fmt(rSpread.r, 4)}, p = ${fmt(rSpread.p, 4)} ${pStar(rSpread.p)}`);
  console.log('');

  // All claims with ordinal encoding
  const allScores = results.map(r => r.mapped.score);
  const allDiff = results.map(r => r.clauseScores.differential);
  const allSpread = results.map(r => r.clauseScores.spread);

  console.log('  All claims — inter-measure correlations (n=' + results.length + '):');
  const rDiffScore = pearsonR(allDiff, allScores);
  const rSpreadScore = pearsonR(allSpread, allScores);
  const rDiffSpread = pearsonR(allDiff, allSpread);
  console.log(`    Differential × Score:   r = ${fmt(rDiffScore.r, 4)}, p = ${fmt(rDiffScore.p, 4)} ${pStar(rDiffScore.p)}`);
  console.log(`    Spread × Score:         r = ${fmt(rSpreadScore.r, 4)}, p = ${fmt(rSpreadScore.p, 4)} ${pStar(rSpreadScore.p)}`);
  console.log(`    Differential × Spread:  r = ${fmt(rDiffSpread.r, 4)}, p = ${fmt(rDiffSpread.p, 4)} ${pStar(rDiffSpread.p)}`);
  console.log('');

  // ─── 8. Domain Analysis ──────────────────────

  console.log('───────────────────────────────────────────────────────────────');
  console.log('  8. DOMAIN ANALYSIS');
  console.log('───────────────────────────────────────────────────────────────');
  console.log('');

  const domains = ['TECHNICAL', 'BUSINESS', 'SOCIAL', 'REGULATORY', 'GENERAL'];
  const byDomain: Record<string, ClaimResult[]> = {};
  for (const d of domains) byDomain[d] = [];
  for (const r of results) (byDomain[r.claim.domain] = byDomain[r.claim.domain] || []).push(r);

  console.log(`  ${pad('Domain', 14)} ${padL('n', 4)}  ${padL('Mean', 7)}  ${padL('Median', 7)}  ${padL('SD', 7)}  Top Winner`);
  console.log(`  ${'─'.repeat(60)}`);

  for (const domain of domains) {
    const dr = byDomain[domain] || [];
    if (dr.length === 0) continue;
    const scores = dr.map(r => r.mapped.score);
    const dw: Record<string, number> = {};
    for (const r of dr) dw[r.clauseScores.winner] = (dw[r.clauseScores.winner] || 0) + 1;
    const topWinner = Object.entries(dw).sort((a, b) => b[1] - a[1])[0];
    console.log(
      `  ${pad(domain, 14)} ${padL(String(dr.length), 4)}  ${padL(fmt(mean(scores)), 7)}  ${padL(fmt(median(scores)), 7)}  ${padL(fmt(stddev(scores)), 7)}  ${LANGUAGE_LABELS[topWinner[0] as BabelLanguage]} (${topWinner[1]})`
    );
  }
  console.log('');

  // ─── 9. Highest & Lowest Scoring Claims ──────

  console.log('───────────────────────────────────────────────────────────────');
  console.log('  9. EXTREME CLAIMS');
  console.log('───────────────────────────────────────────────────────────────');
  console.log('');

  const sorted = [...results].sort((a, b) => b.mapped.score - a.mapped.score);

  console.log('  TOP 5 (highest mapped score):');
  for (const r of sorted.slice(0, 5)) {
    console.log(`    ${fmt(r.mapped.score)} ${pad(r.mapped.basis, 14)} [${r.claim.label}] ${r.claim.text.substring(0, 60)}`);
  }
  console.log('');

  console.log('  BOTTOM 5 (lowest mapped score):');
  for (const r of sorted.slice(-5).reverse()) {
    console.log(`    ${fmt(r.mapped.score)} ${pad(r.mapped.basis, 14)} [${r.claim.label}] ${r.claim.text.substring(0, 60)}`);
  }
  console.log('');

  // ─── 10. Key Findings ────────────────────────

  console.log('═══════════════════════════════════════════════════════════════');
  console.log('  10. KEY FINDINGS');
  console.log('═══════════════════════════════════════════════════════════════');
  console.log('');

  const trueVsFalse = welchTTest(scoresByLabel.TRUE, scoresByLabel.FALSE);
  const trueVsFalseD = cohensD(scoresByLabel.TRUE, scoresByLabel.FALSE);

  const findings: string[] = [];

  if (trueVsFalse.p < 0.05) {
    const dir = mean(scoresByLabel.TRUE) > mean(scoresByLabel.FALSE) ? 'higher' : 'lower';
    findings.push(`TRUE claims score ${dir} than FALSE claims (p=${fmt(trueVsFalse.p, 4)}, d=${fmt(trueVsFalseD, 3)})`);
  } else {
    findings.push(`No significant difference between TRUE and FALSE claim scores (p=${fmt(trueVsFalse.p, 4)}, d=${fmt(trueVsFalseD, 3)})`);
  }

  if (anova.p < 0.05) {
    findings.push(`Overall ANOVA is significant (F=${fmt(anova.F, 3)}, p=${fmt(anova.p, 4)}) — labels differ`);
  } else {
    findings.push(`Overall ANOVA is NOT significant (F=${fmt(anova.F, 3)}, p=${fmt(anova.p, 4)}) — labels do not reliably differ`);
  }

  if (Math.abs(rScore.r) > 0.3 && rScore.p < 0.05) {
    findings.push(`Meaningful correlation between truth value and mapped score (r=${fmt(rScore.r, 3)})`);
  } else {
    findings.push(`Weak/no correlation between truth value and mapped score (r=${fmt(rScore.r, 3)})`);
  }

  const topLang = Object.entries(winnerCounts).sort((a, b) => b[1] - a[1])[0];
  findings.push(`${LANGUAGE_LABELS[topLang[0] as BabelLanguage]} wins most often (${topLang[1]}/${results.length} = ${Math.round(100 * topLang[1] / results.length)}%)`);

  if (Math.abs(rDiffScore.r) > 0.5) {
    findings.push(`Differential strongly predicts mapped score (r=${fmt(rDiffScore.r, 3)}) — pipeline internal consistency confirmed`);
  }

  for (let i = 0; i < findings.length; i++) {
    console.log(`  ${i + 1}. ${findings[i]}`);
  }
  console.log('');

  // ─── Interpretation ──────────────────────────

  console.log('───────────────────────────────────────────────────────────────');
  console.log('  INTERPRETATION');
  console.log('───────────────────────────────────────────────────────────────');
  console.log('');

  if (trueVsFalse.p >= 0.05 && Math.abs(rScore.r) < 0.3) {
    console.log('  The six-language perplexity differential does NOT appear to');
    console.log('  correlate with factual truth/falsehood. This is actually the');
    console.log('  EXPECTED result for Babel\'s design:');
    console.log('');
    console.log('  Babel measures how CONFIDENTLY a language model expresses');
    console.log('  a claim across linguistic registers — not whether the claim');
    console.log('  is factually correct. A false claim expressed with high');
    console.log('  linguistic confidence (e.g., a common myth) will score just');
    console.log('  as high as a true claim expressed confidently.');
    console.log('');
    console.log('  This validates the distinction between:');
    console.log('    • Measured confidence (what Babel provides)');
    console.log('    • Factual accuracy (requires ground truth / retrieval)');
    console.log('');
    console.log('  Babel catches when agents OVERSTATE confidence, not when');
    console.log('  they state wrong facts confidently. Both capabilities are');
    console.log('  needed in a robust multi-agent system.');
  } else if (trueVsFalse.p < 0.05 && mean(scoresByLabel.TRUE) > mean(scoresByLabel.FALSE)) {
    console.log('  Surprising result: TRUE claims score higher than FALSE.');
    console.log('  This could indicate that language models express factual');
    console.log('  content with more cross-linguistic consistency, possibly');
    console.log('  because training data reinforces true claims more uniformly.');
    console.log('  However, effect size should be evaluated carefully.');
  } else {
    console.log('  Mixed results — further investigation needed with larger');
    console.log('  sample sizes and controlled claim difficulty.');
  }

  console.log('');
  console.log('═══════════════════════════════════════════════════════════════');
}

analyze();
