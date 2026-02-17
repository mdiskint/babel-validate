// ═══════════════════════════════════════════════
// BABEL — Perplexity-to-Confidence Mapper
// ═══════════════════════════════════════════════

import { Basis } from '../types';
import {
  ClauseScores,
  MappedConfidence,
  ConfidenceThresholds,
  DEFAULT_THRESHOLDS,
  LanguageScore,
} from './types';

export function mapConfidence(
  clauseScores: ClauseScores,
  thresholds: ConfidenceThresholds = DEFAULT_THRESHOLDS
): MappedConfidence {
  const { clause, scores, winner, differential, spread } = clauseScores;
  const basis = determineBasis(differential, spread, thresholds);
  const score = determineScore(differential, spread, basis, thresholds);

  return {
    clause,
    score: clamp(score, 0, 1),
    basis,
    winner,
    differential,
    spread,
    generation: scores.find(s => s.language === winner)?.generation,
  };
}

export function mapConfidenceBatch(
  allScores: ClauseScores[],
  thresholds: ConfidenceThresholds = DEFAULT_THRESHOLDS
): MappedConfidence[] {
  return allScores.map(cs => mapConfidence(cs, thresholds));
}

function determineBasis(
  differential: number,
  spread: number,
  t: ConfidenceThresholds
): Basis {
  const strongDiff = differential >= t.strongDifferential;
  const moderateDiff = differential >= t.moderateDifferential;
  const highSpread = spread >= t.highSpread;
  const moderateSpread = spread >= t.moderateSpread;

  if (strongDiff && highSpread) return 'VERIFIED_DATA';
  if (strongDiff && moderateSpread) return 'DERIVED';
  if (moderateDiff && moderateSpread) return 'DERIVED';
  if (moderateDiff || moderateSpread) return 'PATTERN_MATCH';
  return 'SPECULATION';
}

function determineScore(
  differential: number,
  spread: number,
  basis: Basis,
  t: ConfidenceThresholds
): number {
  const bands: Record<string, { floor: number; ceiling: number }> = {
    VERIFIED_DATA: { floor: t.derivedCeiling, ceiling: t.verifiedCeiling },
    DERIVED: { floor: t.patternMatchCeiling, ceiling: t.derivedCeiling },
    PATTERN_MATCH: { floor: t.speculationFloor + 0.10, ceiling: t.patternMatchCeiling },
    SPECULATION: { floor: t.speculationFloor, ceiling: t.speculationFloor + 0.15 },
  };

  const band = bands[basis] || bands.SPECULATION;

  const diffNorm = basis === 'VERIFIED_DATA' || basis === 'DERIVED'
    ? normalize(differential, t.moderateDifferential, t.strongDifferential * 1.5)
    : normalize(differential, 0, t.moderateDifferential);

  const spreadNorm = basis === 'VERIFIED_DATA' || basis === 'DERIVED'
    ? normalize(spread, t.moderateSpread, t.highSpread * 1.5)
    : normalize(spread, 0, t.moderateSpread);

  const factor = clamp(diffNorm * 0.6 + spreadNorm * 0.4, 0, 1);

  return band.floor + factor * (band.ceiling - band.floor);
}

export function computeClauseScores(
  clause: { id: number; text: string; assertion?: string },
  scores: LanguageScore[]
): ClauseScores {
  if (scores.length === 0) {
    throw new Error(`No language scores for clause ${clause.id}`);
  }

  const sorted = [...scores].sort((a, b) => a.perplexity - b.perplexity);
  const winner = sorted[0];
  const runnerUp = sorted[1] || winner;

  const range = sorted[sorted.length - 1].perplexity - sorted[0].perplexity;
  const rawGap = runnerUp.perplexity - winner.perplexity;
  
  const MIN_MEANINGFUL_RANGE = 1.0;
  const rangeSignificance = Math.min(range / MIN_MEANINGFUL_RANGE, 1.0);
  const differential = range > 0 
    ? (rawGap / range) * rangeSignificance 
    : 0;

  const perplexities = scores.map(s => s.perplexity);
  const mean = perplexities.reduce((a, b) => a + b, 0) / perplexities.length;
  const variance = perplexities.reduce((sum, p) => sum + (p - mean) ** 2, 0) / perplexities.length;
  const stddev = Math.sqrt(variance);
  const spread = mean > 0 ? stddev / mean : 0;

  return {
    clause,
    scores,
    winner: winner.language,
    differential,
    spread,
  };
}

export function perplexityFromLogprobs(logprobs: number[]): number {
  if (logprobs.length === 0) return Infinity;
  const meanLogprob = logprobs.reduce((a, b) => a + b, 0) / logprobs.length;
  return Math.exp(-meanLogprob);
}

function normalize(value: number, min: number, max: number): number {
  if (max <= min) return 0;
  return clamp((value - min) / (max - min), 0, 1);
}

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}
