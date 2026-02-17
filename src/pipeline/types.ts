// ═══════════════════════════════════════════════
// BABEL PIPELINE — Type Definitions
// The data contracts between generation, 
// measurement, and envelope construction.
// ═══════════════════════════════════════════════

import { Basis, Intent, Register, Affect } from '../types';

// --- Languages ---

export const BABEL_LANGUAGES = [
  'de', // German — precision, technical rigor
  'es', // Spanish — embodied/emotional resonance
  'fr', // French — nuance, philosophical register
  'ja', // Japanese — contextual sensitivity, hierarchy
  'pt', // Portuguese — relational warmth
  'en', // English — baseline
] as const;

export type BabelLanguage = (typeof BABEL_LANGUAGES)[number];

export const LANGUAGE_LABELS: Record<BabelLanguage, string> = {
  de: 'German',
  es: 'Spanish',
  fr: 'French',
  ja: 'Japanese',
  pt: 'Portuguese',
  en: 'English',
};

export const LANGUAGE_REGISTERS: Record<BabelLanguage, string> = {
  de: 'precision and technical rigor',
  es: 'embodied and emotional resonance',
  fr: 'nuance and philosophical distinction',
  ja: 'contextual sensitivity and hierarchical awareness',
  pt: 'relational warmth and social connection',
  en: 'general-purpose baseline',
};

// --- Clause Segmentation ---

export interface Clause {
  id: number;
  text: string;
  assertion?: string;
}

// --- Perplexity Scores ---

export interface LanguageScore {
  language: BabelLanguage;
  perplexity: number;
  logprob_mean: number;
  token_count: number;
  generation?: string;
}

export interface ClauseScores {
  clause: Clause;
  scores: LanguageScore[];
  winner: BabelLanguage;
  differential: number;
  spread: number;
}

// --- Confidence Mapping ---

export interface MappedConfidence {
  clause: Clause;
  score: number;
  basis: Basis;
  winner: BabelLanguage;
  differential: number;
  spread: number;
  generation?: string;
}

// --- Pipeline Configuration ---

export interface PipelineConfig {
  thresholds?: ConfidenceThresholds;
  includeGenerations?: boolean;
  languages?: BabelLanguage[];
  firstTokenSelection?: boolean;
  firstTokenDepth?: number;
}

export interface ConfidenceThresholds {
  strongDifferential: number;
  moderateDifferential: number;
  highSpread: number;
  moderateSpread: number;
  verifiedCeiling: number;
  derivedCeiling: number;
  patternMatchCeiling: number;
  speculationFloor: number;
}

export const DEFAULT_THRESHOLDS: ConfidenceThresholds = {
  strongDifferential: 0.35,
  moderateDifferential: 0.15,
  highSpread: 0.25,
  moderateSpread: 0.10,
  verifiedCeiling: 0.95,
  derivedCeiling: 0.75,
  patternMatchCeiling: 0.50,
  speculationFloor: 0.10,
};

// --- Pipeline Stages (interfaces for dependency injection) ---

export interface ClauseSegmenter {
  segment(text: string): Promise<Clause[]>;
}

export interface ParallelGenerator {
  generate(
    clause: Clause,
    languages: BabelLanguage[],
    config?: { registerDirected?: boolean; firstTokenOnly?: boolean; tokenDepth?: number }
  ): Promise<LanguageScore[]>;
}

export interface PipelineResult {
  clauses: MappedConfidence[];
  envelope: import('../types').BabelEnvelope;
  validation: import('../types').ValidationResult;
  metadata: {
    totalClauses: number;
    languages: BabelLanguage[];
    modelUsed?: string;
    generationTimeMs?: number;
    scoringTimeMs?: number;
  };
}
