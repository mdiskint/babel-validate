// ═══════════════════════════════════════════════
// BABEL — Pipeline Orchestrator
// ═══════════════════════════════════════════════

import { BabelEnvelope, Intent, Register, Affect, Ground, Trajectory, ValidationResult } from '../types';
import { validate } from '../validate';
import { EnvelopeBuilder } from '../builder';
import {
  Clause,
  ClauseScores,
  MappedConfidence,
  PipelineConfig,
  PipelineResult,
  ClauseSegmenter,
  ParallelGenerator,
  BABEL_LANGUAGES,
  BabelLanguage,
  DEFAULT_THRESHOLDS,
} from './types';
import { mapConfidence, computeClauseScores } from './mapper';

export interface PipelineOptions {
  segmenter: ClauseSegmenter;
  generator: ParallelGenerator;
  config?: PipelineConfig;
  envelope: {
    sender: string;
    recipient: string;
    intent: Intent;
    register: Register;
    chainId?: string;
    seq?: number;
    affect?: Affect;
    grounds?: Ground[];
    trajectory?: Trajectory;
  };
}

export async function runPipeline(
  inputText: string,
  options: PipelineOptions
): Promise<PipelineResult> {
  const config = options.config || {};
  const thresholds = config.thresholds || DEFAULT_THRESHOLDS;
  const languages = config.languages || [...BABEL_LANGUAGES];

  const genStart = Date.now();

  const clauses = await options.segmenter.segment(inputText);

  const allClauseScores: ClauseScores[] = [];

  for (const clause of clauses) {
    const languageScores = await options.generator.generate(clause, languages, {
      registerDirected: true,
      firstTokenOnly: config.firstTokenSelection,
      tokenDepth: config.firstTokenDepth,
    });

    const clauseScores = computeClauseScores(clause, languageScores);
    allClauseScores.push(clauseScores);
  }

  const genEnd = Date.now();

  const mappedConfidences: MappedConfidence[] = allClauseScores.map(cs =>
    mapConfidence(cs, thresholds)
  );

  const builder = new EnvelopeBuilder()
    .sender(options.envelope.sender)
    .recipient(options.envelope.recipient)
    .intent(options.envelope.intent)
    .register(options.envelope.register)
    .payload(inputText);

  if (options.envelope.chainId) {
    builder.chain(options.envelope.chainId, options.envelope.seq ?? 0);
  }

  if (options.envelope.affect) {
    builder.affect(
      options.envelope.affect.expansion,
      options.envelope.affect.activation,
      options.envelope.affect.certainty
    );
  }

  if (options.envelope.grounds) {
    for (const g of options.envelope.grounds) {
      builder.ground(g.constraint, g.authority, g.override);
    }
  }

  if (options.envelope.trajectory) {
    builder.withTrajectory(
      options.envelope.trajectory.pattern,
      options.envelope.trajectory.direction,
      {
        duration: options.envelope.trajectory.duration,
        prior_handoffs: options.envelope.trajectory.prior_handoffs,
      }
    );
  }

  for (const mc of mappedConfidences) {
    const assertion = mc.clause.assertion || mc.clause.text;
    builder.assert(assertion, mc.score, mc.basis);
  }

  const envelope = builder.build();
  const validation = validate(envelope);

  return {
    clauses: mappedConfidences,
    envelope,
    validation,
    metadata: {
      totalClauses: clauses.length,
      languages,
      generationTimeMs: genEnd - genStart,
    },
  };
}

export function buildMeasuredEnvelope(
  clauseScores: ClauseScores[],
  envelopeConfig: PipelineOptions['envelope'],
  payload: string,
  thresholds = DEFAULT_THRESHOLDS
): { envelope: BabelEnvelope; validation: ValidationResult; confidences: MappedConfidence[] } {
  const confidences = clauseScores.map(cs => mapConfidence(cs, thresholds));

  const builder = new EnvelopeBuilder()
    .sender(envelopeConfig.sender)
    .recipient(envelopeConfig.recipient)
    .intent(envelopeConfig.intent)
    .register(envelopeConfig.register)
    .payload(payload);

  if (envelopeConfig.chainId) {
    builder.chain(envelopeConfig.chainId, envelopeConfig.seq ?? 0);
  }

  if (envelopeConfig.affect) {
    builder.affect(
      envelopeConfig.affect.expansion,
      envelopeConfig.affect.activation,
      envelopeConfig.affect.certainty
    );
  }

  if (envelopeConfig.grounds) {
    for (const g of envelopeConfig.grounds) {
      builder.ground(g.constraint, g.authority, g.override);
    }
  }

  if (envelopeConfig.trajectory) {
    builder.withTrajectory(
      envelopeConfig.trajectory.pattern,
      envelopeConfig.trajectory.direction,
      {
        duration: envelopeConfig.trajectory.duration,
        prior_handoffs: envelopeConfig.trajectory.prior_handoffs,
      }
    );
  }

  for (const mc of confidences) {
    const assertion = mc.clause.assertion || mc.clause.text;
    builder.assert(assertion, mc.score, mc.basis);
  }

  const envelope = builder.build();
  const validation = validate(envelope);

  return { envelope, validation, confidences };
}

export class SimpleSegmenter implements ClauseSegmenter {
  async segment(text: string): Promise<Clause[]> {
    const sentences = text
      .split(/(?<=[.!?])\s+/)
      .map(s => s.trim())
      .filter(s => s.length > 0);

    return sentences.map((text, i) => ({
      id: i,
      text,
      assertion: text,
    }));
  }
}
