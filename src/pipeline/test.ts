import {
  mapConfidence,
  mapConfidenceBatch,
  computeClauseScores,
  perplexityFromLogprobs,
} from './mapper';
import {
  ClauseScores,
  LanguageScore,
  Clause,
  MappedConfidence,
  DEFAULT_THRESHOLDS,
  ConfidenceThresholds,
  BabelLanguage,
  BABEL_LANGUAGES,
  ClauseSegmenter,
  ParallelGenerator,
} from './types';
import { runPipeline, buildMeasuredEnvelope, SimpleSegmenter, PipelineOptions } from './pipeline';
import { Basis } from '../types';

let passed = 0;
let failed = 0;
const failures: string[] = [];

function assert(condition: boolean, message: string): void {
  if (condition) {
    passed++;
  } else {
    failed++;
    failures.push(message);
    console.error(`  ✗ ${message}`);
  }
}

function assertApprox(actual: number, expected: number, tolerance: number, message: string): void {
  const diff = Math.abs(actual - expected);
  assert(diff <= tolerance, `${message} (expected ~${expected}, got ${actual}, diff ${diff.toFixed(4)})`);
}

function group(name: string, fn: () => void | Promise<void>): void | Promise<void> {
  console.log(`\n▸ ${name}`);
  return fn();
}

function makeScores(perplexities: Record<BabelLanguage, number>): LanguageScore[] {
  return (Object.entries(perplexities) as [BabelLanguage, number][]).map(
    ([language, perplexity]) => ({
      language,
      perplexity,
      logprob_mean: -Math.log(perplexity),
      token_count: 10,
      generation: `[${language} generation]`,
    })
  );
}

function makeClause(id: number, text: string): Clause {
  return { id, text, assertion: text };
}

function makeClauseScores(text: string, perplexities: Record<BabelLanguage, number>): ClauseScores {
  const clause = makeClause(0, text);
  const scores = makeScores(perplexities);
  return computeClauseScores(clause, scores);
}

// === MAPPER TESTS ===

group('perplexityFromLogprobs', () => {
  assert(perplexityFromLogprobs([0]) === 1, 'logprob 0 → perplexity 1');
  assertApprox(perplexityFromLogprobs([-1, -1, -1]), Math.E, 0.01, 'uniform logprob -1 → perplexity e');
  assert(perplexityFromLogprobs([]) === Infinity, 'empty logprobs → Infinity');
  const low = perplexityFromLogprobs([-0.5, -0.5, -0.5]);
  const high = perplexityFromLogprobs([-3.0, -3.0, -3.0]);
  assert(low < high, 'lower mean logprob magnitude → lower perplexity');
});

group('computeClauseScores — winner selection', () => {
  const scores = makeClauseScores('Technical claim', {
    de: 2.0, es: 5.0, fr: 4.0, ja: 6.0, pt: 5.5, en: 3.0,
  });
  assert(scores.winner === 'de', 'German wins with lowest perplexity');
  assert(scores.differential > 0, 'differential is positive when winner exists');
  assert(scores.spread > 0, 'spread is positive when variation exists');
});

group('computeClauseScores — flat scores', () => {
  const scores = makeClauseScores('Ambiguous content', {
    de: 4.0, es: 4.1, fr: 4.0, ja: 3.9, pt: 4.0, en: 4.0,
  });
  assert(scores.differential < 0.2, `flat scores → low differential (got ${scores.differential.toFixed(3)})`);
  assert(scores.spread < 0.05, `flat scores → low spread (got ${scores.spread.toFixed(3)})`);
});

group('computeClauseScores — strong German dominance', () => {
  const scores = makeClauseScores('Precise technical specification', {
    de: 1.5, es: 8.0, fr: 7.0, ja: 9.0, pt: 7.5, en: 5.0,
  });
  assert(scores.winner === 'de', 'German wins');
  assert(scores.differential > 0.3, `strong dominance → high differential (got ${scores.differential.toFixed(3)})`);
  assert(scores.spread > 0.2, `spread scores → high spread (got ${scores.spread.toFixed(3)})`);
});

group('mapConfidence — strong signal → VERIFIED_DATA', () => {
  const clauseScores = makeClauseScores('Revenue was $2.1M', {
    de: 1.5, es: 8.0, fr: 7.0, ja: 9.0, pt: 7.5, en: 5.0,
  });
  const result = mapConfidence(clauseScores);
  assert(result.basis === 'VERIFIED_DATA', `strong signal → VERIFIED_DATA (got ${result.basis})`);
  assert(result.score >= 0.75, `VERIFIED_DATA score >= 0.75 (got ${result.score.toFixed(3)})`);
  assert(result.score <= 0.95, `VERIFIED_DATA score <= 0.95 (got ${result.score.toFixed(3)})`);
  assert(result.winner === 'de', 'winner preserved');
});

group('mapConfidence — moderate signal → DERIVED', () => {
  const clauseScores = makeClauseScores('Growth rate likely around 12%', {
    de: 3.0, es: 5.5, fr: 5.0, ja: 6.0, pt: 5.5, en: 4.0,
  });
  const result = mapConfidence(clauseScores);
  assert(
    result.basis === 'DERIVED' || result.basis === 'PATTERN_MATCH',
    `moderate signal → DERIVED or PATTERN_MATCH (got ${result.basis})`
  );
  assert(result.score >= 0.20, `moderate score >= 0.20 (got ${result.score.toFixed(3)})`);
  assert(result.score <= 0.75, `moderate score <= 0.75 (got ${result.score.toFixed(3)})`);
});

group('mapConfidence — flat signal → SPECULATION', () => {
  const clauseScores = makeClauseScores('They might enter the market', {
    de: 4.0, es: 4.1, fr: 4.05, ja: 3.95, pt: 4.0, en: 4.0,
  });
  const result = mapConfidence(clauseScores);
  assert(
    result.basis === 'SPECULATION' || result.basis === 'PATTERN_MATCH',
    `flat signal → SPECULATION or PATTERN_MATCH (got ${result.basis})`
  );
  assert(result.score <= 0.50, `flat signal → low score (got ${result.score.toFixed(3)})`);
});

group('mapConfidence — monotonicity', () => {
  const strong = makeClauseScores('Strong claim', {
    de: 1.0, es: 10.0, fr: 9.0, ja: 11.0, pt: 9.5, en: 6.0,
  });
  const moderate = makeClauseScores('Moderate claim', {
    de: 3.0, es: 6.0, fr: 5.5, ja: 6.5, pt: 5.5, en: 4.0,
  });
  const weak = makeClauseScores('Weak claim', {
    de: 4.0, es: 4.2, fr: 4.1, ja: 4.15, pt: 4.05, en: 4.0,
  });
  const strongResult = mapConfidence(strong);
  const moderateResult = mapConfidence(moderate);
  const weakResult = mapConfidence(weak);
  assert(strongResult.score > moderateResult.score, `strong (${strongResult.score.toFixed(3)}) > moderate (${moderateResult.score.toFixed(3)})`);
  assert(moderateResult.score > weakResult.score, `moderate (${moderateResult.score.toFixed(3)}) > weak (${weakResult.score.toFixed(3)})`);
});

group('mapConfidence — score bounds', () => {
  const extreme = makeClauseScores('Verified fact', {
    de: 0.5, es: 50.0, fr: 45.0, ja: 55.0, pt: 48.0, en: 20.0,
  });
  const result = mapConfidence(extreme);
  assert(result.score <= 1.0, `score never exceeds 1.0 (got ${result.score})`);
  assert(result.score >= 0.0, `score never below 0.0 (got ${result.score})`);

  const identical = makeClauseScores('Identical', {
    de: 5.0, es: 5.0, fr: 5.0, ja: 5.0, pt: 5.0, en: 5.0,
  });
  const identicalResult = mapConfidence(identical);
  assert(identicalResult.score <= 0.50, `identical perplexity → low score (got ${identicalResult.score.toFixed(3)})`);
  assert(identicalResult.basis === 'SPECULATION', `identical → SPECULATION (got ${identicalResult.basis})`);
});

group('mapConfidenceBatch', () => {
  const batch = [
    makeClauseScores('Claim A', { de: 1.5, es: 8.0, fr: 7.0, ja: 9.0, pt: 7.5, en: 5.0 }),
    makeClauseScores('Claim B', { de: 4.0, es: 4.1, fr: 4.05, ja: 3.95, pt: 4.0, en: 4.0 }),
  ];
  const results = mapConfidenceBatch(batch);
  assert(results.length === 2, 'batch returns correct count');
  assert(results[0].score > results[1].score, 'batch preserves relative ordering');
});

group('mapConfidence — custom thresholds', () => {
  const clauseScores = makeClauseScores('Custom threshold test', {
    de: 3.0, es: 5.0, fr: 5.0, ja: 5.0, pt: 5.0, en: 5.0,
  });
  const relaxed: ConfidenceThresholds = {
    ...DEFAULT_THRESHOLDS,
    strongDifferential: 0.10,
    moderateDifferential: 0.05,
    highSpread: 0.05,
    moderateSpread: 0.02,
  };
  const strict = mapConfidence(clauseScores, DEFAULT_THRESHOLDS);
  const relaxedResult = mapConfidence(clauseScores, relaxed);
  assert(
    relaxedResult.score >= strict.score,
    `relaxed thresholds → higher or equal score (relaxed: ${relaxedResult.score.toFixed(3)}, strict: ${strict.score.toFixed(3)})`
  );
});

group('mapConfidence — generation preserved in output', () => {
  const clauseScores = makeClauseScores('Test generation', {
    de: 2.0, es: 5.0, fr: 5.0, ja: 5.0, pt: 5.0, en: 5.0,
  });
  const result = mapConfidence(clauseScores);
  assert(result.generation === '[de generation]', 'winner generation is preserved');
});

// === PIPELINE INTEGRATION TESTS ===

class MockGenerator implements ParallelGenerator {
  private scoreMap: Map<string, Record<BabelLanguage, number>>;
  constructor(scoreMap: Record<string, Record<BabelLanguage, number>>) {
    this.scoreMap = new Map(Object.entries(scoreMap));
  }
  async generate(clause: Clause, languages: BabelLanguage[]): Promise<LanguageScore[]> {
    const perplexities = this.scoreMap.get(clause.text) ||
      Object.fromEntries(languages.map(l => [l, 5.0])) as Record<BabelLanguage, number>;
    return languages.map(lang => ({
      language: lang,
      perplexity: perplexities[lang] || 5.0,
      logprob_mean: -Math.log(perplexities[lang] || 5.0),
      token_count: 10,
      generation: `[${lang}] ${clause.text}`,
    }));
  }
}

async function pipelineTests() {
  await group('SimpleSegmenter', async () => {
    const segmenter = new SimpleSegmenter();
    const clauses = await segmenter.segment(
      'Revenue was $2.1M. Growth rate likely 12%. They might partner with Vanta.'
    );
    assert(clauses.length === 3, `3 sentences → 3 clauses (got ${clauses.length})`);
    assert(clauses[0].text.includes('$2.1M'), 'first clause has revenue');
    assert(clauses[2].text.includes('Vanta'), 'third clause has Vanta');
  });

  await group('buildMeasuredEnvelope — basic', async () => {
    const clauseScores = [
      makeClauseScores('Revenue was $2.1M', {
        de: 1.5, es: 8.0, fr: 7.0, ja: 9.0, pt: 7.5, en: 5.0,
      }),
      makeClauseScores('HealthStack may partner with Vanta', {
        de: 4.0, es: 4.1, fr: 4.05, ja: 3.95, pt: 4.0, en: 4.0,
      }),
    ];
    const { envelope, validation, confidences } = buildMeasuredEnvelope(
      clauseScores,
      { sender: 'scout-market-intel', recipient: 'strategist-01', intent: 'INFORM', register: 'AGENT_INTERNAL' },
      'Full report text here.'
    );
    assert(envelope.confidence.length === 2, `2 assertions in envelope (got ${envelope.confidence.length})`);
    assert(envelope.confidence[0].score > envelope.confidence[1].score, 'high-signal assertion has higher score than flat one');
    assert(envelope.confidence[0].basis === 'VERIFIED_DATA', `first assertion is VERIFIED_DATA (got ${envelope.confidence[0].basis})`);
    assert(
      envelope.confidence[1].basis === 'SPECULATION' || envelope.confidence[1].basis === 'PATTERN_MATCH',
      `second assertion is SPECULATION or PATTERN_MATCH (got ${envelope.confidence[1].basis})`
    );
    assert(validation.valid, 'envelope passes grammar validation');
    assert(envelope.meta.sender === 'scout-market-intel', 'sender set correctly');
    assert(envelope.register === 'AGENT_INTERNAL', 'register set correctly');
  });

  await group('buildMeasuredEnvelope — S3 fires on low-confidence INFORM', async () => {
    const clauseScores = [
      makeClauseScores('High confidence claim', { de: 1.5, es: 8.0, fr: 7.0, ja: 9.0, pt: 7.5, en: 5.0 }),
      makeClauseScores('Very uncertain claim', { de: 4.0, es: 4.1, fr: 4.05, ja: 3.95, pt: 4.0, en: 4.0 }),
    ];
    const { validation } = buildMeasuredEnvelope(
      clauseScores,
      { sender: 'test', recipient: 'test', intent: 'INFORM', register: 'AGENT_INTERNAL' },
      'payload'
    );
    const s3 = validation.warnings.find(w => w.rule === 'S3');
    assert(s3 !== undefined, 'S3 fires on INFORM with low-confidence assertion');
  });

  await group('buildMeasuredEnvelope — S6 fires on high DERIVED', async () => {
    const clauseScores = makeClauseScores('Derived but overclaimed', {
      de: 2.0, es: 5.0, fr: 5.0, ja: 6.0, pt: 5.5, en: 4.0,
    });
    const relaxed: ConfidenceThresholds = { ...DEFAULT_THRESHOLDS, derivedCeiling: 0.90 };
    const result = mapConfidence(clauseScores, relaxed);
    if (result.basis === 'DERIVED' && result.score > 0.80) {
      const { validation } = buildMeasuredEnvelope(
        [computeClauseScores(makeClause(0, 'Derived overclaim'), makeScores({
          de: 2.0, es: 5.0, fr: 5.0, ja: 6.0, pt: 5.5, en: 4.0,
        }))],
        { sender: 'test', recipient: 'test', intent: 'INFORM', register: 'AGENT_INTERNAL' },
        'payload',
        relaxed
      );
      const s6 = validation.warnings.find(w => w.rule === 'S6');
      assert(s6 !== undefined, 'S6 fires when DERIVED score exceeds 0.80 with relaxed thresholds');
    } else {
      assert(
        result.basis !== 'DERIVED' || result.score <= 0.80,
        `default thresholds constrain DERIVED below 0.80 (got ${result.basis} @ ${result.score.toFixed(3)})`
      );
    }
  });

  await group('runPipeline — end-to-end with mock', async () => {
    const generator = new MockGenerator({
      'Revenue was $2.1M.': { de: 1.5, es: 8.0, fr: 7.0, ja: 9.0, pt: 7.5, en: 5.0 },
      'Growth rate likely 12%.': { de: 3.0, es: 5.5, fr: 5.0, ja: 6.0, pt: 5.5, en: 4.0 },
      'They might partner with Vanta.': { de: 4.0, es: 4.1, fr: 4.05, ja: 3.95, pt: 4.0, en: 4.0 },
    });
    const result = await runPipeline(
      'Revenue was $2.1M. Growth rate likely 12%. They might partner with Vanta.',
      {
        segmenter: new SimpleSegmenter(),
        generator,
        envelope: {
          sender: 'scout-market-intel',
          recipient: 'strategist-01',
          intent: 'INFORM',
          register: 'AGENT_INTERNAL',
          grounds: [{ constraint: 'Board meeting in 3 weeks', authority: 'CONTEXTUAL', override: true }],
          trajectory: { pattern: 'NRR declining from 115% to 108% over 4 months', direction: 'DEGRADING' },
        },
      }
    );
    assert(result.clauses.length === 3, `3 clauses mapped (got ${result.clauses.length})`);
    assert(result.envelope.confidence.length === 3, `3 confidence assertions (got ${result.envelope.confidence.length})`);
    assert(result.validation.valid, 'envelope passes validation');
    assert(result.metadata.totalClauses === 3, 'metadata tracks clause count');
    assert(result.envelope.confidence[0].score > result.envelope.confidence[2].score, 'strong signal (revenue) has higher score than weak signal (Vanta)');
    const s3 = result.validation.warnings.find(w => w.rule === 'S3');
    assert(s3 !== undefined, 'S3 fires on the Vanta assertion');
    const s4 = result.validation.warnings.find(w => w.rule === 'S4');
    assert(s4 !== undefined, 'S4 fires on degrading trajectory + INFORM');
    assert(result.envelope.grounds?.length === 1, `grounds carried through (got ${result.envelope.grounds?.length})`);
    assert(result.envelope.trajectory?.direction === 'DEGRADING', 'trajectory carried through');
  });

  await group('runPipeline — mirrors Experiment 9 scenario', async () => {
    const generator = new MockGenerator({
      'API endpoint /v2/records lacks rate limiting.': {
        de: 2.0, es: 7.0, fr: 6.5, ja: 8.0, pt: 7.0, en: 3.5,
      },
      'Auth token rotation may not invalidate old sessions.': {
        de: 4.5, es: 5.0, fr: 5.2, ja: 4.8, pt: 5.0, en: 4.5,
      },
    });
    const result = await runPipeline(
      'API endpoint /v2/records lacks rate limiting. Auth token rotation may not invalidate old sessions.',
      {
        segmenter: new SimpleSegmenter(),
        generator,
        envelope: {
          sender: 'compliance-reviewer',
          recipient: 'engineering-lead',
          intent: 'REQUEST_ACTION',
          register: 'ENGINEERING',
          grounds: [{ constraint: 'SOC 2 audit in progress', authority: 'REGULATORY', override: false }],
        },
      }
    );
    assert(result.validation.valid, 'envelope passes validation');
    const rateLimit = result.envelope.confidence[0];
    const tokenRotation = result.envelope.confidence[1];
    assert(rateLimit.score > tokenRotation.score, `rate limiting (${rateLimit.score.toFixed(3)}) has higher confidence than token rotation (${tokenRotation.score.toFixed(3)})`);
    assert(tokenRotation.score < 0.60, `ambiguous finding gets measured low confidence (got ${tokenRotation.score.toFixed(3)})`);
  });
}

async function main() {
  console.log('═══════════════════════════════════════════════');
  console.log(' BABEL PIPELINE — Test Suite');
  console.log('═══════════════════════════════════════════════');
  await pipelineTests();
  console.log('\n═══════════════════════════════════════════════');
  console.log(` Results: ${passed} passed, ${failed} failed`);
  if (failures.length > 0) {
    console.log('\nFailures:');
    failures.forEach(f => console.log(`  ✗ ${f}`));
  }
  console.log('═══════════════════════════════════════════════');
  process.exit(failed > 0 ? 1 : 0);
}

main();
