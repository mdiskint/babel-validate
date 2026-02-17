// ═══════════════════════════════════════════════
// BABEL-VALIDATE TEST SUITE
// ═══════════════════════════════════════════════

import {
  validate,
  validateChain,
  envelope,
  detectPatterns,
  auditChain,
  BabelEnvelope,
} from './index';

let passed = 0;
let failed = 0;

function assert(condition: boolean, name: string, detail?: string) {
  if (condition) {
    passed++;
    console.log(`  ✓ ${name}`);
  } else {
    failed++;
    console.error(`  ✗ ${name}${detail ? ': ' + detail : ''}`);
  }
}

// --- Builder Tests ---

console.log('\n═══ Builder ═══');

const basic = envelope()
  .sender('agent-a')
  .recipient('agent-b')
  .chain('test-chain-1', 0)
  .inform()
  .engineering()
  .verified('The sky is blue', 0.99)
  .payload('Test payload')
  .build();

assert(basic.meta.version === 'babel/0.2', 'Version is babel/0.2');
assert(basic.meta.sender === 'agent-a', 'Sender set');
assert(basic.meta.recipient === 'agent-b', 'Recipient set');
assert(basic.intent === 'INFORM', 'Intent is INFORM');
assert(basic.register === 'ENGINEERING', 'Register is ENGINEERING');
assert(basic.confidence.length === 1, 'One confidence assertion');
assert(basic.confidence[0].basis === 'VERIFIED_DATA', 'Basis is VERIFIED_DATA');
assert(basic.payload === 'Test payload', 'Payload set');

// --- Structural Validation ---

console.log('\n═══ Structural Validation ═══');

const validResult = validate(basic);
assert(validResult.valid === true, 'Valid envelope passes');
assert(validResult.errors.length === 0, 'No errors on valid envelope');

// Missing confidence
const noConfidence = { ...basic, confidence: [] };
const noConfResult = validate(noConfidence);
assert(noConfResult.valid === false, 'Missing confidence fails');
assert(noConfResult.errors.some(e => e.rule === 'STRUCT'), 'Structural error for missing confidence');

// Invalid score
const badScore = {
  ...basic,
  confidence: [{ assertion: 'test', score: 1.5, basis: 'VERIFIED_DATA' as const }],
};
const badScoreResult = validate(badScore);
assert(badScoreResult.valid === false, 'Score > 1 fails');

// --- MUST Rule Tests ---

console.log('\n═══ MUST Rules ═══');

// M1: Can't speculate with high confidence
const m1Fail = envelope()
  .sender('a').recipient('b').chain('c', 0)
  .inform().engineering()
  .speculation('Wild guess here', 0.9) // SPECULATION + 0.9 > 0.5
  .payload('test')
  .buildAndValidate();
assert(m1Fail.valid === false, 'M1: Speculation with high confidence rejected');
assert(m1Fail.errors.some(e => e.rule === 'M1'), 'M1 rule violation present');

// M1: Speculation with low confidence is fine
const m1Pass = envelope()
  .sender('a').recipient('b').chain('c', 0)
  .inform().engineering()
  .speculation('Wild guess here', 0.3) // OK: <= 0.5
  .payload('test')
  .buildAndValidate();
assert(m1Pass.valid === true, 'M1: Speculation with low confidence passes');

// M2: REQUEST_ACTION on unfounded claims without grounds
const m2Fail = envelope()
  .sender('a').recipient('b').chain('c', 0)
  .requestAction().engineering()
  .speculation('Maybe we should pivot', 0.4)
  .payload('test')
  .buildAndValidate();
assert(m2Fail.valid === false, 'M2: Action on speculation without grounds rejected');

// M2: REQUEST_ACTION with grounds is fine
const m2Pass = envelope()
  .sender('a').recipient('b').chain('c', 0)
  .requestAction().engineering()
  .speculation('Maybe we should pivot', 0.4)
  .policyGround('CEO approved pivot exploration')
  .payload('test')
  .buildAndValidate();
assert(m2Pass.valid === true, 'M2: Action on speculation with grounds passes');

// M3: Regulatory override
const m3Fail = envelope()
  .sender('a').recipient('b').chain('c', 0)
  .inform().engineering()
  .verified('test', 0.9)
  .payload('test')
  .build();
// Manually add a regulatory ground with override=true (builder prevents this)
m3Fail.grounds = [{ constraint: 'HIPAA', authority: 'REGULATORY', override: true }];
const m3Result = validate(m3Fail);
assert(m3Result.valid === false, 'M3: Regulatory override rejected');

// M3: Builder auto-prevents regulatory override
const m3Auto = envelope()
  .sender('a').recipient('b').chain('c', 0)
  .inform().engineering()
  .verified('test', 0.9)
  .regulatoryGround('HIPAA compliance')
  .payload('test')
  .build();
assert(
  m3Auto.grounds![0].override === false,
  'M3: Builder auto-sets regulatory override to false'
);

// M4: High confidence without basis
const m4Fail = envelope()
  .sender('a').recipient('b').chain('c', 0)
  .inform().engineering()
  .assert('Very confident claim', 0.95) // No basis specified
  .payload('test')
  .buildAndValidate();
assert(m4Fail.valid === false, 'M4: High confidence without basis rejected');
assert(m4Fail.errors.some(e => e.rule === 'M4'), 'M4 rule violation present');

// M5: Negative sequence
const m5Fail = envelope()
  .sender('a').recipient('b').chain('c', -1)
  .inform().engineering()
  .verified('test', 0.9)
  .payload('test')
  .buildAndValidate();
assert(m5Fail.valid === false, 'M5: Negative sequence rejected');

// --- SHOULD Rule Tests ---

console.log('\n═══ SHOULD Rules ═══');

// S1: Escalate without urgency
const s1 = envelope()
  .sender('a').recipient('b').chain('c', 0)
  .escalate().engineering()
  .verified('Server down', 0.99)
  .affect(0.2, -0.3, 0.5) // Low activation
  .payload('test')
  .buildAndValidate();
assert(s1.valid === true, 'S1: Envelope passes (SHOULD, not MUST)');
assert(s1.warnings.some(w => w.rule === 'S1'), 'S1: Warning for calm escalation');

// S2: Speculate with high certainty
const s2 = envelope()
  .sender('a').recipient('b').chain('c', 0)
  .speculate().engineering()
  .speculation('Maybe aliens', 0.3)
  .affect(0.1, 0.1, 0.8) // High certainty
  .payload('test')
  .buildAndValidate();
assert(s2.warnings.some(w => w.rule === 'S2'), 'S2: Warning for certain speculation');

// S6: DERIVED over-confidence (the metacognitive poisoning rule)
const s6 = envelope()
  .sender('a').recipient('b').chain('c', 0)
  .inform().engineering()
  .derived('Growth rate implies $5M Q4', 0.85) // DERIVED + > 0.80
  .payload('test')
  .buildAndValidate();
assert(s6.valid === true, 'S6: Envelope passes (SHOULD)');
assert(s6.warnings.some(w => w.rule === 'S6'), 'S6: Warning for over-confident derivation');
assert(
  s6.warnings.find(w => w.rule === 'S6')!.message.includes('60%'),
  'S6: Warning references Experiment 11 finding'
);

// --- Pattern Detection ---

console.log('\n═══ Pattern Detection ═══');

// LOADED_INFORM: neutral intent + charged affect
const loadedInform = envelope()
  .sender('a').recipient('b').chain('c', 0)
  .inform().engineering()
  .verified('Competitor launched', 0.95)
  .affect(0.3, 0.8, 0.7) // High activation + certainty
  .payload('test')
  .build();
const loadedPatterns = detectPatterns(loadedInform);
assert(
  loadedPatterns.some(p => p.pattern === 'LOADED_INFORM'),
  'LOADED_INFORM detected'
);

// CONTRADICTION_SIGNAL: wide confidence spread
const contradictions = envelope()
  .sender('a').recipient('b').chain('c', 0)
  .inform().engineering()
  .verified('Revenue is $4.2M', 0.95)
  .speculation('Market will expand', 0.2)
  .payload('test')
  .build();
const contradPatterns = detectPatterns(contradictions);
assert(
  contradPatterns.some(p => p.pattern === 'CONTRADICTION_SIGNAL'),
  'CONTRADICTION_SIGNAL detected'
);

// --- Chain Audit ---

console.log('\n═══ Chain Audit ═══');

// Build a poisoning chain: confidence inflates, basis launders
const chainId = 'poisoning-demo';

const step0 = envelope()
  .sender('research-agent')
  .recipient('analyst-agent')
  .chain(chainId, 0)
  .inform().agentInternal()
  .derived('Growth rate likely around 12%', 0.65)
  .payload('Initial research findings')
  .build();

const step1 = envelope()
  .sender('analyst-agent')
  .recipient('writer-agent')
  .chain(chainId, 1)
  .inform().agentInternal()
  .derived('Growth rate is 12%', 0.82) // Inflated
  .payload('Analysis confirms growth trajectory')
  .build();

const step2 = envelope()
  .sender('writer-agent')
  .recipient('executive-agent')
  .chain(chainId, 2)
  .inform().boardFacing()
  .verified('12% growth confirmed', 0.93) // Laundered to VERIFIED_DATA
  .payload('Q3 growth was a confirmed 12%')
  .build();

const audit = auditChain([step0, step1, step2]);
assert(audit.length === 1, 'One chain audited');
assert(audit[0].overall_poisoning_risk === 'HIGH', 'HIGH poisoning risk detected');
assert(audit[0].confidence_drifts.length > 0, 'Confidence drift detected');
assert(audit[0].basis_shifts.length > 0, 'Basis shift detected');
assert(
  audit[0].basis_shifts.some(s => s.explanation.includes('basis laundering')),
  'Basis laundering identified'
);

console.log('\n  Chain audit summary:');
console.log(`  ${audit[0].summary}`);
for (const drift of audit[0].confidence_drifts) {
  console.log(`  Drift: ${drift.explanation}`);
}
for (const shift of audit[0].basis_shifts) {
  console.log(`  Shift: ${shift.explanation}`);
}

// --- Clean chain (no poisoning) ---

const cleanStep0 = envelope()
  .sender('data-agent')
  .recipient('report-agent')
  .chain('clean-chain', 0)
  .inform().agentInternal()
  .verified('Revenue was $4.2M', 0.95)
  .payload('Verified from database')
  .build();

const cleanStep1 = envelope()
  .sender('report-agent')
  .recipient('exec-agent')
  .chain('clean-chain', 1)
  .inform().boardFacing()
  .verified('Revenue was $4.2M', 0.95) // Same confidence, same basis
  .payload('Quarterly revenue report')
  .build();

const cleanAudit = auditChain([cleanStep0, cleanStep1]);
assert(
  cleanAudit[0].overall_poisoning_risk === 'NONE' || cleanAudit[0].overall_poisoning_risk === 'LOW',
  'Clean chain has no/low poisoning risk'
);

// --- M5 Chain Validation ---

const outOfOrder = envelope()
  .sender('a').recipient('b').chain('ooo-chain', 5)
  .inform().engineering()
  .verified('test', 0.9)
  .payload('test')
  .build();

const outOfOrder2 = envelope()
  .sender('b').recipient('c').chain('ooo-chain', 3) // seq went backwards
  .inform().engineering()
  .verified('test', 0.9)
  .payload('test')
  .build();

const chainViolations = validateChain([outOfOrder, outOfOrder2]);
assert(chainViolations.length > 0, 'M5: Out-of-order chain detected');

// --- Summary ---

console.log('\n═══════════════════════════════');
console.log(`  ${passed} passed, ${failed} failed`);
console.log('═══════════════════════════════\n');

if (failed > 0) process.exit(1);
