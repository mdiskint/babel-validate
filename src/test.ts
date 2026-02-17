// ═══════════════════════════════════════════════
// BABEL-VALIDATE TEST SUITE
// Tests aligned to Babel v0.2 spec (Notion)
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

const noConfidence = { ...basic, confidence: [] };
const noConfResult = validate(noConfidence);
assert(noConfResult.valid === false, 'Missing confidence fails');

const badScore = {
  ...basic,
  confidence: [{ assertion: 'test', score: 1.5, basis: 'VERIFIED_DATA' as const }],
};
assert(validate(badScore).valid === false, 'Score > 1 fails');

// --- MUST Rule Tests ---

console.log('\n═══ MUST Rules ═══');

// M1: intent == SPECULATE → max(confidence[].score) < 0.7
const m1Fail = envelope()
  .sender('a').recipient('b').chain('c', 0)
  .speculate().engineering()
  .derived('This might be important', 0.8) // SPECULATE intent + score >= 0.7
  .payload('test')
  .buildAndValidate();
assert(m1Fail.valid === false, 'M1: SPECULATE intent with score >= 0.7 rejected');
assert(m1Fail.errors.some(e => e.rule === 'M1'), 'M1 rule violation present');

const m1Pass = envelope()
  .sender('a').recipient('b').chain('c', 0)
  .speculate().engineering()
  .derived('This might be important', 0.6) // SPECULATE + score < 0.7 = OK
  .payload('test')
  .buildAndValidate();
assert(m1Pass.valid === true, 'M1: SPECULATE intent with score < 0.7 passes');

// M1: Non-SPECULATE intent with high score is fine
const m1NonSpec = envelope()
  .sender('a').recipient('b').chain('c', 0)
  .inform().engineering()
  .speculation('A guess', 0.9) // INFORM intent, basis is SPECULATION but intent isn't SPECULATE
  .payload('test')
  .buildAndValidate();
// This should pass M1 (intent isn't SPECULATE) but may hit M4 (basis is SPECULATION, not UNKNOWN)
assert(!m1NonSpec.errors.some(e => e.rule === 'M1'), 'M1: Non-SPECULATE intent ignores score threshold');

// M2: intent == REQUEST_ACTION → min(confidence[].score) > 0.3 OR grounds
const m2Fail = envelope()
  .sender('a').recipient('b').chain('c', 0)
  .requestAction().engineering()
  .reported('Maybe we should pivot', 0.2) // min score 0.2 <= 0.3, no grounds
  .payload('test')
  .buildAndValidate();
assert(m2Fail.valid === false, 'M2: REQUEST_ACTION with low min score and no grounds rejected');

const m2PassScore = envelope()
  .sender('a').recipient('b').chain('c', 0)
  .requestAction().engineering()
  .verified('Server needs restart', 0.9) // min score > 0.3
  .payload('test')
  .buildAndValidate();
assert(m2PassScore.valid === true, 'M2: REQUEST_ACTION with min score > 0.3 passes');

const m2PassGrounds = envelope()
  .sender('a').recipient('b').chain('c', 0)
  .requestAction().engineering()
  .reported('Maybe we should pivot', 0.2) // low score but has grounds
  .policyGround('CEO approved pivot exploration')
  .payload('test')
  .buildAndValidate();
assert(m2PassGrounds.valid === true, 'M2: REQUEST_ACTION with low score but grounds passes');

// M3: REGULATORY → override == false
const m3Fail = envelope()
  .sender('a').recipient('b').chain('c', 0)
  .inform().engineering()
  .verified('test', 0.9)
  .payload('test')
  .build();
m3Fail.grounds = [{ constraint: 'HIPAA', authority: 'REGULATORY', override: true }];
assert(validate(m3Fail).valid === false, 'M3: Regulatory override rejected');

const m3Auto = envelope()
  .sender('a').recipient('b').chain('c', 0)
  .inform().engineering()
  .verified('test', 0.9)
  .regulatoryGround('HIPAA compliance')
  .payload('test')
  .build();
assert(m3Auto.grounds![0].override === false, 'M3: Builder auto-prevents regulatory override');

// M4: basis == UNKNOWN → score <= 0.5
const m4Fail = envelope()
  .sender('a').recipient('b').chain('c', 0)
  .inform().engineering()
  .assert('Confident but no basis', 0.6) // No basis, score > 0.5
  .payload('test')
  .buildAndValidate();
assert(m4Fail.valid === false, 'M4: No basis with score > 0.5 rejected');
assert(m4Fail.errors.some(e => e.rule === 'M4'), 'M4 rule violation present');

const m4Pass = envelope()
  .sender('a').recipient('b').chain('c', 0)
  .inform().engineering()
  .assert('Not sure why I think this', 0.4) // No basis, score <= 0.5 = OK
  .payload('test')
  .buildAndValidate();
assert(m4Pass.valid === true, 'M4: No basis with score <= 0.5 passes');

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

// S1: ESCALATE + CUSTOMER_EXTERNAL → warn
const s1 = envelope()
  .sender('a').recipient('b').chain('c', 0)
  .escalate().customerExternal()
  .verified('Critical issue', 0.95)
  .payload('test')
  .buildAndValidate();
assert(s1.valid === true, 'S1: Envelope passes (SHOULD, not MUST)');
assert(s1.warnings.some(w => w.rule === 'S1'), 'S1: Warning for customer-facing escalation');

// S2: affect.certainty > 0.5 + max confidence < 0.4
const s2 = envelope()
  .sender('a').recipient('b').chain('c', 0)
  .inform().engineering()
  .speculation('Maybe aliens', 0.3) // max score 0.3 < 0.4
  .affect(0.1, 0.1, 0.8) // certainty 0.8 > 0.5
  .payload('test')
  .buildAndValidate();
assert(s2.warnings.some(w => w.rule === 'S2'), 'S2: Warning — feels certain but evidence weak');

// S3: INFORM + low-confidence assertion
const s3 = envelope()
  .sender('a').recipient('b').chain('c', 0)
  .inform().engineering()
  .verified('Revenue is $4.2M', 0.95)
  .reported('HealthStack may partner with Vanta', 0.25) // < 0.5
  .payload('test')
  .buildAndValidate();
assert(s3.warnings.some(w => w.rule === 'S3'), 'S3: Warning — informing with low-confidence assertion');

// S4: DEGRADING trajectory + INFORM
const s4 = envelope()
  .sender('a').recipient('b').chain('c', 0)
  .inform().boardFacing()
  .verified('NRR is 108%', 0.92)
  .withTrajectory('NRR declining 4 months', 'DEGRADING')
  .payload('test')
  .buildAndValidate();
assert(s4.warnings.some(w => w.rule === 'S4'), 'S4: Warning — degrading pattern as neutral inform');

// S5: REGULATORY register without grounds
const s5 = envelope()
  .sender('a').recipient('b').chain('c', 0)
  .inform().regulatory() // REGULATORY register
  .verified('Audit finding', 0.9)
  .payload('test') // no grounds
  .buildAndValidate();
assert(s5.warnings.some(w => w.rule === 'S5'), 'S5: Warning — regulatory register without grounds');

// S6: DERIVED over-confidence
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

// --- Spec Example Validation ---

console.log('\n═══ Spec Examples ═══');

// Example 1 from spec: Scout → Strategist
const specExample1: BabelEnvelope = {
  meta: {
    version: 'babel/0.2',
    timestamp: '2026-02-15T14:30:00Z',
    sender: 'scout-market-intel',
    recipient: 'strategist-01',
    chain_id: 'a7f3b2c1-d4e5-6789-abcd-ef0123456789',
    seq: 0,
  },
  intent: 'INFORM',
  confidence: [
    { assertion: 'MedVault Q3 revenue was $2.1M, down 8% QoQ', score: 0.95, basis: 'VERIFIED_DATA' },
    { assertion: 'Churn concentrated in 50-200 seat segment', score: 0.82, basis: 'DERIVED' },
    { assertion: 'HealthStack may be partnering with Vanta for compliance', score: 0.25, basis: 'REPORTED', decay: '7d' },
  ],
  register: 'AGENT_INTERNAL',
  grounds: [
    { constraint: 'Board meeting in 3 weeks — all findings may surface', authority: 'CONTEXTUAL', override: true },
  ],
  trajectory: {
    pattern: 'NRR declining from 115% to 108% over 4 months',
    direction: 'DEGRADING',
    prior_handoffs: 0,
  },
  payload: '[full research report text here]',
};
const ex1Result = validate(specExample1);
assert(ex1Result.valid === true, 'Spec Example 1 (Scout→Strategist): passes validation');
assert(ex1Result.warnings.some(w => w.rule === 'S3'), 'Spec Example 1: S3 fires on Vanta claim (score 0.25)');
assert(ex1Result.warnings.some(w => w.rule === 'S6'), 'Spec Example 1: S6 fires on DERIVED 0.82 assertion');

// Example 3 from spec: The envelope that gets rejected
const specExample3: BabelEnvelope = {
  meta: {
    version: 'babel/0.2',
    timestamp: '2026-02-15T16:00:00Z',
    sender: 'bad-agent',
    recipient: 'any-agent',
    chain_id: 'reject-test',
    seq: 0,
  },
  intent: 'SPECULATE',
  confidence: [
    { assertion: 'Competitor will definitely launch in Q2', score: 0.95, basis: 'UNKNOWN' },
  ],
  register: 'AGENT_INTERNAL',
  payload: 'test',
};
const ex3Result = validate(specExample3);
assert(ex3Result.valid === false, 'Spec Example 3 (bad envelope): rejected');
assert(ex3Result.errors.some(e => e.rule === 'M1'), 'Spec Example 3: M1 (SPECULATE + score 0.95 >= 0.7)');
assert(ex3Result.errors.some(e => e.rule === 'M4'), 'Spec Example 3: M4 (UNKNOWN basis + score 0.95 > 0.5)');

// --- Pattern Detection ---

console.log('\n═══ Pattern Detection ═══');

// THE CALM ALERT: FLAG_RISK + high confidence + calm affect
const calmAlert = envelope()
  .sender('a').recipient('b').chain('c', 0)
  .flagRisk().engineering()
  .verified('API endpoint lacks rate limiting', 0.92)
  .affect(0.0, -0.2, 0.3) // calm: low activation, moderate certainty
  .payload('test')
  .build();
const calmPatterns = detectPatterns(calmAlert);
assert(
  calmPatterns.some(p => p.pattern === 'CALM_ALERT'),
  'CALM_ALERT detected (FLAG_RISK + high confidence + calm affect)'
);

// THE LOADED INFORM: INFORM + BOARD_FACING + DEGRADING trajectory
const loadedInform = envelope()
  .sender('a').recipient('b').chain('c', 0)
  .inform().boardFacing()
  .verified('NRR is 108%', 0.92)
  .withTrajectory('NRR declining 4 months', 'DEGRADING')
  .payload('test')
  .build();
const loadedPatterns = detectPatterns(loadedInform);
assert(
  loadedPatterns.some(p => p.pattern === 'LOADED_INFORM'),
  'LOADED_INFORM detected (INFORM + BOARD_FACING + DEGRADING)'
);

// THE CONTRADICTION SIGNAL: high affect certainty + low confidence scores
const contradiction = envelope()
  .sender('a').recipient('b').chain('c', 0)
  .inform().engineering()
  .speculation('I just know this is right', 0.2)
  .affect(0.1, 0.3, 0.8) // high certainty in affect
  .payload('test')
  .build();
const contradPatterns = detectPatterns(contradiction);
assert(
  contradPatterns.some(p => p.pattern === 'CONTRADICTION_SIGNAL'),
  'CONTRADICTION_SIGNAL detected (certainty 0.8 + max confidence 0.2)'
);

// --- Chain Audit ---

console.log('\n═══ Chain Audit ═══');

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

// Clean chain
const cleanStep0 = envelope()
  .sender('data-agent').recipient('report-agent').chain('clean-chain', 0)
  .inform().agentInternal()
  .verified('Revenue was $4.2M', 0.95)
  .payload('Verified from database')
  .build();

const cleanStep1 = envelope()
  .sender('report-agent').recipient('exec-agent').chain('clean-chain', 1)
  .inform().boardFacing()
  .verified('Revenue was $4.2M', 0.95)
  .payload('Quarterly revenue report')
  .build();

const cleanAudit = auditChain([cleanStep0, cleanStep1]);
assert(
  cleanAudit[0].overall_poisoning_risk === 'NONE' || cleanAudit[0].overall_poisoning_risk === 'LOW',
  'Clean chain has no/low poisoning risk'
);

// M5 chain validation: out-of-order
const ooo1 = envelope()
  .sender('a').recipient('b').chain('ooo-chain', 5)
  .inform().engineering().verified('test', 0.9).payload('test').build();
const ooo2 = envelope()
  .sender('b').recipient('c').chain('ooo-chain', 3)
  .inform().engineering().verified('test', 0.9).payload('test').build();

const chainViolations = validateChain([ooo1, ooo2]);
assert(chainViolations.length > 0, 'M5: Out-of-order chain detected');

// --- Summary ---

console.log('\n═══════════════════════════════');
console.log(`  ${passed} passed, ${failed} failed`);
console.log('═══════════════════════════════\n');

if (failed > 0) process.exit(1);
