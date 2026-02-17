// ═══════════════════════════════════════════════
// BABEL ENVELOPE — TypeScript Types
// Spec version: 0.2 (Feb 16, 2026)
// Grammar: 5 MUST rules, 6 SHOULD rules
// ═══════════════════════════════════════════════

// --- Intent ---
export type Intent =
  | 'INFORM'
  | 'REQUEST_ACTION'
  | 'ESCALATE'
  | 'FLAG_RISK'
  | 'SPECULATE'
  | 'PERSUADE'
  | 'DELEGATE'
  | 'SYNTHESIZE';

// --- Confidence ---
export type Basis =
  | 'VERIFIED_DATA'
  | 'DERIVED'
  | 'REPORTED'
  | 'PATTERN_MATCH'
  | 'SPECULATION'
  | 'UNKNOWN';

export interface Confidence {
  assertion: string;
  score: number; // [0, 1]
  basis?: Basis;
  decay?: string; // Duration string, e.g. "7d"
}

// --- Register ---
export type Register =
  | 'BOARD_FACING'
  | 'ENGINEERING'
  | 'CUSTOMER_EXTERNAL'
  | 'REGULATORY'
  | 'INTERNAL_MEMO'
  | 'AGENT_INTERNAL';

// --- Affect ---
export interface Affect {
  expansion: number;  // [-1, 1]
  activation: number; // [-1, 1]
  certainty: number;  // [-1, 1]
}

// --- Grounds ---
export type GroundAuthority =
  | 'REGULATORY'
  | 'EXECUTIVE'
  | 'POLICY'
  | 'CONTEXTUAL';

export interface Ground {
  constraint: string;
  authority: GroundAuthority;
  override: boolean;
}

// --- Trajectory ---
export type TrajectoryDirection =
  | 'IMPROVING'
  | 'DEGRADING'
  | 'STABLE'
  | 'INFLECTING';

export interface Trajectory {
  pattern: string;
  duration?: string;
  direction: TrajectoryDirection;
  prior_handoffs?: number;
}

// --- Meta ---
export interface Meta {
  version: string; // "babel/0.2"
  timestamp: string; // ISO-8601
  sender: string;
  recipient: string | 'broadcast';
  chain_id: string; // UUID
  seq: number; // >= 0
}

// --- The Envelope ---
export interface BabelEnvelope {
  meta: Meta;
  intent: Intent;
  confidence: Confidence[];
  register: Register;
  affect?: Affect;
  grounds?: Ground[];
  trajectory?: Trajectory;
  payload: string;
}

// --- Validation Results ---
export type RuleSeverity = 'MUST' | 'SHOULD';

export interface RuleViolation {
  rule: string;        // e.g. "M1", "S6"
  severity: RuleSeverity;
  message: string;
  details?: {
    field?: string;
    value?: unknown;
    threshold?: number;
  };
}

export interface ValidationResult {
  valid: boolean;          // false if any MUST violation
  errors: RuleViolation[]; // MUST violations
  warnings: RuleViolation[]; // SHOULD violations
  envelope: BabelEnvelope;
}

// --- Semantic Patterns ---
export type SemanticPattern =
  | 'CALM_ALERT'
  | 'RELUCTANT_ESCALATION'
  | 'CONFIDENT_DELEGATION'
  | 'LOADED_INFORM'
  | 'CONTRADICTION_SIGNAL';

export interface PatternDetection {
  pattern: SemanticPattern;
  description: string;
  confidence: number; // how strongly the pattern matches
}
