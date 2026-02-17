// ═══════════════════════════════════════════════
// BABEL ENVELOPE BUILDER
// Fluent API for creating valid envelopes
// ═══════════════════════════════════════════════

import {
  BabelEnvelope,
  Meta,
  Intent,
  Register,
  Confidence,
  Basis,
  Affect,
  Ground,
  GroundAuthority,
  Trajectory,
  TrajectoryDirection,
  ValidationResult,
} from './types';
import { validate } from './validate';

export class EnvelopeBuilder {
  private _meta: Partial<Meta> = { version: 'babel/0.2' };
  private _intent: Intent = 'INFORM';
  private _confidence: Confidence[] = [];
  private _register: Register = 'AGENT_INTERNAL';
  private _affect?: Affect;
  private _grounds: Ground[] = [];
  private _trajectory?: Trajectory;
  private _payload: string = '';

  // --- Meta ---

  sender(sender: string): this {
    this._meta.sender = sender;
    return this;
  }

  recipient(recipient: string): this {
    this._meta.recipient = recipient;
    return this;
  }

  broadcast(): this {
    this._meta.recipient = 'broadcast';
    return this;
  }

  chain(chainId: string, seq: number): this {
    this._meta.chain_id = chainId;
    this._meta.seq = seq;
    return this;
  }

  // --- Intent ---

  intent(intent: Intent): this {
    this._intent = intent;
    return this;
  }

  inform(): this { return this.intent('INFORM'); }
  requestAction(): this { return this.intent('REQUEST_ACTION'); }
  escalate(): this { return this.intent('ESCALATE'); }
  flagRisk(): this { return this.intent('FLAG_RISK'); }
  speculate(): this { return this.intent('SPECULATE'); }
  persuade(): this { return this.intent('PERSUADE'); }
  delegate(): this { return this.intent('DELEGATE'); }
  synthesize(): this { return this.intent('SYNTHESIZE'); }

  // --- Confidence ---

  assert(assertion: string, score: number, basis?: Basis): this {
    this._confidence.push({ assertion, score, basis });
    return this;
  }

  verified(assertion: string, score: number): this {
    return this.assert(assertion, score, 'VERIFIED_DATA');
  }

  derived(assertion: string, score: number): this {
    return this.assert(assertion, score, 'DERIVED');
  }

  reported(assertion: string, score: number): this {
    return this.assert(assertion, score, 'REPORTED');
  }

  patternMatch(assertion: string, score: number): this {
    return this.assert(assertion, score, 'PATTERN_MATCH');
  }

  speculation(assertion: string, score: number): this {
    return this.assert(assertion, score, 'SPECULATION');
  }

  // --- Register ---

  register(register: Register): this {
    this._register = register;
    return this;
  }

  boardFacing(): this { return this.register('BOARD_FACING'); }
  engineering(): this { return this.register('ENGINEERING'); }
  customerExternal(): this { return this.register('CUSTOMER_EXTERNAL'); }
  regulatory(): this { return this.register('REGULATORY'); }
  internalMemo(): this { return this.register('INTERNAL_MEMO'); }
  agentInternal(): this { return this.register('AGENT_INTERNAL'); }

  // --- Affect ---

  affect(expansion: number, activation: number, certainty: number): this {
    this._affect = { expansion, activation, certainty };
    return this;
  }

  // --- Grounds ---

  ground(constraint: string, authority: GroundAuthority, override: boolean = false): this {
    // Auto-enforce M3: REGULATORY grounds are never overridable
    if (authority === 'REGULATORY') {
      override = false;
    }
    this._grounds.push({ constraint, authority, override });
    return this;
  }

  regulatoryGround(constraint: string): this {
    return this.ground(constraint, 'REGULATORY', false);
  }

  policyGround(constraint: string, override: boolean = false): this {
    return this.ground(constraint, 'POLICY', override);
  }

  // --- Trajectory ---

  withTrajectory(
    pattern: string,
    direction: TrajectoryDirection,
    opts?: { duration?: string; prior_handoffs?: number }
  ): this {
    this._trajectory = {
      pattern,
      direction,
      duration: opts?.duration,
      prior_handoffs: opts?.prior_handoffs,
    };
    return this;
  }

  // --- Payload ---

  payload(payload: string): this {
    this._payload = payload;
    return this;
  }

  // --- Build ---

  build(): BabelEnvelope {
    const now = new Date().toISOString();

    return {
      meta: {
        version: this._meta.version || 'babel/0.2',
        timestamp: this._meta.timestamp || now,
        sender: this._meta.sender || 'unknown',
        recipient: this._meta.recipient || 'broadcast',
        chain_id: this._meta.chain_id || generateUUID(),
        seq: this._meta.seq ?? 0,
      },
      intent: this._intent,
      confidence: this._confidence,
      register: this._register,
      affect: this._affect,
      grounds: this._grounds.length > 0 ? this._grounds : undefined,
      trajectory: this._trajectory,
      payload: this._payload,
    };
  }

  /**
   * Build and validate in one step.
   * Returns the validation result with the envelope attached.
   */
  buildAndValidate(): ValidationResult {
    const envelope = this.build();
    return validate(envelope);
  }
}

// --- Factory ---

export function envelope(): EnvelopeBuilder {
  return new EnvelopeBuilder();
}

// --- UUID helper ---

function generateUUID(): string {
  // Simple UUID v4 generator (no crypto dependency)
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, (c) => {
    const r = (Math.random() * 16) | 0;
    const v = c === 'x' ? r : (r & 0x3) | 0x8;
    return v.toString(16);
  });
}
