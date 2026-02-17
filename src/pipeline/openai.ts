// ═══════════════════════════════════════════════
// BABEL — OpenAI Parallel Generator
// ═══════════════════════════════════════════════

import {
  Clause,
  LanguageScore,
  BabelLanguage,
  ParallelGenerator,
  LANGUAGE_LABELS,
  LANGUAGE_REGISTERS,
} from './types';
import { perplexityFromLogprobs } from './mapper';

export interface OpenAIGeneratorConfig {
  apiKey?: string;
  model?: string;
  maxTokens?: number;
  temperature?: number;
  baseUrl?: string;
  registerDirected?: boolean;
}

interface OpenAIChatResponse {
  choices: Array<{
    message: { content: string };
    logprobs?: {
      content: Array<{
        token: string;
        logprob: number;
      }>;
    };
  }>;
  model: string;
  usage: { prompt_tokens: number; completion_tokens: number; total_tokens: number };
}

export class OpenAIParallelGenerator implements ParallelGenerator {
  private apiKey: string;
  private model: string;
  private maxTokens: number;
  private temperature: number;
  private baseUrl: string;
  private registerDirected: boolean;

  constructor(config: OpenAIGeneratorConfig = {}) {
    this.apiKey = config.apiKey || process.env.OPENAI_API_KEY || '';
    if (!this.apiKey) {
      throw new Error(
        'OpenAI API key required. Set OPENAI_API_KEY env var or pass apiKey in config.'
      );
    }
    this.model = config.model || 'gpt-4o-mini';
    this.maxTokens = config.maxTokens || 150;
    this.temperature = config.temperature || 0.7;
    this.baseUrl = config.baseUrl || 'https://api.openai.com/v1';
    this.registerDirected = config.registerDirected ?? true;
  }

  async generate(
    clause: Clause,
    languages: BabelLanguage[],
    config?: { registerDirected?: boolean; firstTokenOnly?: boolean; tokenDepth?: number }
  ): Promise<LanguageScore[]> {
    const useRegisterDirected = config?.registerDirected ?? this.registerDirected;
    const firstTokenOnly = config?.firstTokenOnly ?? false;
    const tokenDepth = config?.tokenDepth ?? 3;

    if (firstTokenOnly) {
      return this.generateFirstTokenSelection(clause, languages, useRegisterDirected, tokenDepth);
    }

    const promises = languages.map(lang =>
      this.generateSingle(clause, lang, useRegisterDirected, this.maxTokens)
    );

    return Promise.all(promises);
  }

  private async generateFirstTokenSelection(
    clause: Clause,
    languages: BabelLanguage[],
    registerDirected: boolean,
    tokenDepth: number
  ): Promise<LanguageScore[]> {
    const probes = await Promise.all(
      languages.map(lang =>
        this.generateSingle(clause, lang, registerDirected, tokenDepth)
      )
    );

    const sorted = [...probes].sort((a, b) => a.perplexity - b.perplexity);
    const winner = sorted[0];

    const fullWinner = await this.generateSingle(
      clause,
      winner.language,
      registerDirected,
      this.maxTokens
    );

    return probes.map(probe =>
      probe.language === winner.language ? fullWinner : probe
    );
  }

  private async generateSingle(
    clause: Clause,
    language: BabelLanguage,
    registerDirected: boolean,
    maxTokens: number
  ): Promise<LanguageScore> {
    const prompt = registerDirected
      ? this.buildRegisterDirectedPrompt(clause, language)
      : this.buildSimplePrompt(clause, language);

    try {
      const response = await this.callOpenAI(prompt, maxTokens);
      const choice = response.choices[0];

      if (!choice?.logprobs?.content) {
        return {
          language,
          perplexity: 50,
          logprob_mean: -4,
          token_count: 0,
          generation: choice?.message?.content,
        };
      }

      const logprobs = choice.logprobs.content.map(t => t.logprob);
      const perplexity = perplexityFromLogprobs(logprobs);
      const logprobMean = logprobs.reduce((a, b) => a + b, 0) / logprobs.length;

      return {
        language,
        perplexity,
        logprob_mean: logprobMean,
        token_count: logprobs.length,
        generation: choice.message.content,
      };
    } catch (error) {
      return {
        language,
        perplexity: 1000,
        logprob_mean: -10,
        token_count: 0,
      };
    }
  }

  private buildRegisterDirectedPrompt(clause: Clause, language: BabelLanguage): string {
    const langName = LANGUAGE_LABELS[language];
    const register = LANGUAGE_REGISTERS[language];

    return [
      `Express the following thought natively in ${langName}.`,
      `Optimize for ${register}.`,
      `Do not translate — generate fresh in ${langName} as a native speaker would express this concept.`,
      `Output only the ${langName} text, nothing else.`,
      ``,
      `Thought: ${clause.text}`,
    ].join('\n');
  }

  private buildSimplePrompt(clause: Clause, language: BabelLanguage): string {
    const langName = LANGUAGE_LABELS[language];

    return [
      `Express the following thought natively in ${langName}.`,
      `Do not translate — generate as a native ${langName} speaker would express this concept.`,
      `Output only the ${langName} text, nothing else.`,
      ``,
      `Thought: ${clause.text}`,
    ].join('\n');
  }

  private async callOpenAI(prompt: string, maxTokens: number): Promise<OpenAIChatResponse> {
    const response = await fetch(`${this.baseUrl}/chat/completions`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${this.apiKey}`,
      },
      body: JSON.stringify({
        model: this.model,
        messages: [{ role: 'user', content: prompt }],
        max_tokens: maxTokens,
        temperature: this.temperature,
        logprobs: true,
        top_logprobs: 5,
      }),
    });

    if (!response.ok) {
      const error = await response.text();
      throw new Error(`OpenAI API error (${response.status}): ${error}`);
    }

    return response.json() as Promise<OpenAIChatResponse>;
  }
}
