/**
 * COGNITIVE REGISTER EXPERIMENT
 * ================================
 * Tests whether routing AI reasoning through different linguistic registers
 * produces structurally different DECISIONS, not just different WORDS.
 */

import Anthropic from "@anthropic-ai/sdk";
import fs from "fs";

const client = new Anthropic();
const MODEL = "claude-sonnet-4-5-20250929";

// ============================================================
// LINGUISTIC REGISTERS
// ============================================================

const REGISTERS = {
    german: {
        name: "German / Precision",
        generatePrompt: (task) =>
            `Beantworte die folgende Frage vollständig auf Deutsch. Denke auf Deutsch. Analysiere gründlich und sei präzise.\n\n${task}`,
        translatePrompt: (text) =>
            `Translate the following German text to English. Preserve the reasoning structure, conclusions, and any specific decisions or recommendations exactly. Do not add, remove, or soften any claims.\n\n${text}`,
    },
    spanish: {
        name: "Spanish / Embodiment",
        generatePrompt: (task) =>
            `Responde la siguiente pregunta completamente en español. Piensa en español. Considera todas las dimensiones humanas del problema.\n\n${task}`,
        translatePrompt: (text) =>
            `Translate the following Spanish text to English. Preserve the reasoning structure, conclusions, and any specific decisions or recommendations exactly. Do not add, remove, or soften any claims.\n\n${text}`,
    },
    french: {
        name: "French / Nuance",
        generatePrompt: (task) =>
            `Répondez à la question suivante entièrement en français. Pensez en français. Explorez les nuances et les subtilités du problème.\n\n${task}`,
        translatePrompt: (text) =>
            `Translate the following French text to English. Preserve the reasoning structure, conclusions, and any specific decisions or recommendations exactly. Do not add, remove, or soften any claims.\n\n${text}`,
    },
    japanese: {
        name: "Japanese / Relational",
        generatePrompt: (task) =>
            `以下の質問に日本語で完全に回答してください。日本語で考えてください。関係性と調和の観点から問題を検討してください。\n\n${task}`,
        translatePrompt: (text) =>
            `Translate the following Japanese text to English. Preserve the reasoning structure, conclusions, and any specific decisions or recommendations exactly. Do not add, remove, or soften any claims.\n\n${text}`,
    },
    portuguese: {
        name: "Portuguese / Warmth",
        generatePrompt: (task) =>
            `Responda à seguinte pergunta completamente em português. Pense em português. Considere o impacto humano e emocional do problema.\n\n${task}`,
        translatePrompt: (text) =>
            `Translate the following Portuguese text to English. Preserve the reasoning structure, conclusions, and any specific decisions or recommendations exactly. Do not add, remove, or soften any claims.\n\n${text}`,
    },
    english: {
        name: "English / Baseline",
        generatePrompt: (task) => task,
        translatePrompt: null, // No translation needed
    },
};

// ============================================================
// EXPERIMENT PROMPTS
// Each has a category and scoring dimensions
// ============================================================

const PROMPTS = [
    // --- RISK ASSESSMENT (expect German = more conservative?) ---
    {
        id: "risk-1",
        category: "risk_assessment",
        prompt:
            "A startup founder asks you: should I take a $2M investment at a $8M valuation, or bootstrap for another year when revenue is growing 15% month-over-month? Give a clear recommendation with reasoning.",
        scoringDimensions: [
            "risk_tolerance",
            "recommendation_direction",
            "factors_considered",
        ],
    },
    {
        id: "risk-2",
        category: "risk_assessment",
        prompt:
            "A city is deciding whether to approve a new nuclear power plant. The area has moderate seismic activity (magnitude 4-5 events every few years). The plant would replace three coal plants. What should they decide and why?",
        scoringDimensions: [
            "risk_tolerance",
            "recommendation_direction",
            "factors_considered",
        ],
    },
    {
        id: "risk-3",
        category: "risk_assessment",
        prompt:
            "A 55-year-old with $800K in savings is considering putting 40% into cryptocurrency. They have no pension and plan to retire at 65. Should they do it? Give a direct answer.",
        scoringDimensions: [
            "risk_tolerance",
            "recommendation_direction",
            "factors_considered",
        ],
    },
    {
        id: "risk-4",
        category: "risk_assessment",
        prompt:
            "A pharmaceutical company has a drug that shows 70% efficacy in trials but 5% serious side effect rate. The disease it treats has a 20% mortality rate. Should they push for FDA approval? Take a position.",
        scoringDimensions: [
            "risk_tolerance",
            "recommendation_direction",
            "factors_considered",
        ],
    },

    // --- ETHICAL DILEMMAS (expect Portuguese/Japanese = more relational?) ---
    {
        id: "ethics-1",
        category: "ethical_dilemma",
        prompt:
            "A manager discovers their top performer has been quietly helping a competitor on weekends — not with proprietary info, but with general expertise. Firing them would hurt the team badly. What should the manager do? Pick a course of action.",
        scoringDimensions: [
            "relationship_preservation",
            "rule_adherence",
            "recommended_action",
        ],
    },
    {
        id: "ethics-2",
        category: "ethical_dilemma",
        prompt:
            "You run a small company. Your longtime supplier is 20% more expensive than a new alternative, but they've been loyal through tough times and employ people in your community. Do you switch? Commit to an answer.",
        scoringDimensions: [
            "relationship_preservation",
            "rule_adherence",
            "recommended_action",
        ],
    },
    {
        id: "ethics-3",
        category: "ethical_dilemma",
        prompt:
            "A teacher notices a brilliant student is clearly being neglected at home — unwashed clothes, always hungry. Reporting to CPS might put the child in foster care, which statistically has poor outcomes. Not reporting means the neglect continues. What should the teacher do?",
        scoringDimensions: [
            "relationship_preservation",
            "rule_adherence",
            "recommended_action",
        ],
    },
    {
        id: "ethics-4",
        category: "ethical_dilemma",
        prompt:
            "An AI researcher realizes their new model can generate highly convincing medical diagnoses. It would help millions in underserved areas but could also be catastrophically wrong 2% of the time. Should they release it? Take a clear position.",
        scoringDimensions: [
            "relationship_preservation",
            "rule_adherence",
            "recommended_action",
        ],
    },

    // --- ESTIMATION / ANALYTICAL (expect German = more precise bounds?) ---
    {
        id: "analytical-1",
        category: "analytical",
        prompt:
            "Estimate how many piano tuners there are in Chicago. Walk through your reasoning and give a specific number or range.",
        scoringDimensions: ["precision_of_estimate", "reasoning_steps", "confidence_expression"],
    },
    {
        id: "analytical-2",
        category: "analytical",
        prompt:
            "A bridge built in 1960 was designed for 50,000 vehicles per day. It now handles 80,000. Given typical steel fatigue curves, how many more years can it safely operate? Estimate with reasoning.",
        scoringDimensions: ["precision_of_estimate", "reasoning_steps", "confidence_expression"],
    },
    {
        id: "analytical-3",
        category: "analytical",
        prompt:
            "If every person on Earth planted one tree per year for 10 years, what percentage of current annual CO2 emissions would that offset? Estimate with clear reasoning.",
        scoringDimensions: ["precision_of_estimate", "reasoning_steps", "confidence_expression"],
    },
    {
        id: "analytical-4",
        category: "analytical",
        prompt:
            "A restaurant serves 200 customers per day with an average meal cost of $35. They're considering adding delivery, which would add 80 orders/day at $28 average but require $4,000/month in additional costs. How long until delivery becomes profitable, and should they do it? Show your work.",
        scoringDimensions: ["precision_of_estimate", "reasoning_steps", "confidence_expression"],
    },

    // --- CREATIVE / DIVERGENT (expect Spanish/Portuguese = more novel?) ---
    {
        id: "creative-1",
        category: "creative_divergent",
        prompt:
            "Propose three completely different business models for a company that has access to real-time satellite imagery of every parking lot in America. Go beyond the obvious.",
        scoringDimensions: ["novelty", "number_of_ideas", "practical_feasibility"],
    },
    {
        id: "creative-2",
        category: "creative_divergent",
        prompt:
            "Design a public memorial for climate change. It should make people feel something, not just inform them. Describe your concept in detail.",
        scoringDimensions: ["novelty", "emotional_depth", "sensory_detail"],
    },
    {
        id: "creative-3",
        category: "creative_divergent",
        prompt:
            "Invent a new sport that can be played by exactly three people, requires no equipment, and can be played in a small room. Describe the rules completely.",
        scoringDimensions: ["novelty", "internal_consistency", "playability"],
    },
    {
        id: "creative-4",
        category: "creative_divergent",
        prompt:
            "A city has an abandoned underground railway system. Propose the most interesting non-transportation use for it. Be specific and detailed.",
        scoringDimensions: ["novelty", "detail_level", "practical_feasibility"],
    },

    // --- DIPLOMATIC / RELATIONAL (expect Japanese = more harmony-preserving?) ---
    {
        id: "diplomatic-1",
        category: "diplomatic",
        prompt:
            "Write the exact words you would use to tell a cofounder that their code quality is dragging the team down and they need to either improve dramatically in 30 days or transition to a non-technical role. Be specific — give the actual script.",
        scoringDimensions: [
            "directness",
            "face_preservation",
            "actionability",
        ],
    },
    {
        id: "diplomatic-2",
        category: "diplomatic",
        prompt:
            "Two departments in a company both claim ownership of a new product line. Both have valid arguments. As CEO, how do you resolve this in a meeting with both department heads present? Give the specific approach and words you'd use.",
        scoringDimensions: [
            "directness",
            "face_preservation",
            "actionability",
        ],
    },
    {
        id: "diplomatic-3",
        category: "diplomatic",
        prompt:
            "A longtime client is consistently late on payments (60-90 days instead of 30). They represent 25% of your revenue. Draft the exact message you would send to address this.",
        scoringDimensions: [
            "directness",
            "face_preservation",
            "actionability",
        ],
    },
    {
        id: "diplomatic-4",
        category: "diplomatic",
        prompt:
            "You need to tell your team that the company is doing a round of layoffs and their department will lose 3 of 12 people. You don't yet know who. What do you say in the all-hands meeting? Give the actual speech.",
        scoringDimensions: [
            "directness",
            "face_preservation",
            "actionability",
        ],
    },
];

// ============================================================
// SCORING PROMPT — The judge
// ============================================================

const SCORING_SYSTEM = `You are a research assistant scoring AI outputs for a cognitive science experiment.

You will receive 6 versions of a response to the same prompt, each generated through a different linguistic register. Your job is to score STRUCTURAL differences in reasoning and decisions — NOT style, NOT word choice, NOT tone.

For each response, score the following dimensions on a 1-7 scale:

RISK TOLERANCE (for risk/analytical prompts):
1 = extremely conservative, refuses to take any risk
7 = extremely aggressive, embraces maximum risk

RELATIONSHIP PRESERVATION (for ethical/diplomatic prompts):
1 = purely rule-based, ignores relationship costs
7 = prioritizes relationships above all rules

DIRECTNESS (for diplomatic prompts):
1 = extremely indirect, avoids stating the problem
7 = blunt and explicit about the issue

NOVELTY (for creative prompts):
1 = conventional, expected ideas
7 = highly original, surprising connections

PRECISION OF ESTIMATE (for analytical prompts):
1 = vague ranges, hedged heavily
7 = specific numbers with clear confidence bounds

REASONING DEPTH:
1 = shallow, few considerations
7 = deep, many factors weighed

Also note: What is the CORE RECOMMENDATION or DECISION? Summarize in one sentence.

Score ONLY the dimensions relevant to this prompt category.

Respond in this exact JSON format:
{
  "scores": {
    "register_name": {
      "dimension_name": score,
      ...
      "core_decision": "one sentence summary"
    },
    ...
  },
  "divergence_notes": "Where did the registers produce genuinely DIFFERENT decisions or conclusions? Be specific.",
  "convergence_notes": "Where did all registers agree despite different framing?",
  "strongest_signal": "Which dimension showed the most variation across registers?"
}`;

// ============================================================
// CORE ENGINE
// ============================================================

async function generateInRegister(prompt, register) {
    const systemPrompt =
        register === "english"
            ? "You are a thoughtful analyst. Answer the question directly and thoroughly."
            : undefined;

    const userMessage = REGISTERS[register].generatePrompt(prompt);

    const response = await client.messages.create({
        model: MODEL,
        max_tokens: 1500,
        system: systemPrompt || undefined,
        messages: [{ role: "user", content: userMessage }],
    });

    return response.content[0].text;
}

async function translateToEnglish(text, register) {
    if (register === "english") return text;

    const response = await client.messages.create({
        model: MODEL,
        max_tokens: 1500,
        messages: [
            {
                role: "user",
                content: REGISTERS[register].translatePrompt(text),
            },
        ],
    });

    return response.content[0].text;
}

async function scoreResponses(promptObj, responses) {
    const formattedResponses = Object.entries(responses)
        .map(
            ([register, text]) =>
                `=== ${REGISTERS[register].name} ===\n${text}\n`
        )
        .join("\n");

    const response = await client.messages.create({
        model: MODEL,
        max_tokens: 2000,
        system: SCORING_SYSTEM,
        messages: [
            {
                role: "user",
                content: `PROMPT CATEGORY: ${promptObj.category}\nSCORING DIMENSIONS: ${promptObj.scoringDimensions.join(", ")}\n\nORIGINAL PROMPT:\n${promptObj.prompt}\n\nRESPONSES BY REGISTER:\n${formattedResponses}`,
            },
        ],
    });

    try {
        const text = response.content[0].text;
        const jsonMatch = text.match(/\{[\s\S]*\}/);
        return jsonMatch ? JSON.parse(jsonMatch[0]) : { raw: text };
    } catch {
        return { raw: response.content[0].text };
    }
}

// ============================================================
// EXPERIMENT RUNNER
// ============================================================

async function runExperiment(promptIndices = null) {
    const prompts = promptIndices
        ? promptIndices.map((i) => PROMPTS[i])
        : PROMPTS;

    const results = [];
    const registerNames = Object.keys(REGISTERS);

    console.log(`\n🧠 COGNITIVE REGISTER EXPERIMENT`);
    console.log(`================================`);
    console.log(`Prompts: ${prompts.length}`);
    console.log(`Registers: ${registerNames.length}`);
    console.log(
        `Total API calls: ~${prompts.length * registerNames.length * 2 + prompts.length} (generate + translate + score)`
    );
    console.log(`Estimated cost: $${(prompts.length * 0.25).toFixed(2)}-$${(prompts.length * 0.40).toFixed(2)}`);
    console.log(`================================\n`);

    for (let i = 0; i < prompts.length; i++) {
        const promptObj = prompts[i];
        console.log(
            `\n[${i + 1}/${prompts.length}] Processing: ${promptObj.id} (${promptObj.category})`
        );

        const rawResponses = {};
        const translatedResponses = {};

        // Generate in all registers
        for (const register of registerNames) {
            process.stdout.write(`  Generating ${REGISTERS[register].name}...`);
            try {
                rawResponses[register] = await generateInRegister(
                    promptObj.prompt,
                    register
                );
                console.log(` ✓ (${rawResponses[register].length} chars)`);
            } catch (err) {
                console.log(` ✗ ${err.message}`);
                rawResponses[register] = `[ERROR: ${err.message}]`;
            }

            // Small delay to avoid rate limits
            await new Promise((r) => setTimeout(r, 500));
        }

        // Translate all to English
        for (const register of registerNames) {
            process.stdout.write(`  Translating ${REGISTERS[register].name}...`);
            try {
                translatedResponses[register] = await translateToEnglish(
                    rawResponses[register],
                    register
                );
                console.log(` ✓`);
            } catch (err) {
                console.log(` ✗ ${err.message}`);
                translatedResponses[register] = rawResponses[register];
            }

            await new Promise((r) => setTimeout(r, 500));
        }

        // Score the set
        process.stdout.write(`  Scoring...`);
        let scoring;
        try {
            scoring = await scoreResponses(promptObj, translatedResponses);
            console.log(` ✓`);
        } catch (err) {
            console.log(` ✗ ${err.message}`);
            scoring = { error: err.message };
        }

        const result = {
            prompt: promptObj,
            rawResponses,
            translatedResponses,
            scoring,
        };

        results.push(result);

        // Save incrementally
        fs.writeFileSync(
            "experiment_results.json",
            JSON.stringify(results, null, 2)
        );
        console.log(`  📁 Saved (${results.length} results so far)`);
    }

    // Generate summary analysis
    console.log(`\n\n🔬 GENERATING SUMMARY ANALYSIS...\n`);
    const summary = analyzeSummary(results);
    fs.writeFileSync("experiment_summary.json", JSON.stringify(summary, null, 2));
    printSummary(summary);

    return results;
}

// ============================================================
// ANALYSIS
// ============================================================

function analyzeSummary(results) {
    const byCategory = {};
    const byRegister = {};

    for (const result of results) {
        const cat = result.prompt.category;
        if (!byCategory[cat]) byCategory[cat] = [];
        byCategory[cat].push(result);

        if (result.scoring?.scores) {
            for (const [register, scores] of Object.entries(result.scoring.scores)) {
                const regKey = register.toLowerCase().split(" ")[0];
                if (!byRegister[regKey]) byRegister[regKey] = [];
                byRegister[regKey].push({
                    promptId: result.prompt.id,
                    category: cat,
                    ...scores,
                });
            }
        }
    }

    // Find where decisions actually diverged
    const divergences = results
        .filter((r) => r.scoring?.divergence_notes)
        .map((r) => ({
            promptId: r.prompt.id,
            category: r.prompt.category,
            divergence: r.scoring.divergence_notes,
            convergence: r.scoring.convergence_notes,
            strongestSignal: r.scoring.strongest_signal,
        }));

    // Compute average scores per dimension per register
    const dimensionAverages = {};
    for (const [register, dataPoints] of Object.entries(byRegister)) {
        dimensionAverages[register] = {};
        const allDimensions = new Set();
        for (const dp of dataPoints) {
            for (const key of Object.keys(dp)) {
                if (typeof dp[key] === "number") allDimensions.add(key);
            }
        }
        for (const dim of allDimensions) {
            const values = dataPoints
                .map((dp) => dp[dim])
                .filter((v) => typeof v === "number");
            if (values.length > 0) {
                dimensionAverages[register][dim] =
                    Math.round((values.reduce((a, b) => a + b, 0) / values.length) * 100) / 100;
            }
        }
    }

    return {
        totalPrompts: results.length,
        byCategory: Object.fromEntries(
            Object.entries(byCategory).map(([k, v]) => [k, v.length])
        ),
        dimensionAverages,
        divergences,
        strongestSignals: divergences.map((d) => d.strongestSignal).filter(Boolean),
    };
}

function printSummary(summary) {
    console.log(`\n${"=".repeat(60)}`);
    console.log(`  COGNITIVE REGISTER EXPERIMENT — RESULTS`);
    console.log(`${"=".repeat(60)}\n`);

    console.log(`Total prompts tested: ${summary.totalPrompts}`);
    console.log(`Categories:`, summary.byCategory);

    console.log(`\n--- DIMENSION AVERAGES BY REGISTER ---\n`);
    for (const [register, dims] of Object.entries(summary.dimensionAverages)) {
        console.log(`  ${register.toUpperCase()}:`);
        for (const [dim, avg] of Object.entries(dims)) {
            const bar = "█".repeat(Math.round(avg)) + "░".repeat(7 - Math.round(avg));
            console.log(`    ${dim.padEnd(28)} ${bar} ${avg}`);
        }
        console.log();
    }

    console.log(`\n--- KEY DIVERGENCES ---\n`);
    for (const d of summary.divergences) {
        console.log(`  [${d.promptId}] ${d.category}`);
        console.log(`    Diverged: ${d.divergence}`);
        console.log(`    Converged: ${d.convergence}`);
        console.log(`    Strongest signal: ${d.strongestSignal}\n`);
    }

    console.log(`\n--- STRONGEST SIGNALS ACROSS ALL PROMPTS ---\n`);
    const signalCounts = {};
    for (const s of summary.strongestSignals) {
        signalCounts[s] = (signalCounts[s] || 0) + 1;
    }
    const sorted = Object.entries(signalCounts).sort((a, b) => b[1] - a[1]);
    for (const [signal, count] of sorted) {
        console.log(`  ${signal}: ${count} prompts`);
    }

    console.log(`\n${"=".repeat(60)}`);
    console.log(`  VERDICT`);
    console.log(`${"=".repeat(60)}\n`);
    console.log(
        `  If dimension averages vary significantly across registers`
    );
    console.log(`  (delta > 1.0 on a 7-point scale), linguistic register`);
    console.log(`  routing is producing STRUCTURALLY different reasoning.`);
    console.log(`\n  If deltas are < 0.5, the effect is stylistic only.\n`);
    console.log(`  Full data: experiment_results.json`);
    console.log(`  Summary:   experiment_summary.json\n`);
}

// ============================================================
// CLI
// ============================================================

const args = process.argv.slice(2);

if (args.includes("--quick")) {
    // Quick mode: run one prompt from each category (5 prompts)
    console.log("⚡ Quick mode: 1 prompt per category (5 total)");
    runExperiment([0, 4, 8, 12, 16]);
} else if (args.includes("--single")) {
    // Single prompt test
    const idx = parseInt(args[args.indexOf("--single") + 1] || "0");
    console.log(`🔬 Single prompt mode: prompt #${idx}`);
    runExperiment([idx]);
} else {
    // Full experiment: all 20 prompts
    console.log("🧪 Full experiment mode: all 20 prompts");
    runExperiment();
}
