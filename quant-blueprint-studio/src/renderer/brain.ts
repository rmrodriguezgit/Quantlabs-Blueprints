import { engineRequest } from "./engine";
import { catalog, type Graph, type Param } from "./types/blueprint";
import type { Fix } from "./validation";

export type BrainEngine = "auto" | "ollama" | "rules";
export interface BrainStatus {
  available: boolean;
  model: string | null;
  models: string[];
  reason: string | null;
}
export interface BrainPlan {
  plan_id?: string | null;
  title: string;
  objective: string;
  graph: Graph;
  steps: { id: string; tool: string; label: string; why: string; params: Record<string, Param> }[];
  dropped: { step: string; tool: string; reason: string }[];
  notes: string[];
  missing: { name: string; purpose: string }[];
  check: { ok: boolean; errors: string[] };
  engine: "ollama" | "rules";
  model: string | null;
  fallback_reason: string | null;
  detected: { tickers?: string[]; years?: number; source?: string };
  retrieved: { type: string; id: string; title: string; score: number }[];
}
export interface Explanation {
  summary: string;
  findings: string[];
  cautions: string[];
  next_steps: string[];
  rules: { summary: string; findings: string[]; cautions: string[]; next_steps: string[] };
  engine: "ollama" | "rules";
  model: string | null;
  unverified: string[];
  fallback_reason: string | null;
}

/** The catalog the planner may use: only declared kinds, pins and parameters. */
export function catalogCards() {
  return Object.fromEntries(
    Object.entries(catalog).map(([kind, d]) => [
      kind,
      {
        label: d.label,
        subtitle: d.subtitle,
        category: d.category,
        input: d.input ?? null,
        output: d.output ?? null,
        fields: d.fields.map(({ key, label, value, options, min, max }) => ({ key, label, value, options, min, max })),
      },
    ]),
  );
}

export const brainStatus = () => engineRequest<BrainStatus>("/api/brain/status");
export const planBlueprint = (prompt: string, engine: BrainEngine) =>
  engineRequest<BrainPlan>("/api/brain/plan", { prompt, engine, catalog: catalogCards() });
export const explainResult = (report: unknown, engine: BrainEngine = "auto") =>
  engineRequest<Explanation>("/api/brain/explain", { report, engine });

export interface Ping {
  state: "ok" | "slow" | "no_model" | "offline";
  url: string;
  model: string | null;
  seconds: number | null;
  reply: string | null;
  error: string | null;
}
export type TutorSource = { id: string; kind: "tool" | "concept" | "example"; title: string };
export type TutorAction =
  | { type: "add_tool"; kind: string; label: string }
  | { type: "open_example"; id: string; label: string };
export interface TutorAnswer {
  question: string;
  answer: string;
  engine: "ollama" | "rules";
  model: string | null;
  fallback_reason: string | null;
  sources: TutorSource[];
  actions: TutorAction[];
  follow_up: string[];
}
export interface BrainMemory {
  history: { id: string; at: number; prompt: string; title: string | null; engine: string; model: string | null; steps: string[]; ok: boolean; rating?: number }[];
  requests: { id: string; name: string; purpose: string; count: number; status: "pending" | "planned" | "discarded"; prompts: string[] }[];
  liked: number;
  path: string;
}
export const brainPing = () => engineRequest<Ping>("/api/brain/ping");
export const askTutor = (question: string, nodeType?: string) =>
  engineRequest<TutorAnswer>("/api/brain/ask", { question, engine: "auto", node_type: nodeType ?? null, catalog: catalogCards() });
export const brainMemory = () => engineRequest<BrainMemory>("/api/brain/memory");
export const ratePlan = (planId: string, rating: -1 | 0 | 1) =>
  engineRequest<{ id: string; rating: number }>("/api/brain/feedback", { plan_id: planId, rating });
export const setRequestStatus = (id: string, status: BrainMemory["requests"][number]["status"]) =>
  engineRequest<unknown>("/api/brain/request-status", { id, status });

export interface CerebroFix {
  fixes: Fix[];
  explanation: string[];
  unfixable: string[];
  retry: boolean;
  engine: "rules" | "ollama";
  model: string | null;
  fallback_reason: string | null;
}
/** Ask Cerebro for checked graph edits that fix validation and/or run-time errors. */
export const cerebroFix = (graph: Graph, error: string | null, issues: { message: string; node?: string; severity: string }[]) =>
  engineRequest<CerebroFix>("/api/brain/fix", { graph, error, issues, catalog: catalogCards() });
