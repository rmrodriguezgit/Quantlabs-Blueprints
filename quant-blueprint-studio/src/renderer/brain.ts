import { engineRequest } from "./engine";
import { catalog, type Graph, type Param } from "./types/blueprint";

export type BrainEngine = "auto" | "ollama" | "rules";
export interface BrainStatus {
  available: boolean;
  model: string | null;
  models: string[];
  reason: string | null;
}
export interface BrainPlan {
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
function catalogCards() {
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
