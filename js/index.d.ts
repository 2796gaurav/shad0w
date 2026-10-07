/** One compiled question. */
export interface TableDecision {
  index: number;
  choice: string | number;
  confidence: number;
  probabilities: number[];
}
export class Table {
  constructor(buffer: ArrayBuffer | ArrayBufferView, labels?: string[]);
  readonly F: number;
  readonly K: number;
  decide(text: string): TableDecision;
}

export interface Answer {
  confidence: number;
  /** true: serve this answer. false: call your model instead. */
  certified: boolean;
  flag: "low_confidence" | "uncalibrated" | null;
  /** choice questions */
  choice?: string;
  probabilities?: Record<string, number>;
  /** yes/no questions */
  answer?: boolean;
  probability?: number;
}
export interface Manifest {
  format: number;
  meta: Record<string, unknown>;
  questions: Record<string, { type: "choice" | "yesno"; options: string[]; threshold: number | null; alpha: number; calibrated: boolean }>;
}
export class Bundle {
  constructor(manifest: Manifest, tables: Record<string, ArrayBuffer | ArrayBufferView>);
  /** Node: a directory path. Browsers and workers: a URL. */
  static load(base: string): Promise<Bundle>;
  readonly manifest: Manifest;
  decide(text: string, opts?: { questions?: string[] }): { answers: Record<string, Answer> };
}
export function items(text: string): number[];
export function words(text: string): Uint8Array[];
export const TABLE_VERSION: number;

/** One decision from a Shadow cascade. */
export interface Decision<A = string | boolean> {
  answer: A;
  /** "table": certified, answered locally in microseconds. "teacher": your LLM answered (and it was logged). */
  source: "table" | "teacher";
  confidence: number | null;
  certified: boolean;
  /** why the table deferred */
  flag: "no_bundle" | "low_confidence" | "uncalibrated" | null;
  latencyUs: number;
}
export type Teacher = ((text: string) => Promise<string | boolean> | string | boolean) & { question?: string };
export type LogSink = ((row: Record<string, unknown>) => unknown) | { write(line: string): unknown } | string;

export interface ShadowOptions {
  teacher?: Teacher;
  question?: string;
  /** share of certified answers spot-checked against the teacher in the background (default 0.01) */
  auditRate?: number;
  onDecision?: (d: Decision, text: string) => void;
  /** receives every teacher answer as {text, <question>: answer, source, ts}: a function, a stream, or (Node) a file path */
  log?: LogSink;
  random?: () => number;
}
export class Shadow {
  constructor(bundle: Bundle | null, opts?: ShadowOptions);
  readonly question: string;
  decide(text: string, opts?: { teacher?: Teacher }): Promise<Decision>;
  stats(): { table: number; teacher: number; audits: number; auditDisagreements: number; offload: number; auditDisagreement: number | null };
}

export interface TeacherOptions {
  /** option names, {name: description}, or a shad0w schema entry */
  options: string[] | Record<string, string | null> | Record<string, unknown>;
  /** "provider/model", e.g. "openai/gpt-4o-mini", "groq/llama-3.1-8b-instant", "ollama/llama3.1"; or a bare name with baseURL */
  model?: string;
  question?: string;
  baseURL?: string;
  apiKey?: string;
  fetch?: typeof fetch;
  system?: string;
  temperature?: number;
  maxTokens?: number;
  retries?: number;
  headers?: Record<string, string>;
  /** ask for JSON-schema structured output first (default true; falls back to plain text automatically) */
  structured?: boolean;
}
/** A teacher backed by any OpenAI-compatible chat endpoint. */
export function openaiTeacher(opts: TeacherOptions): Teacher & { options: string[]; complete(text: string): Promise<string> };

/** decision("intent", {options, llm: "openai/gpt-4o-mini", bundle: "intent.bundle", log: "intent.log.jsonl"}) */
export function decision(name: string, opts: Partial<TeacherOptions> & ShadowOptions & {
  llm?: string | Teacher;
  bundle?: Bundle | string | null;
}): Promise<Shadow>;

export function matchOption(reply: unknown, options: string[], field?: string): string | null;
export const PROVIDERS: Record<string, [string, string | null]>;
