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
  decide(text: string, probabilities?: boolean): TableDecision;
}

export interface Answer {
  confidence: number;
  /** true: serve this answer. false: call your model instead. */
  certified: boolean;
  flag: string | null;
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
  decide(text: string, opts?: { questions?: string[]; probabilities?: boolean }): { answers: Record<string, Answer> };
}
export function items(text: string): number[];
export function words(text: string): Uint8Array[];
export const TABLE_VERSION: number;

export type Flag = "no_bundle" | "low_confidence" | "low_radius" | "empty_input" | "drift" | "uncalibrated" | "min_confidence" | "manual_threshold"
  | "canary" | "never_serve" | "shadow" | "off" | "options_changed" | "option_removed" | "new_options_served" | null;

/** One decision from a Shadow cascade. */
export interface Decision<A = string | boolean> {
  answer: A;
  /** "table": certified, answered locally in microseconds. "teacher": your LLM answered (and it was logged). */
  source: "table" | "teacher" | "fallback";
  confidence: number | null;
  certified: boolean;
  /** why the table deferred */
  flag: Flag;
  latencyUs: number;
  question: string;
  /** the certified confidence threshold of the bundle's question, or null without a bundle */
  threshold: number | null;
  /** per-option probabilities (table decisions from peek() only) */
  probabilities?: Record<string, number> | null;
  /** why this answer came from where it did, in plain words */
  why: string;
}

/** What the table thinks of a text, without calling your model. */
export interface Explanation {
  text: string;
  answer: string | boolean | null;
  confidence: number | null;
  threshold: number | null;
  certified: boolean;
  flag: Flag;
  /** the reason in plain words */
  why: string;
  /** the top options and their probabilities */
  top: [string, number][];
}
export const FLAG_WORDS: Record<string, string>;
/** The reason behind a flag, in plain words. */
export function explainFlag(flag: Flag): string;
export type Teacher = ((text: string) => Promise<string | boolean> | string | boolean) & { question?: string; keySource?: string };
/** An API key: a string, or a function called on every request (rotating or vault keys). Never logged or printed. */
export type ApiKey = string | (() => string | Promise<string>);
/** Process-wide defaults for decision() and the teachers. configure() with no argument clears them; the result masks the key. */
export function configure(opts?: { llm?: string; apiKey?: ApiKey; apiKeyEnv?: string; baseURL?: string }): Record<string, unknown>;
/** "sk-…3f9a": a key's prefix and last 4 characters. */
export function maskKey(key: string | null | undefined): string;
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
  /** the options you offer now; compared with the trained bundle to catch added / removed options */
  options?: string[] | Record<string, string | null> | Record<string, unknown>;
  /** options added since training: "defer" (default) sends every decision to the teacher until you retrain */
  onNewOption?: "defer" | "serve";
  /** answer when the table defers and there is no teacher (running without an LLM); never logged */
  fallback?: string | boolean | ((text: string) => string | boolean | Promise<string | boolean>);
  /** rename labels without retraining, e.g. { lost_card: "card_lost" } */
  rename?: Record<string, string>;
}
export class Shadow {
  constructor(bundle: Bundle | null, opts?: ShadowOptions);
  readonly question: string;
  readonly bundle: Bundle | null;
  readonly teacher: Teacher | null;
  decide(text: string, opts?: { teacher?: Teacher }): Promise<Decision>;
  /** The table's decision when it would be served (counted like any table decision), else null. Never calls the teacher. */
  peek(text: string, opts?: { teacher?: Teacher }): Promise<Decision | null>;
  /** Log an answer you obtained from your model yourself; counts as a teacher decision. */
  record(text: string, answer: string | boolean): Promise<Decision>;
  /** What the table thinks of `text`, without calling your model. */
  explain(text: string): Explanation;
  /** the option names of this question (from the bundle, else the teacher), or null */
  options(): string[] | null;
  threshold(): number | null;
  /** options added / removed since the bundle was trained (see ShadowOptions.options) */
  readonly optionsAdded: string[];
  readonly optionsRemoved: string[];
  /** Decide a batch in order: table answers first, only deferred texts go to the teacher, `concurrency` (default 8) at a time. */
  decideMany(texts: Iterable<string>, opts?: { concurrency?: number; teacher?: Teacher }): Promise<Decision[]>;
  /** question, options, state, teacher and where its key comes from (masked) */
  toString(): string;
  stats(): { table: number; teacher: number; fallback: number; audits: number; auditDisagreements: number; offload: number; auditDisagreement: number | null;
             optionsAdded: string[]; optionsRemoved: string[] };
}

export interface TeacherOptions {
  /** option names, {name: description}, or a shad0w schema entry */
  options: string[] | Record<string, string | null> | Record<string, unknown>;
  /** "provider/model", e.g. "openai/gpt-6-luna", "groq/llama-3.1-8b-instant", "ollama/llama3.1"; or a bare name with baseURL */
  model?: string;
  question?: string;
  baseURL?: string;
  /** the key: a string or a function called on every request. Default: apiKeyEnv, configure(), then the provider's variable */
  apiKey?: ApiKey;
  /** the NAME of the environment variable holding the key */
  apiKeyEnv?: string;
  fetch?: typeof fetch;
  system?: string;
  temperature?: number;
  maxTokens?: number;
  retries?: number;
  headers?: Record<string, string>;
  /** ask for JSON-schema structured output first (default true; falls back to plain text automatically) */
  structured?: boolean;
  /** milliseconds per request (default 30000); a timed-out request is retried like a network error */
  timeout?: number;
}
/** A teacher backed by any OpenAI-compatible chat endpoint. */
export function openaiTeacher(opts: TeacherOptions): Teacher & { options: string[]; complete(text: string): Promise<string> };

export interface DecisionTeacherOptions {
  options: string[] | Record<string, string | null> | Record<string, unknown>;
  question?: string;
  /** the model name sent to the server (System One: e.g. "kev-0.8b"; Decisions API: default "gpt-6-luna") */
  model?: string;
  baseURL?: string;
  apiKey?: ApiKey;
  apiKeyEnv?: string;
  fetch?: typeof fetch;
  headers?: Record<string, string>;
  timeout?: number;
  retries?: number;
}
/** A teacher that asks a System One server (TypeSafe wire format: Jev, Kev, Laya, Ollaya, llama.cpp) at baseURL/v1/systemone. */
export function systemoneTeacher(opts: DecisionTeacherOptions & { baseURL: string }): Teacher & { options: string[] };
/** A teacher that asks the OpenAI Decisions API (POST /v1/decisions) one choice or predicate question. */
export function decisionsTeacher(opts: DecisionTeacherOptions): Teacher & { options: string[] };

/**
 * decision("intent", {options, llm: "openai/gpt-6-luna", apiKey: process.env.OPENAI_API_KEY, bundle: "intent.bundle", log: "intent.log.jsonl"})
 * apiKey / apiKeyEnv / baseURL with a function llm throw a TypeError (the function holds its own key); unknown options throw too.
 * llm may also be "systemone/<model>" (with baseURL) or "openai-decisions/gpt-6-luna".
 * A bundle path that does not exist starts log-only; a corrupt or unsupported bundle throws.
 */
export type DecisionOptions = Partial<TeacherOptions> & ShadowOptions & {
  /** what this decision is called (default "decision"): its log field, bundle question and metrics label */
  name?: string;
  llm?: string | Teacher;
  bundle?: Bundle | string | null;
};
export function decision(name: string, opts: DecisionOptions): Promise<Shadow>;
export function decision(opts: DecisionOptions): Promise<Shadow>;

/** Vercel AI SDK language-model middleware: wrapLanguageModel({ model, middleware: shad0wMiddleware(intent) }). */
export interface MiddlewareOptions {
  /** must match the installed `ai` major: ai 5 → "v2", ai 6 → "v3", ai 7 → "v4" (default "v3") */
  specificationVersion?: "v2" | "v3" | "v4";
  /** option names to match in the model's replies (default: the Shadow's options) */
  options?: string[];
  /** shapes the reply text for a table answer (default: the bare option) */
  format?: (answer: string | boolean) => string;
}
export function shad0wMiddleware(shadow: Shadow, opts?: MiddlewareOptions): {
  specificationVersion: string;
  middlewareVersion: string;
  wrapGenerate(args: { doGenerate: () => PromiseLike<any>; params: any }): Promise<any>;
};

/** AI SDK decision model (spec v4) for experimental_decide({ model: decisionModel(intent), state, questions }). */
export interface DecisionModelLike {
  specificationVersion: "v4";
  provider: string;
  modelId: string;
  supportedQuestionTypes: readonly string[];
  doDecide(options: { state: unknown; questions: Record<string, any>; abortSignal?: AbortSignal; headers?: Record<string, string>; providerOptions?: unknown }): PromiseLike<any>;
}
export function decisionModel(shadows: Shadow | Record<string, Shadow>, opts?: {
  /** another decision model (e.g. openai.decisionModel("gpt-6-luna")) for questions the tables cannot certify; its answers are logged */
  fallback?: DecisionModelLike;
  name?: string;
  provider?: string;
}): DecisionModelLike;

export function matchOption(reply: unknown, options: string[], field?: string): string | null;
export const PROVIDERS: Record<string, [string, string | null]>;

declare const shad0w: {
  Table: typeof Table; Bundle: typeof Bundle; Shadow: typeof Shadow; openaiTeacher: typeof openaiTeacher;
  systemoneTeacher: typeof systemoneTeacher; decisionsTeacher: typeof decisionsTeacher; decision: typeof decision;
  shad0wMiddleware: typeof shad0wMiddleware; decisionModel: typeof decisionModel; configure: typeof configure;
  maskKey: typeof maskKey; explainFlag: typeof explainFlag; FLAG_WORDS: typeof FLAG_WORDS; matchOption: typeof matchOption;
  PROVIDERS: typeof PROVIDERS; items: typeof items; words: typeof words; TABLE_VERSION: typeof TABLE_VERSION;
};
export default shad0w;
