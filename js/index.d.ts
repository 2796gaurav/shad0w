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
