/* shad0w C API: load a compiled table and decide in about a microsecond. No dependencies beyond libc + libm.
 *
 *   cc -O2 app.c reflex.c -lm -o app
 *
 *   S0Reflex *m = s0_load("bundle/intent.s0");          // NULL if missing, truncated or not a shad0w table
 *   float probs[S0_MAX_CLASSES], radius;
 *   int k = s0_decide(m, text, (int)strlen(text), probs, &radius);
 *   // the answer is option k (options[] and "threshold" are in bundle/manifest.json, in this order);
 *   // it is certified when probs[k] >= threshold. Otherwise ask your LLM.
 *   s0_free(m);
 *
 * A loaded table is read-only: share it across threads; give each thread its own probs buffer.
 */
#ifndef SHAD0W_H
#define SHAD0W_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

#define S0_MAX_CLASSES 1024

typedef struct S0Reflex S0Reflex;

/* Load a .s0 table. Returns NULL on any error (missing file, bad header, wrong size, unsorted keys). */
S0Reflex *s0_load(const char *path);

/* Decide for UTF-8 text of `len` bytes. Returns the index of the most likely option, writes the probability of
 * every option to `probs` (s0_num_classes(m) floats) and the exact robustness radius to `radius`
 * (how many inserted words it takes to flip the answer; INFINITY when none can). */
int s0_decide(const S0Reflex *m, const char *text, int len, float *probs, float *radius);

/* Number of options. */
uint32_t s0_num_classes(const S0Reflex *m);

/* Mean nanoseconds per call over `reps` passes of `texts` (per-call times go to per_call_ns, may be NULL). */
double s0_bench(const S0Reflex *m, const char **texts, const int *lens, int n, int reps, double *per_call_ns);

void s0_free(S0Reflex *m);

#ifdef __cplusplus
}
#endif
#endif
