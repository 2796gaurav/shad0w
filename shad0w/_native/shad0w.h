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
 * (how many inserted words it takes to flip the answer; INFINITY when none can; NAN for a deep table, which has
 * no exact radius: treat it as "not robust" where robustness matters). */
int s0_decide(const S0Reflex *m, const char *text, int len, float *probs, float *radius);

/* 0 for a linear table (format 1), else the width of a deep table's hidden layer (format 2). */
uint32_t s0_hidden(const S0Reflex *m);

/* s0_decide plus counts[0] = items in the text, counts[1] = items that have a table row (1 - counts[1] / counts[0]
 * is the "unseen" share). `sw` is the bundle's static word table or NULL. */
typedef struct S0Static S0Static;
S0Static *s0_static_load(const char *path);  /* a <question>.s0w file of an older bundle trained with static= */
void s0_static_free(S0Static *s);
int s0_decide_static(const S0Reflex *m, const S0Static *sw, const char *text, int len, float *probs, float *radius);
int s0_decide_counts(const S0Reflex *m, const S0Static *sw, const char *text, int len, float *probs, float *radius,
                     uint32_t *counts);

/* Decide from item ids the caller computed (e.g. an image featurizer, see shad0w/vision.py): same scores, probabilities
 * and radius as s0_decide, without hashing text. Ids are reduced to 22 bits. */
int s0_decide_items(const S0Reflex *m, const uint32_t *items, int n_items, float *probs, float *radius);
int s0_decide_items_counts(const S0Reflex *m, const uint32_t *items, int n_items, float *probs, float *radius,
                           uint32_t *counts);

/* Images (shad0w/vision.py): pixels -> item ids with no neural network, then the same table.
 *
 *   S0ImageSpec spec = {32, 3, 256, 8, 2, 4, 4, 2, 0, 2, 1};  // bundle/manifest.json "meta" -> "image", in this order
 *   S0Image *f = s0_image_new(&spec);
 *   int k = s0_decide_image(m, f, rgb, width, height, 3, width * 3, probs, &radius);   // -1 on bad input
 *
 * Pixels are uint8, interleaved, 1 (grey), 3 (RGB) or 4 (RGBA) channels, rows `stride` bytes apart. */
typedef struct {
  int32_t size, channels, ferns, bits, radius, grid, levels, margin;
  uint32_t seed;
  int32_t version, dhash;  /* "ferns/2" -> version 2; dhash 1 adds 4 near-duplicate items */
} S0ImageSpec;
typedef struct S0Image S0Image;

S0Image *s0_image_new(const S0ImageSpec *spec);  /* NULL when the spec is out of range */
int s0_image_num_items(const S0Image *f);
int s0_image_items(const S0Image *f, const uint8_t *px, int w, int h, int c, int stride, uint32_t *out);
int s0_decide_image(const S0Reflex *m, const S0Image *f, const uint8_t *px, int w, int h, int c, int stride,
                    float *probs, float *radius);
double s0_bench_image(const S0Reflex *m, const S0Image *f, const uint8_t *const *px, int n, int w, int h, int c,
                      int stride, int reps);
void s0_image_free(S0Image *f);

/* Text featurizer of the bundle: manifest.json "meta" -> "text_kind" (absent = 1). Call right after s0_load when it
 * is 2 (script-aware words for Chinese and Japanese; identical on ASCII). s0_load sets 1. */
void s0_set_text_kind(S0Reflex *m, int kind);
int s0_text_kind(const S0Reflex *m);

/* Number of options. */
uint32_t s0_num_classes(const S0Reflex *m);

/* Mean nanoseconds per call over `reps` passes of `texts` (per-call times go to per_call_ns, may be NULL). */
double s0_bench(const S0Reflex *m, const char **texts, const int *lens, int n, int reps, double *per_call_ns);

void s0_free(S0Reflex *m);

#ifdef __cplusplus
}
#endif
#endif
