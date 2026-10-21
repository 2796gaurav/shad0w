// shad0w native request path: hash -> lookup -> add -> softmax -> exact radius.
// Must match shad0w/features.py (item hashing) and shad0w/reflex.py (table format).
// Built as the Python extension shad0w._reflex (loaded through ctypes) or as a plain shared library (`make`).
#ifdef SHAD0W_PYEXT
#define PY_SSIZE_T_CLEAN
#include <Python.h>  // first, as Python requires
#endif
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef _WIN32
#include <windows.h>
#define S0_API __declspec(dllexport)
#else
#define S0_API __attribute__((visibility("default")))
#endif

#define BITS 22
#define MASK ((1u << BITS) - 1u)
#define FNV_OFF 0x811C9DC5u
#define FNV_PRIME 0x01000193u
#define MAX_K 1024
#define MAX_WORD 256
#define FORMAT_VERSION 1u   // linear table: rows K wide, exact radius
#define FORMAT_DEEP 2u      // deep table: rows H wide, relu, dense H x K (no radius); see shad0w/reflex.py
#define MAX_F (1u << 22)  // at most one row per 22-bit bucket

typedef struct S0Reflex {
  uint32_t F, K;
  float temperature;
  uint32_t *keys;
  int8_t *T;  // F x W, W = K (linear) or H (deep)
  float *scale, *bias, *G;  // W, W, K x K (linear only)
  uint32_t H, W;  // hidden width (0 = linear) and row width
  float *W2, *b2;  // H x K, K (deep only)
  uint32_t cap;  // hash table capacity (power of 2)
  uint32_t *slot_key;  // key+1, 0 = empty
  uint32_t *slot_row;
  int text_kind;  // text featurizer: 1 (default, every older bundle) or 2 (script-aware), see shad0w/features.py
} S0Reflex;

static inline uint32_t mix(uint32_t h) {  // cheap scramble for table slots
  h ^= h >> 16; h *= 0x7feb352du; h ^= h >> 15; h *= 0x846ca68bu; h ^= h >> 16;
  return h;
}

static inline int lookup(const S0Reflex *m, uint32_t key) {
  uint32_t i = mix(key) & (m->cap - 1);
  for (;;) {
    uint32_t k = m->slot_key[i];
    if (k == 0) return -1;
    if (k == key + 1) return (int)m->slot_row[i];
    i = (i + 1) & (m->cap - 1);
  }
}

S0_API S0Reflex *s0_load(const char *path) {
  FILE *f = fopen(path, "rb");
  if (!f) return NULL;
  char magic[4];
  uint32_t ver;
  S0Reflex *m = calloc(1, sizeof(S0Reflex));
  if (!m) { fclose(f); return NULL; }
  if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "S0RX", 4) != 0) goto fail;
  if (fread(&ver, 4, 1, f) != 1 || fread(&m->F, 4, 1, f) != 1 || fread(&m->K, 4, 1, f) != 1 ||
      fread(&m->temperature, 4, 1, f) != 1) goto fail;
  // refuse unknown versions, out-of-range shapes and a non-positive temperature before allocating anything
  if ((ver != FORMAT_VERSION && ver != FORMAT_DEEP) || m->K < 2 || m->K > MAX_K || m->F == 0 || m->F > MAX_F ||
      !(m->temperature > 0.f) || !isfinite(m->temperature)) goto fail;
  m->H = 0;
  m->text_kind = 1;
  if (ver == FORMAT_DEEP) {
    if (fread(&m->H, 4, 1, f) != 1 || m->H < 1 || m->H > MAX_K) goto fail;
  }
  m->W = m->H ? m->H : m->K;
  {
    // the file must be exactly the size the header implies (no truncation, no trailing bytes)
    long head = m->H ? 24L : 20L;
    long tail = m->H ? 4L * m->H * m->K + 4L * m->K : 4L * m->K * m->K;
    long expect = head + 4L * m->F + (long)m->F * m->W + 8L * m->W + tail;
    long here = ftell(f);
    if (fseek(f, 0, SEEK_END) != 0 || ftell(f) != expect || fseek(f, here, SEEK_SET) != 0) goto fail;
  }
  m->keys = malloc(4 * (size_t)m->F);
  m->T = malloc((size_t)m->F * m->W);
  m->scale = malloc(4 * m->W);
  m->bias = malloc(4 * m->W);
  if (!m->keys || !m->T || !m->scale || !m->bias) goto fail;
  if (fread(m->keys, 4, m->F, f) != m->F || fread(m->T, 1, (size_t)m->F * m->W, f) != (size_t)m->F * m->W ||
      fread(m->scale, 4, m->W, f) != m->W || fread(m->bias, 4, m->W, f) != m->W) goto fail;
  if (m->H) {
    m->W2 = malloc(4 * (size_t)m->H * m->K);
    m->b2 = malloc(4 * m->K);
    if (!m->W2 || !m->b2) goto fail;
    if (fread(m->W2, 4, (size_t)m->H * m->K, f) != (size_t)m->H * m->K || fread(m->b2, 4, m->K, f) != m->K) goto fail;
  } else {
    m->G = malloc(4 * (size_t)m->K * m->K);
    if (!m->G) goto fail;
    if (fread(m->G, 4, (size_t)m->K * m->K, f) != (size_t)m->K * m->K) goto fail;
  }
  fclose(f);
  f = NULL;
  for (uint32_t r = 1; r < m->F; r++)  // keys must be strictly increasing 22-bit bucket ids
    if (m->keys[r] <= m->keys[r - 1]) goto fail;
  if (m->keys[m->F - 1] > MASK) goto fail;
  for (uint32_t k = 0; k < m->W; k++)  // non-finite weights would turn every confidence into NaN
    if (!isfinite(m->scale[k]) || !isfinite(m->bias[k])) goto fail;
  if (m->H) {
    for (size_t k = 0; k < (size_t)m->H * m->K; k++)
      if (!isfinite(m->W2[k])) goto fail;
    for (uint32_t k = 0; k < m->K; k++)
      if (!isfinite(m->b2[k])) goto fail;
  } else {
    for (size_t k = 0; k < (size_t)m->K * m->K; k++)
      if (!isfinite(m->G[k])) goto fail;
  }
  m->cap = 1;
  while (m->cap < 2 * m->F) m->cap <<= 1;
  m->slot_key = calloc(m->cap, 4);
  m->slot_row = calloc(m->cap, 4);
  if (!m->slot_key || !m->slot_row) goto fail;
  for (uint32_t r = 0; r < m->F; r++) {
    uint32_t i = mix(m->keys[r]) & (m->cap - 1);
    while (m->slot_key[i]) i = (i + 1) & (m->cap - 1);
    m->slot_key[i] = m->keys[r] + 1;
    m->slot_row[i] = r;
  }
  return m;
fail:
  if (f) fclose(f);
  free(m->keys); free(m->T); free(m->scale); free(m->bias); free(m->G); free(m->W2); free(m->b2);
  free(m->slot_key); free(m->slot_row); free(m);
  return NULL;
}

S0_API void s0_free(S0Reflex *m) {
  if (!m) return;
  free(m->keys); free(m->T); free(m->scale); free(m->bias); free(m->G); free(m->W2); free(m->b2);
  free(m->slot_key); free(m->slot_row); free(m);
}

// 0 for a linear table, else the width of the hidden layer (such a table has no insertion radius).
S0_API uint32_t s0_hidden(const S0Reflex *m) { return m->H; }

static inline uint32_t fnv_byte(uint32_t h, uint8_t b) { return (h ^ b) * FNV_PRIME; }
static inline uint32_t fnv_bytes(uint32_t h, const uint8_t *p, int n) {
  for (int i = 0; i < n; i++) h = fnv_byte(h, p[i]);
  return h;
}

static inline void add_item(const S0Reflex *m, uint32_t h, int32_t *acc, int *n) {
  (*n)++;
  int r = lookup(m, h & MASK);
  if (r < 0) return;
  n[1]++;  // n[0] counts every item, n[1] those with a row (callers pass int n[2])
  const int8_t *row = m->T + (size_t)r * m->W;
  for (uint32_t j = 0; j < m->W; j++) acc[j] += row[j];
}

static inline int is_word(uint8_t c) {
  return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') || c >= 0x80;
}

static int finish(const S0Reflex *m, const int32_t *acc, int n, float *probs, float *radius);

// ---- static word vectors (shad0w/static.py): text -> mean of int8 word vectors -> quantile-bucket items ----------
// A .s0w file: "S0SW", u32 version (1), u32 dims D, u32 buckets NB, u32 words W, f64 thr[D * (NB - 1)] (bucket edges in
// int8 units: dimension d is in bucket q when its int32 SUM over the text's known words exceeds q of thr[d] * n),
// u32 word hashes[W] (strictly increasing; the full 32-bit FNV-1a of 'w' + word, the hash of the word item), int8 V[W * D].
typedef struct S0Static {
  uint32_t D, NB, W;
  double *thr;
  uint32_t *hash;
  int8_t *V;
} S0Static;

S0_API S0Static *s0_static_load(const char *path) {
  FILE *f = fopen(path, "rb");
  if (!f) return NULL;
  S0Static *s = calloc(1, sizeof(S0Static));
  char magic[4];
  uint32_t ver;
  if (!s) { fclose(f); return NULL; }
  if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "S0SW", 4) != 0 || fread(&ver, 4, 1, f) != 1 || ver != 1 ||
      fread(&s->D, 4, 1, f) != 1 || fread(&s->NB, 4, 1, f) != 1 || fread(&s->W, 4, 1, f) != 1) goto fail;
  if (s->D < 1 || s->D > 4096 || s->NB < 2 || s->NB > 16 || s->W > (1u << 24)) goto fail;
  {
    long expect = 20L + 8L * s->D * (s->NB - 1) + 4L * s->W + (long)s->W * s->D;
    long here = ftell(f);
    if (fseek(f, 0, SEEK_END) != 0 || ftell(f) != expect || fseek(f, here, SEEK_SET) != 0) goto fail;
  }
  s->thr = malloc(8 * (size_t)s->D * (s->NB - 1));
  s->hash = malloc(4 * (size_t)s->W + 4);
  s->V = malloc((size_t)s->W * s->D + 1);
  if (!s->thr || !s->hash || !s->V) goto fail;
  if (fread(s->thr, 8, (size_t)s->D * (s->NB - 1), f) != (size_t)s->D * (s->NB - 1) ||
      fread(s->hash, 4, s->W, f) != s->W || fread(s->V, 1, (size_t)s->W * s->D, f) != (size_t)s->W * s->D) goto fail;
  fclose(f);
  for (uint32_t i = 1; i < s->W; i++)
    if (s->hash[i] <= s->hash[i - 1]) { f = NULL; goto fail; }
  for (size_t i = 0; i < (size_t)s->D * (s->NB - 1); i++)
    if (!isfinite(s->thr[i])) { f = NULL; goto fail; }
  return s;
fail:
  if (f) fclose(f);
  free(s->thr); free(s->hash); free(s->V); free(s);
  return NULL;
}

S0_API void s0_static_free(S0Static *s) {
  if (!s) return;
  free(s->thr); free(s->hash); free(s->V); free(s);
}

static inline int static_row(const S0Static *s, uint32_t h) {
  int lo = 0, hi = (int)s->W - 1;
  while (lo <= hi) {
    int mid = (lo + hi) >> 1;
    if (s->hash[mid] < h) lo = mid + 1;
    else if (s->hash[mid] > h) hi = mid - 1;
    else return mid;
  }
  return -1;
}

static int decide_text(const S0Reflex *m, const S0Static *sw, const char *text, int len, float *probs, float *radius,
                       uint32_t *counts);

// Returns argmax; writes K probabilities and the exact insertion radius (INFINITY if none; NAN for a deep table).
S0_API int s0_decide(const S0Reflex *m, const char *text, int len, float *probs, float *radius) {
  return decide_text(m, NULL, text, len, probs, radius, NULL);
}

// s0_decide plus the static word-vector items of `sw` (a table trained with static= needs them).
S0_API int s0_decide_static(const S0Reflex *m, const S0Static *sw, const char *text, int len, float *probs, float *radius) {
  return decide_text(m, sw, text, len, probs, radius, NULL);
}

// s0_decide_static (sw may be NULL) that also reports counts[0] = items in the input, counts[1] = items with a row:
// an input whose items are mostly unseen (junk, another language) is one the table has no evidence about.
S0_API int s0_decide_counts(const S0Reflex *m, const S0Static *sw, const char *text, int len, float *probs, float *radius,
                            uint32_t *counts) {
  return decide_text(m, sw, text, len, probs, radius, counts);
}

// Text featurizer kind (manifest meta "text_kind"; absent = 1). Call once after s0_load, before sharing the table.
// 2 = script-aware words: identical to 1 on ASCII; Han/kana/Han/kana characters are words of their own
// (no trigrams), and Latin-1/Greek/Cyrillic capitals are folded. Any other value selects 1.
S0_API void s0_set_text_kind(S0Reflex *m, int kind) {
  if (m) m->text_kind = kind == 2 ? 2 : 1;
}

S0_API int s0_text_kind(const S0Reflex *m) { return m->text_kind; }

typedef struct {  // decide_text's running state: accumulators, the previous word (for bigrams), static sums
  const S0Reflex *m;
  const S0Static *sw;
  int32_t *acc, *n, *vsum;
  int vn, prev_len;
  uint8_t prev[MAX_WORD];
} S0Words;

// One word: its 'w' item (and static vector), its 'c' trigrams over ' '+word+' ' unless `single`, the bigram with the
// previous word. `buf` holds the word at buf[1..wl] with one free byte on each side.
static void emit_word(S0Words *st, uint8_t *buf, int wl, int single) {
  const uint8_t *w = buf + 1;
  uint32_t wh = fnv_bytes(fnv_byte(FNV_OFF, 'w'), w, wl);
  add_item(st->m, wh, st->acc, st->n);
  if (st->sw) {
    int row = static_row(st->sw, wh);
    if (row >= 0) {
      const int8_t *v = st->sw->V + (size_t)row * st->sw->D;
      for (uint32_t d = 0; d < st->sw->D; d++) st->vsum[d] += v[d];
      st->vn++;
    }
  }
  if (!single) {
    buf[0] = ' ';
    buf[wl + 1] = ' ';
    for (int t = 0; t + 3 <= wl + 2; t++) add_item(st->m, fnv_bytes(fnv_byte(FNV_OFF, 'c'), buf + t, 3), st->acc, st->n);
  }
  if (st->prev_len >= 0) {
    uint32_t h = fnv_bytes(fnv_byte(FNV_OFF, 'b'), st->prev, st->prev_len);
    h = fnv_byte(h, 0x01);
    add_item(st->m, fnv_bytes(h, w, wl), st->acc, st->n);
  }
  memcpy(st->prev, w, wl);
  st->prev_len = wl;
}

static inline int is_char_script(uint32_t cp) {  // scripts written without spaces: one word per character (kind 2)
  return (cp >= 0x2E80 && cp <= 0x2FDF) || (cp >= 0x3040 && cp <= 0x30FF) || (cp >= 0x31F0 && cp <= 0x31FF) ||
         (cp >= 0x3400 && cp <= 0x4DBF) || (cp >= 0x4E00 && cp <= 0x9FFF) || (cp >= 0xF900 && cp <= 0xFAFF) ||
         (cp >= 0x20000 && cp <= 0x2FA1F);
}

static inline uint32_t fold_cp(uint32_t cp) {  // Latin-1 Supplement, Greek and Cyrillic capitals (kind 2)
  if ((cp >= 0xC0 && cp <= 0xDE && cp != 0xD7) || (cp >= 0x391 && cp <= 0x3A9 && cp != 0x3A2) || (cp >= 0x410 && cp <= 0x42F))
    return cp + 0x20;
  if (cp >= 0x400 && cp <= 0x40F) return cp + 0x50;
  return cp;
}

// The UTF-8 sequence at p (p[0] >= 0x80, `avail` bytes left): its length with *cp set, or 0 when invalid
// (lead byte C2..F4 followed by its continuation bytes 80..BF; the same test as shad0w/features.py utf8_at).
static inline int utf8_at(const uint8_t *p, int avail, uint32_t *cp) {
  uint8_t c = p[0];
  int L;
  uint32_t v;
  if (c >= 0xC2 && c <= 0xDF) { L = 2; v = c & 0x1F; }
  else if (c >= 0xE0 && c <= 0xEF) { L = 3; v = c & 0x0F; }
  else if (c >= 0xF0 && c <= 0xF4) { L = 4; v = c & 0x07; }
  else return 0;
  if (L > avail) return 0;
  for (int k = 1; k < L; k++) {
    if ((p[k] & 0xC0) != 0x80) return 0;
    v = (v << 6) | (p[k] & 0x3F);
  }
  *cp = v;
  return L;
}

static int decide_text(const S0Reflex *m, const S0Static *sw, const char *text, int len, float *probs, float *radius,
                       uint32_t *counts) {
  int32_t acc[MAX_K];
  memset(acc, 0, 4 * m->W);
  int nn[2] = {0, 0};
  int *n = nn;
  int32_t vsum[4096];
  if (sw) memset(vsum, 0, 4 * sw->D);
  S0Words st = {m, sw, acc, n, vsum, 0, -1, {0}};
  uint8_t cur[MAX_WORD + 2];
  int wl = 0;
  const uint8_t *b = (const uint8_t *)text;
  if (m->text_kind != 2) {
    for (int i = 0; i <= len; i++) {
      uint8_t c = i < len ? b[i] : ' ';
      if (c >= 'A' && c <= 'Z') c += 32;
      if (i < len && is_word(c)) {
        if (wl < MAX_WORD) cur[1 + wl++] = c;
        continue;
      }
      if (wl == 0) continue;
      emit_word(&st, cur, wl, 0);
      wl = 0;
    }
  } else {
    uint8_t one[8];
    int i = 0;
    while (i < len) {
      uint8_t c = b[i];
      if (c < 0x80) {
        if (c >= 'A' && c <= 'Z') c += 32;
        if (is_word(c)) { if (wl < MAX_WORD) cur[1 + wl++] = c; }
        else if (wl) { emit_word(&st, cur, wl, 0); wl = 0; }
        i++;
        continue;
      }
      uint32_t cp = 0;
      int L = utf8_at(b + i, len - i, &cp);
      if (!L) {  // not valid UTF-8: one word byte, as kind 1
        if (wl < MAX_WORD) cur[1 + wl++] = c;
        i++;
        continue;
      }
      if (is_char_script(cp)) {
        if (wl) { emit_word(&st, cur, wl, 0); wl = 0; }
        memcpy(one + 1, b + i, L);
        emit_word(&st, one, L, 1);
      } else {
        uint32_t f = fold_cp(cp);
        if (f != cp) {  // every folded code point is 2 bytes
          uint8_t e[2] = {(uint8_t)(0xC0 | (f >> 6)), (uint8_t)(0x80 | (f & 0x3F))};
          for (int k = 0; k < 2; k++) if (wl < MAX_WORD) cur[1 + wl++] = e[k];
        } else {
          for (int k = 0; k < L; k++) if (wl < MAX_WORD) cur[1 + wl++] = b[i + k];
        }
      }
      i += L;
    }
    if (wl) emit_word(&st, cur, wl, 0);
  }
  if (n[0] == 0) add_item(m, fnv_byte(FNV_OFF, 'e'), acc, n);
  if (sw) {
    const double vcount = st.vn > 0 ? (double)st.vn : 1.0;
    for (uint32_t d = 0; d < sw->D; d++) {
      uint32_t q = 0;
      for (uint32_t bb = 0; bb + 1 < sw->NB; bb++) q += (double)vsum[d] > sw->thr[(size_t)d * (sw->NB - 1) + bb] * vcount;
      add_item(m, mix(0xD0000000u ^ (d * sw->NB + q)), acc, n);
    }
  }
  if (counts) { counts[0] = (uint32_t)n[0]; counts[1] = (uint32_t)n[1]; }
  return finish(m, acc, n[0], probs, radius);
}

// Decide from item ids already computed by the caller (an image featurizer, or features.items()): the same
// table, scores, probabilities and radius as s0_decide, without the text hashing. Ids are reduced to 22 bits.
S0_API int s0_decide_items_counts(const S0Reflex *m, const uint32_t *items, int n_items, float *probs, float *radius,
                                  uint32_t *counts) {
  int32_t acc[MAX_K];
  memset(acc, 0, 4 * m->W);
  int n[2] = {0, 0};
  for (int i = 0; i < n_items; i++) add_item(m, items[i], acc, n);
  if (n[0] == 0) add_item(m, fnv_byte(FNV_OFF, 'e'), acc, n);
  if (counts) { counts[0] = (uint32_t)n[0]; counts[1] = (uint32_t)n[1]; }
  return finish(m, acc, n[0], probs, radius);
}

S0_API int s0_decide_items(const S0Reflex *m, const uint32_t *items, int n_items, float *probs, float *radius) {
  return s0_decide_items_counts(m, items, n_items, probs, radius, NULL);
}

static int finish(const S0Reflex *m, const int32_t *acc, int n, float *probs, float *radius) {
  const uint32_t K = m->K;
  float s[MAX_K], z[MAX_K];
  int y = 0;
  if (m->H) {  // deep table: hidden = relu(acc * scale / n + bias), logits = hidden @ W2 + b2 (float32, as Python)
    float h[MAX_K];
    for (uint32_t j = 0; j < m->H; j++) {
      float v = (float)acc[j] * m->scale[j] / (float)n + m->bias[j];
      h[j] = v > 0.f ? v : 0.f;
    }
    for (uint32_t k = 0; k < K; k++) z[k] = m->b2[k];
    for (uint32_t j = 0; j < m->H; j++) {
      const float hj = h[j];
      if (hj == 0.f) continue;
      const float *w = m->W2 + (size_t)j * K;
      for (uint32_t k = 0; k < K; k++) {  // product rounded, then added: no fused multiply-add (numpy parity; the
        const float t = hj * w[k];        // build also passes -ffp-contract=off)
        z[k] = z[k] + t;
      }
    }
    for (uint32_t k = 0; k < K; k++) {
      z[k] /= m->temperature;
      if (z[k] > z[y]) y = (int)k;
    }
    float sum = 0.f;
    for (uint32_t j = 0; j < K; j++) { probs[j] = expf(z[j] - z[y]); sum += probs[j]; }
    for (uint32_t j = 0; j < K; j++) probs[j] /= sum;
    if (radius) *radius = NAN;  // not additive in the items: no exact radius
    return y;
  }
  for (uint32_t j = 0; j < K; j++) {
    s[j] = (float)acc[j] * m->scale[j];
    z[j] = (s[j] / n + m->bias[j]) / m->temperature;
    if (z[j] > z[y]) y = (int)j;
  }
  float sum = 0.f;
  for (uint32_t j = 0; j < K; j++) { probs[j] = expf(z[j] - z[y]); sum += probs[j]; }
  for (uint32_t j = 0; j < K; j++) probs[j] /= sum;
  if (radius) {
    float best = INFINITY;
    for (uint32_t j = 0; j < K; j++) {
      if ((int)j == y) continue;
      float db = m->bias[j] - m->bias[y];
      float den = m->G[(size_t)y * K + j] + db;
      if (den <= 0.f) continue;
      float k = floorf((-(s[j] - s[y]) - db * n) / den) + 1.f;
      if (k < best) best = k;
    }
    *radius = best;
  }
  return y;
}

// ---- images: pixels -> item ids (must match shad0w/vision.py exactly) ----------------------------------------
// A thumbnail (size x size x channels, integer box means over at most 4x4 samples per cell), then `ferns` groups of
// `bits` local pixel comparisons (one item per fern: its code) and a coarse grid of quantised cell means.
#define S0_IMG_MAX_SIZE 64     // thumbnail side (the thumbnail lives on the stack)
#define S0_IMG_MAX_FERNS 4096
#define S0_IMG_MAX_ITEMS (S0_IMG_MAX_FERNS + 16 * 16 * 3 + 4)

// version 1: thumbnail cells average 4 x 4 samples (up to 7 x 7 on some sizes). version 2: at most 2 x 2 (several times
// cheaper on large inputs), plus optional dHash items (`dhash` = 1): a 64-bit
// difference hash of the thumbnail in four 16-bit bands, so exact and near-duplicate images share items.
typedef struct {
  int32_t size, channels, ferns, bits, radius, grid, levels, margin;
  uint32_t seed;
  int32_t version, dhash;
} S0ImageSpec;

typedef struct S0Image {
  S0ImageSpec s;
  int32_t *a, *b;  // ferns*bits thumbnail offsets
} S0Image;

static inline uint32_t s0_rand(uint32_t *st) {
  uint32_t z = (*st += 0x9E3779B9u);
  z = (z ^ (z >> 16)) * 0x85EBCA6Bu;
  z = (z ^ (z >> 13)) * 0xC2B2AE35u;
  return z ^ (z >> 16);
}

static inline int clampi(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }

S0_API S0Image *s0_image_new(const S0ImageSpec *spec) {
  const S0ImageSpec s = *spec;
  if ((s.version != 1 && s.version != 2) || (s.dhash != 0 && s.dhash != 1) || (s.dhash && (s.version < 2 || s.size < 9)))
    return NULL;
  if (s.size < 4 || s.size > S0_IMG_MAX_SIZE || (s.channels != 1 && s.channels != 3) || s.ferns < 1 ||
      s.ferns > S0_IMG_MAX_FERNS || s.bits < 1 || s.bits > 16 || s.radius < 1 || s.grid < 0 || s.grid > 16 || s.grid > s.size ||
      s.levels < 1 || s.levels > 256 || s.margin < 0)
    return NULL;
  S0Image *f = calloc(1, sizeof(S0Image));
  if (!f) return NULL;
  f->s = s;
  size_t n = (size_t)s.ferns * s.bits;
  f->a = malloc(4 * n);
  f->b = malloc(4 * n);
  if (!f->a || !f->b) { free(f->a); free(f->b); free(f); return NULL; }
  uint32_t st = s.seed * 0x9E3779B9u + 0x7F4A7C15u;
  int S = s.size, C = s.channels, r = s.radius;
  for (size_t i = 0; i < n; i++) {
    int y1 = (int)(s0_rand(&st) % (uint32_t)S), x1 = (int)(s0_rand(&st) % (uint32_t)S);
    int dy = (int)(s0_rand(&st) % (uint32_t)(2 * r + 1)) - r, dx = (int)(s0_rand(&st) % (uint32_t)(2 * r + 1)) - r;
    int c1 = (int)(s0_rand(&st) % (uint32_t)C), c2 = (int)(s0_rand(&st) % (uint32_t)C);
    int y2 = clampi(y1 + dy, 0, S - 1), x2 = clampi(x1 + dx, 0, S - 1);
    f->a[i] = (y1 * S + x1) * C + c1;
    f->b[i] = (y2 * S + x2) * C + c2;
  }
  return f;
}

S0_API void s0_image_free(S0Image *f) {
  if (!f) return;
  free(f->a); free(f->b); free(f);
}

S0_API int s0_image_num_items(const S0Image *f) {
  return f->s.ferns + f->s.grid * f->s.grid * f->s.channels + 4 * f->s.dhash;
}

// pixels: h rows of w pixels with c interleaved uint8 channels (1 gray, 3 RGB, 4 RGBA), rows `stride` bytes apart.
// Writes s0_image_num_items(f) ids to out; returns that count, or -1 on bad input.
S0_API int s0_image_items(const S0Image *f, const uint8_t *px, int w, int h, int c, int stride, uint32_t *out) {
  const S0ImageSpec *s = &f->s;
  if (!px || w < 1 || h < 1 || (c != 1 && c != 3 && c != 4) || stride < w * c) return -1;
  const int S = s->size, C = s->channels, SAMP = s->version >= 2 ? 2 : 4;
  uint8_t th[S0_IMG_MAX_SIZE * S0_IMG_MAX_SIZE * 3];
  // cell bounds once per call (start, step, sample count) for rows and columns; the per-cell work is then adds
  int ry0[S0_IMG_MAX_SIZE], rsy[S0_IMG_MAX_SIZE], rny[S0_IMG_MAX_SIZE];
  int cx0[S0_IMG_MAX_SIZE], csx[S0_IMG_MAX_SIZE], cnx[S0_IMG_MAX_SIZE];
  for (int i = 0; i < S; i++) {
    int a0 = (int)((int64_t)i * h / S), a1 = (int)((int64_t)(i + 1) * h / S);
    if (a1 <= a0) a1 = a0 + 1;
    int st = (a1 - a0) / SAMP; if (st < 1) st = 1;
    ry0[i] = a0; rsy[i] = st; rny[i] = (a1 - a0 + st - 1) / st;
    if (s->version >= 2 && rny[i] > SAMP) rny[i] = SAMP;  // ferns/2: at most 2 samples per axis
    int b0 = (int)((int64_t)i * w / S), b1 = (int)((int64_t)(i + 1) * w / S);
    if (b1 <= b0) b1 = b0 + 1;
    st = (b1 - b0) / SAMP; if (st < 1) st = 1;
    cx0[i] = b0; csx[i] = st; cnx[i] = (b1 - b0 + st - 1) / st;
    if (s->version >= 2 && cnx[i] > SAMP) cnx[i] = SAMP;
  }
  if (w == S && h == S && c == C) {  // already thumbnail-sized with the same channels: the thumbnail is the input
    for (int i = 0; i < S; i++) memcpy(th + (size_t)i * S * C, px + (size_t)i * stride, (size_t)S * C);
  } else
  for (int i = 0; i < S; i++) {
    for (int j = 0; j < S; j++) {
      uint8_t *o = th + (i * S + j) * C;
      const int ny = rny[i], nx = cnx[j];
      if (ny == 1 && nx == 1) {  // one sample: a copy (inputs no bigger than the thumbnail)
        const uint8_t *p = px + (size_t)ry0[i] * stride + (size_t)cx0[j] * c;
        if (C == 1) o[0] = c >= 3 ? (uint8_t)((p[0] + 2 * p[1] + p[2]) >> 2) : p[0];
        else for (int k = 0; k < 3; k++) o[k] = c >= 3 ? p[k] : p[0];
        continue;
      }
      uint32_t acc[3] = {0, 0, 0};
      for (int yy = 0, y = ry0[i]; yy < ny; yy++, y += rsy[i]) {
        const uint8_t *row = px + (size_t)y * stride;
        for (int xx = 0, x = cx0[j]; xx < nx; xx++, x += csx[j]) {
          const uint8_t *p = row + (size_t)x * c;
          if (C == 1) acc[0] += c >= 3 ? (uint32_t)((p[0] + 2 * p[1] + p[2]) >> 2) : p[0];
          else for (int k = 0; k < 3; k++) acc[k] += c >= 3 ? p[k] : p[0];
        }
      }
      const uint32_t cnt = (uint32_t)(ny * nx);
      for (int k = 0; k < C; k++) {
        uint32_t v = acc[k];
        switch (cnt) {  // constant divisors compile to multiplies; same results as v / cnt
          case 2: v /= 2; break;
          case 3: v /= 3; break;
          case 4: v /= 4; break;
          case 6: v /= 6; break;
          case 9: v /= 9; break;
          default: v /= cnt;
        }
        o[k] = (uint8_t)v;
      }
    }
  }
  int n = 0;
  const int B = s->bits;
  for (int fe = 0; fe < s->ferns; fe++) {
    uint32_t code = 0;
    const int32_t *a = f->a + (size_t)fe * B, *b = f->b + (size_t)fe * B;
    for (int k = 0; k < B; k++) code |= (uint32_t)(th[a[k]] > th[b[k]] + s->margin) << k;
    out[n++] = mix(0xF0000000u ^ (((uint32_t)fe << B) | code)) & MASK;
  }
  const int G = s->grid;
  for (int gi = 0; gi < G; gi++)
    for (int gj = 0; gj < G; gj++) {
      int r0 = gi * S / G, r1 = (gi + 1) * S / G, q0 = gj * S / G, q1 = (gj + 1) * S / G;
      for (int k = 0; k < C; k++) {
        uint32_t acc = 0, cnt = 0;
        for (int y = r0; y < r1; y++)
          for (int x = q0; x < q1; x++) { acc += th[(y * S + x) * C + k]; cnt++; }
        uint32_t q = (acc / cnt) * (uint32_t)s->levels >> 8;
        uint32_t j = (uint32_t)((gi * G + gj) * C + k);
        out[n++] = mix(0xE0000000u ^ (j * (uint32_t)s->levels + q)) & MASK;
      }
    }
  if (s->dhash) {  // 9 x 8 grey block means of the thumbnail; bit = left block brighter than its right neighbour
    uint32_t g[8][9];
    for (int r = 0; r < 8; r++)
      for (int c = 0; c < 9; c++) {
        int r0 = r * S / 8, r1 = (r + 1) * S / 8, c0 = c * S / 9, c1 = (c + 1) * S / 9;
        uint32_t acc = 0, cnt = 0;
        for (int y = r0; y < r1; y++)
          for (int x = c0; x < c1; x++) {
            const uint8_t *p = th + (y * S + x) * C;
            acc += C == 1 ? p[0] : (uint32_t)((p[0] + 2 * p[1] + p[2]) >> 2);
            cnt++;
          }
        g[r][c] = acc / cnt;
      }
    for (int band = 0; band < 4; band++) {
      uint32_t bits = 0;
      for (int k = 0; k < 16; k++) {
        int r = band * 2 + k / 8, c = k % 8;
        bits |= (uint32_t)(g[r][c] > g[r][c + 1]) << k;
      }
      out[n++] = mix(0xB0000000u ^ ((uint32_t)band << 16 | bits)) & MASK;
    }
  }
  return n;
}

// Image in, decision out: s0_image_items then s0_decide_items. Returns the option index, or -1 on bad input.
S0_API int s0_decide_image(const S0Reflex *m, const S0Image *f, const uint8_t *px, int w, int h, int c, int stride,
                           float *probs, float *radius) {
  uint32_t its[S0_IMG_MAX_ITEMS];
  int n = s0_image_items(f, px, w, h, c, stride, its);
  if (n < 0) return -1;
  return s0_decide_items(m, its, n, probs, radius);
}

static double now_ns(void) {
#ifdef _WIN32
  static LARGE_INTEGER freq;
  LARGE_INTEGER c;
  if (!freq.QuadPart) QueryPerformanceFrequency(&freq);
  QueryPerformanceCounter(&c);
  return (double)c.QuadPart * 1e9 / (double)freq.QuadPart;
#else
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec * 1e9 + t.tv_nsec;
#endif
}

// Benchmarks n texts x reps in-process; returns mean ns per decision; fills per-call ns.
S0_API double s0_bench(const S0Reflex *m, const char **texts, const int *lens, int n, int reps, double *per_call_ns) {
  float probs[MAX_K], r;
  double total = 0;
  for (int rep = 0; rep < reps; rep++) {
    for (int i = 0; i < n; i++) {
      double t0 = now_ns();
      s0_decide(m, texts[i], lens[i], probs, &r);
      double ns = now_ns() - t0;
      if (per_call_ns && rep == reps - 1) per_call_ns[i] = ns;
      total += ns;
    }
  }
  return total / ((double)n * reps);
}

// Mean ns per s0_decide_image over n images of the same shape, `reps` passes.
S0_API double s0_bench_image(const S0Reflex *m, const S0Image *f, const uint8_t *const *px, int n, int w, int h, int c,
                             int stride, int reps) {
  float probs[MAX_K], r;
  double t0 = now_ns();
  for (int rep = 0; rep < reps; rep++)
    for (int i = 0; i < n; i++) s0_decide_image(m, f, px[i], w, h, c, stride, probs, &r);
  return (now_ns() - t0) / ((double)n * reps);
}

S0_API uint32_t s0_num_classes(const S0Reflex *m) { return m->K; }

#ifdef SHAD0W_PYEXT
// Built by setup.py as the extension module shad0w._reflex so wheels carry the C core. Python never calls into
// this module object; shad0w/native.py opens the same file with ctypes. The init only satisfies the importer.
static struct PyModuleDef s0_module = {PyModuleDef_HEAD_INIT, "_reflex", "shad0w C core (used via ctypes)", -1, NULL};
PyMODINIT_FUNC PyInit__reflex(void) { return PyModule_Create(&s0_module); }
#endif
