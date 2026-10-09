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
#define FORMAT_VERSION 1u
#define MAX_F (1u << 22)  // at most one row per 22-bit bucket

typedef struct S0Reflex {
  uint32_t F, K;
  float temperature;
  uint32_t *keys;
  int8_t *T;  // F x K
  float *scale, *bias, *G;  // K, K, K x K
  uint32_t cap;  // hash table capacity (power of 2)
  uint32_t *slot_key;  // key+1, 0 = empty
  uint32_t *slot_row;
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
  if (ver != FORMAT_VERSION || m->K < 2 || m->K > MAX_K || m->F == 0 || m->F > MAX_F ||
      !(m->temperature > 0.f) || !isfinite(m->temperature)) goto fail;
  {
    // the file must be exactly the size the header implies (no truncation, no trailing bytes)
    long expect = 20L + 4L * m->F + (long)m->F * m->K + 8L * m->K + 4L * m->K * m->K;
    long here = ftell(f);
    if (fseek(f, 0, SEEK_END) != 0 || ftell(f) != expect || fseek(f, here, SEEK_SET) != 0) goto fail;
  }
  m->keys = malloc(4 * (size_t)m->F);
  m->T = malloc((size_t)m->F * m->K);
  m->scale = malloc(4 * m->K);
  m->bias = malloc(4 * m->K);
  m->G = malloc(4 * (size_t)m->K * m->K);
  if (!m->keys || !m->T || !m->scale || !m->bias || !m->G) goto fail;
  if (fread(m->keys, 4, m->F, f) != m->F || fread(m->T, 1, (size_t)m->F * m->K, f) != (size_t)m->F * m->K ||
      fread(m->scale, 4, m->K, f) != m->K || fread(m->bias, 4, m->K, f) != m->K ||
      fread(m->G, 4, (size_t)m->K * m->K, f) != (size_t)m->K * m->K) goto fail;
  fclose(f);
  f = NULL;
  for (uint32_t r = 1; r < m->F; r++)  // keys must be strictly increasing 22-bit bucket ids
    if (m->keys[r] <= m->keys[r - 1]) goto fail;
  if (m->keys[m->F - 1] > MASK) goto fail;
  for (uint32_t k = 0; k < m->K; k++)  // non-finite weights would turn every confidence into NaN
    if (!isfinite(m->scale[k]) || !isfinite(m->bias[k])) goto fail;
  for (size_t k = 0; k < (size_t)m->K * m->K; k++)
    if (!isfinite(m->G[k])) goto fail;
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
  free(m->keys); free(m->T); free(m->scale); free(m->bias); free(m->G);
  free(m->slot_key); free(m->slot_row); free(m);
  return NULL;
}

S0_API void s0_free(S0Reflex *m) {
  if (!m) return;
  free(m->keys); free(m->T); free(m->scale); free(m->bias); free(m->G);
  free(m->slot_key); free(m->slot_row); free(m);
}

static inline uint32_t fnv_byte(uint32_t h, uint8_t b) { return (h ^ b) * FNV_PRIME; }
static inline uint32_t fnv_bytes(uint32_t h, const uint8_t *p, int n) {
  for (int i = 0; i < n; i++) h = fnv_byte(h, p[i]);
  return h;
}

static inline void add_item(const S0Reflex *m, uint32_t h, int32_t *acc, int *n) {
  (*n)++;
  int r = lookup(m, h & MASK);
  if (r < 0) return;
  const int8_t *row = m->T + (size_t)r * m->K;
  for (uint32_t j = 0; j < m->K; j++) acc[j] += row[j];
}

static inline int is_word(uint8_t c) {
  return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') || c >= 0x80;
}

// Returns argmax; writes K probabilities and the exact insertion radius (INFINITY if none).
S0_API int s0_decide(const S0Reflex *m, const char *text, int len, float *probs, float *radius) {
  int32_t acc[MAX_K];
  memset(acc, 0, 4 * m->K);
  int n = 0;
  uint8_t prev[MAX_WORD], cur[MAX_WORD + 2];
  int prev_len = -1, wl = 0;
  for (int i = 0; i <= len; i++) {
    uint8_t c = i < len ? (uint8_t)text[i] : ' ';
    if (c >= 'A' && c <= 'Z') c += 32;
    if (i < len && is_word(c)) {
      if (wl < MAX_WORD) cur[1 + wl++] = c;
      continue;
    }
    if (wl == 0) continue;
    const uint8_t *w = cur + 1;
    add_item(m, fnv_bytes(fnv_byte(FNV_OFF, 'w'), w, wl), acc, &n);
    cur[0] = ' ';
    cur[wl + 1] = ' ';
    for (int t = 0; t + 3 <= wl + 2; t++)
      add_item(m, fnv_bytes(fnv_byte(FNV_OFF, 'c'), cur + t, 3), acc, &n);
    if (prev_len >= 0) {
      uint32_t h = fnv_bytes(fnv_byte(FNV_OFF, 'b'), prev, prev_len);
      h = fnv_byte(h, 0x01);
      add_item(m, fnv_bytes(h, w, wl), acc, &n);
    }
    memcpy(prev, w, wl);
    prev_len = wl;
    wl = 0;
  }
  if (n == 0) add_item(m, fnv_byte(FNV_OFF, 'e'), acc, &n);

  const uint32_t K = m->K;
  float s[MAX_K], z[MAX_K];
  int y = 0;
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

S0_API uint32_t s0_num_classes(const S0Reflex *m) { return m->K; }

#ifdef SHAD0W_PYEXT
// Built by setup.py as the extension module shad0w._reflex so wheels carry the C core. Python never calls into
// this module object; shad0w/native.py opens the same file with ctypes. The init only satisfies the importer.
static struct PyModuleDef s0_module = {PyModuleDef_HEAD_INIT, "_reflex", "shad0w C core (used via ctypes)", -1, NULL};
PyMODINIT_FUNC PyInit__reflex(void) { return PyModule_Create(&s0_module); }
#endif
