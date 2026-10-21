Measured on Apple M5 Pro, 2026-10-19. Decision models run zero-shot on the CPU only with 5 threads; shad0w runs on one thread and learns from each task's training split. Accuracy is against the true labels; certified = share of test rows served under the same 5% Learn-then-Test certificate.

| task | shad0w: accuracy / certified / p50 | most accurate decision model | fastest decision model | shad0w in front of the most accurate |
|---|---|---|---|---|
| ag_news | 91.2% / 66% / 10.1 µs | Laya: 93.8%, 124.4 ms | Julia-1: 82.8%, 50.1 ms | 94.2%, 2.9× faster |
| banking77 | 94.0% / 89% / 6.0 µs | Kev-0.8B: 83.3%, 2.6 s (150 rows) | Julia-1: 62.8%, 197.1 ms | 91.3%, 7.8× faster |
| clinc150 | 84.6% / 86% / 6.5 µs | Decider-0.8B: 83.3%, 5.6 s (150 rows) | Julia-1: 27.4%, 314.1 ms | 91.3%, 7.5× faster |
| prompt_injection | 98.8% / 100% / 2.4 µs | GLiNER2.5-Decide: 94.6%, 112.6 ms | Julia-1: 34.4%, 14.3 ms | 98.8%, 28,988× faster |
| sms_spam | 97.8% / 100% / 1.9 µs | Decider-0.8B: 95.3%, 3.6 s (150 rows) | Julia-1: 66.4%, 15.9 ms | 97.3%, 1,704,195× faster |
| sst2 | 83.2% / 62% / 3.0 µs | Laya: 91.6%, 49.2 ms | Julia-1: 76.8%, 15.5 ms | 92.4%, 2.5× faster |
