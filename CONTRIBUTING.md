# Contributing

Thanks for helping. Small, focused pull requests are the easiest to review.

## The most useful contribution
Run shad0w on one of your own decisions and [share the numbers](https://github.com/2796gaurav/shad0w/issues/new?template=results.yml), failures included. Include:
- the task and number of options;
- the teacher;
- α and the shadow log size;
- agreement and certified share;
- realised disagreement on fresh traffic.

## Setup
```bash
python -m venv .venv && . .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # optional: CPU-only torch is much smaller
pip install -e ".[dev]"        # also builds the C core in place
make test lint
```

## Ground rules
1. **Numbers need evidence.** A change that alters published behaviour or numbers needs a test or a result file that shows it.
2. **Keep the runtime lean.** At decision time, `shad0w/` imports only numpy. Compile-time dependencies stay behind lazy imports and the optional extras.
3. **Keep the runtimes identical.** Any change to hashing or scoring lands in all four places together, with the parity tests passing: `shad0w/features.py`, `shad0w/reflex.py`, `shad0w/_native/reflex.c` and `js/index.js`.
4. **Honest docs.** If a feature has a limit, document it in the same pull request.
