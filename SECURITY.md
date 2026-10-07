# Security

Please report vulnerabilities privately through GitHub's **Report a vulnerability** button (Security tab), not in a public issue.

Scope:
- The runtime performs no network access and executes no code from bundles. All three runtimes validate table headers and sizes, so a malformed `.s0` file is refused. Crashes or memory errors in `shad0w/_native/reflex.c` or `js/index.js` on crafted files are in scope.
- Bundles store hashed n-gram weights, not training text. Frequent short phrases can still be guessed by brute force, so treat bundles trained on sensitive data as sensitive.
- `shad0w serve` is a reference server without authentication. Keep it on localhost or behind your own front end.
