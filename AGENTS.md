# Working on tdigest-rs

Use the commands under Development and release in the root README. Run `make build` and `make test` on the final change. Each commit requires a staged changelog entry.

Read the 1.0 release notes before changing numerical behavior. Keep the original Python tests and benchmark byte for byte identical to their pinned sources. Rust owns compression and query arithmetic. Bindings select the calling convention, translate errors and preserve dtype and shape. Check Python, Polars, Java and the CLI when changing shared behavior.
