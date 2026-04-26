# Upstream tdigest-rs Compatibility Snapshot

Source repository: `https://github.com/ingolfured/tdigest-rs.git`

Upstream remote recorded locally: `https://github.com/G-Research/tdigest-rs.git`

Source path used for this snapshot:

```text
/home/ingo/git/tdigest-rs/tdigest-rs
```

Source commit:

```text
8d9cf1fe82c5c9254e5dc9593f01717e317f4333
```

Copied files are intentionally kept byte-for-byte identical to the source
snapshot. Verification command:

```bash
diff -qr /home/ingo/git/tdigest-rs/tdigest-rs/bindings/python/tests compat/tdigest-rs-upstream/bindings/python/tests
diff -q /home/ingo/git/tdigest-rs/tdigest-rs/bindings/python/benchmarks/run.py compat/tdigest-rs-upstream/bindings/python/benchmarks/run.py
```

## SHA256

```text
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  bindings/python/tests/__init__.py
61a4ea1a12b6ec3f835a2e5086fc7aa70c5dc02cb100befe5db7fb9abd47a074  bindings/python/tests/conftest.py
3ca8e570306c33a08076193f59e6255e2e9cc51dd5914b324ea1277dcee86365  bindings/python/tests/constants.py
1ec4ef38fbf1a650aa36ed5ca459aa1cb2c63339e5f47bd9a685dbb9d0e3b3e3  bindings/python/tests/test_distributions.py
aa2b939ce5c5f90f10acb4080475cf939d7d3b0dc0dea004dbf345fda930af9a  bindings/python/tests/test_tdigest.py
94d05a31fe66c9712770f88ecb09adfb484ba179c3c01f69de90bc55f8538d64  bindings/python/tests/utils.py
2161ce8d9c5be6b0cfd9ef7f08a389838681ce2989aa8c87c9018ec12f506662  bindings/python/benchmarks/run.py
```
