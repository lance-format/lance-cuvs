# lance-cuvs

Experimental cuVS-backed vector build backend for Lance.

This package targets Lance's partition-local artifact build path and currently depends on Lance git versions that include unreleased build APIs.
Use Python 3.12+ so the installed RAPIDS wheels match the `cuvs-sys` 26.2.0 toolchain.

It can also be installed as a Python package with `maturin develop --release`, exposing:

- `lance_cuvs.train_ivf_pq(...)`
- `lance_cuvs.build_ivf_pq_artifact(...)`

`train_ivf_pq(...)` returns Arrow-native training outputs for IVF centroids and PQ codebook.

`lance-cuvs` is a backend API provider. It stops at training and partition-artifact build output, and does not finalize or register Lance indices on behalf of callers.
