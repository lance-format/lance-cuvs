# lance-cuvs

Experimental cuVS-backed vector build backend for Lance.

This repository is being split out from the main Lance tree. The current implementation still targets Lance's partition-local artifact build path and depends on Lance crates via local path dependencies.

It can also be installed as a Python package with `maturin develop --release`, exposing:

- `lance_cuvs.build_ivf_pq_artifact(...)`
- `lance_cuvs.create_ivf_pq_index(...)`

The current source build assumes a sibling `../lance` checkout is available.
