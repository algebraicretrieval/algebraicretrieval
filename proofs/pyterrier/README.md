# Independent score-device and PyTerrier relation receipt

`receipt.json` records six bounded comparisons over the 11,429-document Vaswani collection.

## Numerical contract

```text
E rows                         L2-normalized float32
primitive query vectors       L2-normalized individually
composed query vector         not normalized
E @ q                          raw float32 inner product
execution partition           one query per execution
scorer support                preserve input identities and order
selection tie policy          score DESC, id ASC
comparison tolerance          absolute 1e-6
```

A primitive unit query is cosine-equivalent. Retaining the magnitude of a signed composition preserves the defined inner-product function in real arithmetic. Float32 evaluation orders may differ by rounding; the tolerance is an acceptance criterion for these fixtures, not a universal error bound. The receipt checks resulting ranks and threshold membership directly.

## Independent oracle

Algebra scores through its NumPy reference device. The numerical oracle is FAISS 1.15.0 `IndexFlatIP`, which performs exact inner-product search without normalizing the query. For the collapse cases, the reference performs two separate FAISS searches and combines their aligned score arrays. PyTerrier 1.1.2 constructs the result relation and applies cutoff/top-k; it is no longer the numerical oracle.

| Algebra expression | Obligation | Result |
|---|---|---|
| `τ₁₀(E @ q1)` | primitive score and selection | 0 rank/id differences; delta `0.0` |
| `τ₁₀(m ▷ (E @ q1))` | support alignment and pushdown | 0 differences; delta `0.0` |
| `τ₁₀((E @ q1) - 0.5·(E @ q2))` | one folded matmul versus two FAISS searches | 0 differences; delta `2.98e-8` |
| `τ₅₀₀(threshold(0.20, E @ q1))` | ordinary cutoff | 294 rows; 0 differences; delta `1.49e-8` |
| `τ₅₀₀(threshold(0.20, (E @ q1) - 0.5·(E @ q2)))` | collapse observed through cutoff | 13 rows; 0 differences; delta `2.98e-8` |
| `τ₁₀(m ▷ (w ⊙ (E @ q1)))` | aligned weighting on the 52-id restricted support | 0 differences; delta `0.0` |

“0 differences” means no rank/identity difference and no score difference above the declared absolute tolerance. The receipt preserves the actual maximum score delta rather than rounding it to zero.

The weight expression retains the receipt's original source spelling. Its runtime plan restricts candidates before applying weights from the same 52-id relation. This establishes agreement on that common support, not weight coverage of the entire corpus or general enforcement of the paper's total-weight precondition.

## Alternative-semantics checks

The same receipt records deliberately wrong alternatives:

```text
normalize composed q     correct 13 rows → mutated 37 rows    detected
remove id tie-break      correct [a,z] → mutated [z,a]        detected
zero-fill a mask         correct [a,b] → mutated [c,d]        detected
```

The normalization alternative is executed through FAISS; the tie check compares the actual ordering helper with stipulated input order; the mask alternatives are calculated in Python. Their differing outputs expose the stated semantic errors, but they are not three injected compiler mutations. The separate four-row regression in [`tests/test_contract.py`](../../tests/test_contract.py) runs support, signed scores, weighting, and a cutoff tie through the actual compiler with literal expected results.

## Reproduce

The runner expects a [flex source checkout](https://github.com/damiandelmas/flex) or installation plus the Python packages below. The fixture builder also requires Java; the fresh-clone check used OpenJDK 21 and Python 3.11.

```bash
python -m pip install 'pyterrier[java]==1.1.2' 'faiss-cpu==1.15.0' \
  'numpy==2.4.6' 'pandas==3.0.5' 'ir-datasets==0.6.3'
python proofs/pyterrier/build_fixture.py

FLEX_SOURCE_ROOT=/path/to/flex \
python proofs/pyterrier/run.py > proofs/pyterrier/receipt.json
```

The fixture builder downloads `ir_datasets:vaswani`, creates the FTS5 corpus, constructs deterministic 128-dimensional signed-hash vectors, and stores live PyTerrier BM25 and TF_IDF result relations. The vectors are a deterministic compiler/scorer fixture, not a retrieval-quality benchmark.

### Paper examples and rounding check

[`examples.py`](examples.py) runs the paper's three Algebra expressions, complete sqlite-vec SQL queries, and native PyTerrier pipelines. It also compares folded and two-pass NumPy scoring, reproduces sqlite-vec 0.1.9's scalar cosine accumulation and distance rounding for documents `7429` and `9217`, and checks that all 52 BM25 weights agree across the three paths.

After the fixture and dependencies above are available:

```bash
python -m pip install 'sqlite-vec==0.1.9'
FLEX_SOURCE_ROOT=/path/to/flex python proofs/pyterrier/examples.py
```

Both runners accept `VASWANI_DB=/path/to/vaswani.db` to reuse an existing fixture. `examples.py` opens it read-only and creates only temporary SQL views. It prints versions, ordered results, score differences aligned by identity, and the intermediate rounding values as JSON. Exit status zero requires agreement within `1e-6`, identical normalized weights, and reproduction of the reported tie split; the exact rounding observation is specific to this fixture and the reported execution environment.

## Digests

Public receipt SHA-256:

```text
53da6adcafc108f19548968e4b2377afb589bc715a7800c2e3313ca6bd19fffe
```

Algebra Python bundle SHA-256:

```text
d3d16edf465675a920a3240af66e79479043af54dfa5afe9da72057385a799e1
```

The public and installed receipt hashes are intentionally different. The public digest covers the compact fixture, score-device contract, six cases, mutation results, and copied Algebra bundle. The installed digest covers a larger envelope containing that public score proof plus absolute installation paths, contract orientation, agent-transport rows, and live embedded-cell smoke evidence. The shared score-proof and bundle digests are the cross-envelope integrity checks; numerical results are not expected to differ.
