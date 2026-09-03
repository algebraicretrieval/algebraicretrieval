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

A primitive unit query is cosine-equivalent. Retaining the magnitude of a signed composition makes linear collapse score-preserving and keeps absolute thresholds stable.

## Independent oracle

Algebra scores through its NumPy reference device. The numerical oracle is FAISS 1.15.0 `IndexFlatIP`, which performs exact inner-product search without normalizing the query. For the collapse cases, the reference performs two separate FAISS searches and combines their aligned score arrays. PyTerrier 1.1.2 constructs the result relation and applies cutoff/top-k; it is no longer the numerical oracle.

| Algebra expression | Obligation | Result |
|---|---|---|
| `τ₁₀(E @ q1)` | primitive score and selection | 0 rank/id differences; delta `0.0` |
| `τ₁₀(m ▷ (E @ q1))` | support alignment and pushdown | 0 differences; delta `0.0` |
| `τ₁₀((E @ q1) - 0.5·(E @ q2))` | one folded matmul versus two FAISS searches | 0 differences; delta `2.98e-8` |
| `τ₅₀₀(threshold(0.20, E @ q1))` | ordinary cutoff | 294 rows; 0 differences; delta `1.49e-8` |
| `τ₅₀₀(threshold(0.20, (E @ q1) - 0.5·(E @ q2)))` | collapse observed through cutoff | 13 rows; 0 differences; delta `2.98e-8` |
| `τ₁₀(m ▷ (w ⊙ (E @ q1)))` | support plus total aligned weighting | 0 differences; delta `0.0` |

“0 differences” means no rank/identity difference and no score difference above the declared absolute tolerance. The receipt preserves the actual maximum score delta rather than rounding it to zero.

## Mutation detection

The same receipt records deliberately wrong alternatives:

```text
normalize composed q     correct 13 rows → mutated 37 rows    detected
remove id tie-break      correct [a,z] → mutated [z,a]        detected
zero-fill a mask         correct [a,b] → mutated [c,d]        detected
```

These establish that the harness can fail on the numerical, ordering, and support defects it claims to test.

## Reproduce

The runner expects a flex source checkout or installation plus NumPy, pandas, FAISS CPU 1.15.0, PyTerrier 1.1.2, and `ir_datasets`.

```bash
python -m pip install 'pyterrier[java]==1.1.2' 'faiss-cpu==1.15.0' ir-datasets numpy pandas
python proofs/pyterrier/build_fixture.py

FLEX_SOURCE_ROOT=/path/to/flex \
python proofs/pyterrier/run.py > proofs/pyterrier/receipt.json
```

The fixture builder downloads `ir_datasets:vaswani`, creates the FTS5 corpus, constructs deterministic 128-dimensional signed-hash vectors, and stores live PyTerrier BM25 and TF_IDF result relations. The vectors are a deterministic compiler/scorer fixture, not a retrieval-quality benchmark.

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
