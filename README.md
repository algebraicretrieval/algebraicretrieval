# Algebraic Retrieval

**Composable retrieval over masks and weights.**

$$
\tau_k \left(m \triangleright \left(w \odot (E @ q)\right)\right)
$$

Equivalent ASCII source is accepted directly:

```text
top(10, restrict(m, modulate(w, E @ q)))
```

We present an agent-written mathematical retrieval language whose typed expressions form an IR in which relations, query vectors, masks, weights, and scored relations compose as retrieval programs. These programs are mechanically lowered through an existing SQL/NumPy runtime. Six bounded cases compare the Algebra scorer with independent exact FAISS inner-product results shaped into relations by PyTerrier.

The implementation is an experimental external module for [flex](https://github.com/damiandelmas/flex). It reuses the existing IR, planner, execution backends, and SQLite materializer path rather than introducing another retrieval framework.

This repository is a public software snapshot accompanying the paper **“Algebraic Retrieval: Composable Retrieval over Masks and Weights”** (in preparation).

## Lineage

Algebraic Retrieval builds directly on [*flexvec: SQL Vector Retrieval with Programmatic Embedding Modulation*](https://arxiv.org/abs/2603.22587) (Delmas, 2026). flexvec introduced Programmatic Embedding Modulation (PEM), exposing the embedding matrix and score array as programmable surfaces and integrating query and score transformations into SQL through a query materializer.

Algebraic Retrieval moves that model into the query representation itself: relations, query vectors, masks, weights, and scored relations become typed operands in an agent-written mathematical language that lowers through the existing SQL/NumPy runtime. PEM is the immediate execution precursor and one operator family within the broader algebra.

## Agent interface

The agent writes a mathematical retrieval program:

```text
τ₁₀(
  (type(user_prompt) & recent(30))
  ▷ (E @ similar(algebra planner))
)
```

The program denotes a scored relation. Masks restrict its support through `▷`; weights modulate surviving scores through `⊙`; fusion uses `⊕`; transforms and selectors remain explicit. The parser rejects mask multiplication, heterogeneous `+` fusion, multiplication-spelled rescoring, and selectors used as weights.

Precedence is `@`, then scalar `*` / weight `⊙` / mask conjunction `&`, then restriction `▷`, then `+` / `-` / fusion `⊕`. Examples use canonical parentheses. `&` intersects declared mask support. Mask union is simply outside the current implemented/proven surface; complement and negation additionally require a closed universe and could synthesize support the contract never declared.

Numerically, `E` has L2-normalized float32 rows and primitive query vectors are normalized individually. `E @ q` is raw float32 inner product. A unit primitive is therefore cosine-equivalent, while a signed composition retains its magnitude so linear collapse preserves scores and absolute thresholds. Score comparisons use an absolute tolerance of `1e-6`; ranking uses `score DESC, id ASC`.

A four-row regression proves that replacing excluded identities with score zero can make them outrank eligible negative scores. A separate three-vector regression proves that pushing restriction through centroid feedback reverses the result order, so the compiler preserves feedback as a barrier.

### Orient

Before composing a program, the agent asks the current cell which operands and operators are live:

```text
algebra('orient')
```

Orient returns:

```text
section | token | domain | status
```

`token` is exact writable syntax, `domain` describes its live relation or value range, and `status` is `ok` or `inert (<reason>)`. Contract-backed SQLite databases expose only explicitly declared retrieval semantics; schemas are not guessed. `mask(name)` and `weight(name)` resolve only declared operands. An undeclared name fails compilation rather than returning an empty relation.

### Query examples

Retrieve recent user intent about a failed deployment:

```text
τ₁₀(
  (type(user_prompt) & recent(30))
  ▷ (E @ similar(why the production deployment was rolled back))
)
```

Compose a target direction while suppressing a dominant but unwanted cluster:

```text
target = similar(implementation architecture and runtime behavior)
noise = similar(marketing copy documentation and product positioning)
m = type(assistant)

τ₁₀(
  m
  ▷ (
    (E @ target)
    - 0.5·(E @ noise)
  )
)
```

A negative composition is not monotone in either input ranking, so over-fetching candidates from `target` and reranking them is not exact. Collapse produces one query vector that an inner-product ANN device could search directly. A cosine ANN device preserves the order after normalizing that vector, but absolute thresholds must be rescaled by its norm. This repository proves exact scoring semantics; it does not yet integrate an ANN execution device.

Rerank a contract-declared first-pass support relation with vector scoring:

```text
τ₁₀(m ▷ (E @ q))
```

Here `m` is the typed operand advertised by `algebra('orient')`; it is not an arbitrary string-keyed lookup.

Apply an explicit score cutoff rather than requesting a fixed number of rows:

```text
τ₅₀₀(threshold(0.20, E @ q))
```

Fuse lexical evidence and recover its structural neighborhood:

```text
seed =
  kw("rolled back")
  ⊕ kw("superseded")
  ⊕ kw("tests passed")

τ₆₀(seed ⊕ window(12, expand(1, seed)))
```

Assignments are single-assignment, top-to-bottom, parenthesized macros. They may reference earlier names; rebinding, forward references, self-reference, and cycles fail. Expansion is not memoization: the two uses of `seed` above build and evaluate two equivalent subplans against the same query snapshot, and the planner currently performs no common-subexpression elimination across those relation-valued branches.

### Runtime demonstration over public arXiv papers

The runtime vocabulary is broader than the paper's declared executable subset. The following self-verifying example ran over 37,129 indexed chunks from public arXiv papers; it demonstrates execution and transport, not retrieval effectiveness.

Find the earliest papers in a relevant result set:

```text
earliest(
  5,
  kw("dense retrieval")
  ⊕ (E @ similar(dense retrieval systems))
)
```

The five returned papers were dated from April through September 2020. This is the strongest self-verifying runtime example: `earliest` selects by source time after retrieval rather than by score alone.

```text
2004.04906  2020-04-10  Dense Passage Retrieval for Open-Domain Question Answering
2005.00181  2020-05-01  Sparse, Dense, and Attentional Representations for Text Retrieval
2007.00808  2020-07-01  Approximate Nearest Neighbor Negative Contrastive Learning...
2009.08553  2020-09-17  Generation-Augmented Retrieval for Open-Domain Question Answering
2009.13013  2020-09-28  SPARTA: Efficient Open-Domain Question Answering...
```

### Transport constructor

MCP and CLI accept the program through one call-style constructor:

```text
algebra('<mathematical program>')
```

For example, the signed program above is submitted exactly as:

```text
algebra('
  target = similar(implementation architecture and runtime behavior)
  noise = similar(marketing copy documentation and product positioning)
  m = type(assistant)

  τ₁₀(m ▷ ((E @ target) - 0.5·(E @ noise)))
')
```

The materializer supplies the host `SELECT`, executes the compiled program, and returns ordered records. The SQL envelope is not part of the agent-facing retrieval language.

## SQL interoperability

An Algebra result is also an ordinary SQL relation. Outer SQL is available when a program needs hydration, joins, grouping, or comparison with another result:

```sql
WITH a AS (
  SELECT * FROM algebra('τ₅₀(E @ similar(virtual machine agent))')
), b AS (
  SELECT * FROM algebra('τ₅₀(E @ similar(durable actor runtime))')
)
SELECT a.id, a.score, b.score
FROM a JOIN b USING (id)
ORDER BY min(a.score, b.score) DESC;
```

SQL is therefore an execution and composition host, not the primary notation for retrieval programs.

## Independent score-device validation

The Vaswani receipt compares Algebra's NumPy reference scorer with FAISS 1.15.0 `IndexFlatIP` over the same 11,429 identities and deterministic 128-dimensional signed-hash vectors. For linear composition, the oracle performs two separate FAISS searches and combines their aligned scores; PyTerrier constructs the relation and applies cutoff/top-k.

| Expression | Result |
|---|---|
| `τ₁₀(E @ q1)` | 0 rank/id differences; delta `0.0` |
| `τ₁₀(m ▷ (E @ q1))` | 0 differences; delta `0.0` |
| `τ₁₀((E @ q1) - 0.5·(E @ q2))` | matmuls `2 → 1`; delta `2.98e-8` |
| `τ₅₀₀(threshold(0.20, E @ q1))` | 294 rows; delta `1.49e-8` |
| `τ₅₀₀(threshold(0.20, (E @ q1) - 0.5·(E @ q2)))` | 13 rows; delta `2.98e-8` |
| `τ₁₀(m ▷ (w ⊙ (E @ q1)))` | 0 differences; delta `0.0` |

No case has a rank/identity difference or a score difference above `1e-6`. Normalized-composition, missing-tie-break, and zero-filled-mask mutations are all detected. See [`proofs/pyterrier`](proofs/pyterrier) for the exact contract, mutation results, and receipt boundaries.

## Repository layout

```text
module/algebra/      flex external module — parser, IR, planner, backends, materializer
proofs/pyterrier/    bounded comparison runner and machine-readable receipt
tests/                portable surface, contract, transport, and dimension regressions
```

## Run the portable regressions

```bash
python -m unittest discover -s tests -v
```

## Load as a flex external module

```bash
export FLEX_MODULE_PATH="$PWD/module"
export FLEX_ALGEBRA=1

flex search --cell <cell> --json "algebra('orient')"
```

For large vector cells, use the persistent flex MCP service so its vector cache remains warm.

## Citation

Citation metadata is in [`CITATION.cff`](CITATION.cff). A version DOI will be attached when the first release is archived through Zenodo; the paper citation will become the preferred citation when the manuscript is public.

## Author

Damian Delmas — initial implementation, research direction, and public snapshot.
