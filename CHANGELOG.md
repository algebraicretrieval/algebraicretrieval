# Changelog

## Unreleased

Public software snapshot under active paper preparation.

- typed mathematical notation with `τₖ`, support restriction `▷`, weight modulation `⊙`, explicit RRF fusion `⊕`, and `E @ q`;
- single-assignment, top-down, parenthesized macros with quote-safe retrieval text and fail-closed rebinding/forward/cyclic references;
- equivalent ASCII spellings for restriction, modulation, fusion, rescore, and selection;
- typed SQLite contracts for `R`, `E`, `q`, `m`, and `w`;
- direct `algebra('<mathematical program>')` queries through MCP and CLI;
- mixed SQL/NumPy/graph planning through the existing flex runtime;
- SQL-derived support masks and aligned scalar weights with distinct operators;
- fail-closed vector dimensions;
- multiple `algebra(...)` relations in one outer SQL statement;
- six-case independent score-device validation against FAISS `IndexFlatIP`, with PyTerrier limited to relation construction and cutoff/top-k;
- raw float32 inner-product semantics: normalized primitives, magnitude-preserving linear compositions, and `1e-6` comparison tolerance;
- composed-score cutoff coverage and normalization/tie-break/zero-fill mutation detection;
- score-unit enforcement for commensurable `+` and deterministic `score DESC, id ASC` selection;
- hard rejection of legacy mask multiplication, implicit heterogeneous `+` fusion, and multiplication-spelled rescore/selection.
