"""Algebraic retrieval — a materializer that exposes the arithmetic query surface.

Thesis (research arc: algebraic-retrieval): a recurring linear kernel is
    result = select(m ▷ (w ⊙ (E @ q)))
where Mask restriction changes support and Weight modulation changes scores.
Fusion, transforms, feedback, and selection remain explicit barriers around that
kernel. SQL is one front-end dialect; mathematical notation is another. Both
compile through the existing engine rather than a second retrieval framework.

    FROM algebra('q: similar(architecture) - similar(deployment)
                  | m: type=user_prompt recent:7
                  | w: decay:7
                  | top: 10') a
    JOIN chunks c ON a.id = c.id
    ORDER BY a.score DESC

The materializer parses that spec into (q, m, w, top), compiles it to a vec_ops
token string + a SQL pre-filter mask, calls the already-registered vec_ops UDF,
and materializes (id, score) into a temp table — identical handoff to keyword().

What it deliberately does NOT do yet (the boundary, named):
  - per-term suppression coefficients (vec_ops suppress: uses a fixed internal
    alpha; the grammar accepts a coefficient and records it, engine approximates)
  - column weights (centrality, bridge_score) as element-wise w — needs a
    post-score fold; decay compiles natively because vec_ops has it.
These are exactly where transpile-to-existing-engine stops being faithful, which
is the finding, not a defect.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import uuid

_SIB: dict = {}


def _claim_query_prefix() -> None:
    """Tell flex that algebra(...) is a query before SQL validation runs."""
    try:
        from flex.mcp_core import register_query_prefix
    except ImportError:
        return
    register_query_prefix("algebra(")


# flex loads external plugins before dispatching MCP/CLI queries. Claim the
# constructor at import time so the core validator routes it to this materializer.
if os.environ.get("FLEX_ALGEBRA"):
    _claim_query_prefix()


def _sibling(name: str):
    """Load a sibling module (engine.py / compose.py) by path — robust regardless
    of how the plugin itself was loaded by flex. Memoized."""
    if name not in _SIB:
        import sys as _sys
        p = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"{name}.py")
        spec = importlib.util.spec_from_file_location(f"_alg_{name}", p)
        m = importlib.util.module_from_spec(spec)
        _sys.modules[spec.name] = m   # register before exec — @dataclass under
        spec.loader.exec_module(m)    # `from __future__ annotations` needs it
        _SIB[name] = m
    return _SIB[name]


# ── detection (mirror keyword.py / vec_ops.py exactly) ────────────────────────

def algebra_materializer(db, sql: str) -> str:
    """Materialize algebra('spec') as a temp table of (id, score).

    Passthrough (returns sql unchanged) when no algebra() table source is found.
    Returns JSON error string on a bad spec or failed compile.
    """
    start = re.search(r'algebra\s*\(', sql)
    if not start:
        return sql

    end_pos = _match_paren(sql, start.end() - 1)
    if end_pos is None:
        return sql

    # CLOSED SURFACE: when the WHOLE query is `algebra('<expr>')` — nothing before,
    # nothing after — the agent wrote PURE ALGEBRA. We synthesize the host SELECT
    # internally (ordered records: content + provenance), so the agent never writes
    # SELECT/JOIN/ORDER BY. Otherwise algebra() is a table source inside the agent's
    # own SQL (the legacy/power form), unchanged.
    before_txt = sql[:start.start()].strip()
    after_txt = sql[end_pos:].strip().rstrip(';').strip()
    is_bare = (before_txt == "" and after_txt == "")
    if not is_bare:
        before = before_txt.upper()
        if not (before.endswith('FROM') or before.endswith('JOIN') or before.endswith(',')):
            return json.dumps({"error":
                "algebra() must be the whole query — algebra('<expr>') — or a table "
                "source after FROM/JOIN. Pure form: algebra('top(10, similar(X))')"})

    inner = sql[start.end():end_pos - 1].strip()
    if len(inner) >= 2 and inner[0] == "'" and inner[-1] == "'":
        inner = inner[1:-1].replace("''", "'")
    if not inner.strip():
        return json.dumps({"error": "algebra() requires a non-empty spec"})

    try:
        contract = _sibling("contract").Contract.discover(db)
    except Exception as e:
        return json.dumps({"error": f"algebra() contract failed: {e}"})

    # orient — the cell projected into the algebra's operand space (in-dialect
    # orientation): masks/scores/transforms/operators/weights/selectors with live
    # domains plus an availability flag. `SELECT * FROM algebra('orient')`.
    if inner.strip().lower() == "orient":
        try:
            orows = (
                contract.orient_rows(db)
                if contract is not None
                else _sibling("orient").orient_rows(db)
            )
        except Exception as e:
            return json.dumps({"error": f"algebra('orient') failed: {e}"})
        tmp = f"_alg_orient_{uuid.uuid4().hex[:8]}"
        db.execute(f"CREATE TEMP TABLE [{tmp}] "
                   "(section TEXT, token TEXT, domain TEXT, status TEXT)")
        if orows:
            db.executemany(
                f"INSERT INTO [{tmp}] (section, token, domain, status) VALUES (?,?,?,?)",
                orows)
        if is_bare:
            return f"SELECT section, token, domain, status FROM [{tmp}]"
        rewritten = sql[:start.start()] + tmp + sql[end_pos:]
        return algebra_materializer(db, rewritten)

    # Two front-ends over one IR:
    #   arithmetic notation  (E @ q1 - 0.5*E @ q2, top(...), centroid(top(...)))  → the compiler
    #   pipe-spec slots       (q: … | m: … | w: …)                                → transpile (legacy)
    # Detect the slot form by its `q:` marker; everything else is arithmetic.
    is_pipe = bool(re.search(r"\bq\s*:", inner))
    if not is_pipe:
        # Arithmetic → the planner (run_chain): restriction pushdown + Scope
        # threading + PRF and explicit selector barriers. Always materialize a temp table
        # (even on 0 rows) so the rewritten SQL never hits "no such table".
        try:
            planner = _sibling("planner")
            if contract is not None:
                top = _sibling("math").parse_math(inner, contract.bindings())
                embed_fn, E, ids = contract.runtime(db)
            else:
                eng = _sibling("engine"); parse_m = _sibling("parse")
                normalized = _sibling("math").transpile(inner)
                top = parse_m.parse(normalized)
                embed_fn, E, ids = eng.cell_engine_from_db(db)
            sig_map = {}
            try:
                pairs, sig_map = planner.run_records(
                    top, embed_fn, E, ids, db, contract=contract
                )
            except NotImplementedError:
                pairs, _log, _mm = _sibling("lower").run(top, embed_fn, E, ids, db)
        except Exception as e:
            return json.dumps({"error": f"algebra() compile failed: {e}"})
        if is_bare:
            # closed surface: ordered records (content + provenance). A rank column
            # preserves the algebra's output order (earliest=oldest-first, top=by
            # score) — the (id,score) temp-table set loses it; this keeps it.
            # signal = corpus-null z (distinctiveness vs the scored pool's null);
            # NULL on kw/fuse/transform paths that ran no cosine scorer. Arity-robust:
            # planner emits 3-tuples (id, score, signal); the lower.run fallback 2-tuples.
            tmp = f"_alg_surface_{uuid.uuid4().hex[:8]}"
            db.execute(f"CREATE TEMP TABLE [{tmp}] "
                       "(rank INTEGER, id TEXT, score REAL, signal REAL)")
            if pairs:
                db.executemany(
                    f"INSERT INTO [{tmp}] (rank, id, score, signal) VALUES (?, ?, ?, ?)",
                    [(k, i, s,
                      (round(sig_map[i], 3) if sig_map.get(i) is not None else None))
                     for k, (i, s) in enumerate(pairs) if i is not None])
            if contract is not None:
                return (f"SELECT s.rank, s.id, s.score, s.signal "
                        f"FROM [{tmp}] s ORDER BY s.rank")
            return (f"SELECT s.id, round(s.score, 4) AS score, s.signal, "
                    f"datetime(r.timestamp, 'unixepoch') AS at, r.content "
                    f"FROM [{tmp}] s JOIN _raw_chunks r ON s.id = r.id ORDER BY s.rank")
        tmp = f"_alg_results_{uuid.uuid4().hex[:8]}"
        db.execute(f"CREATE TEMP TABLE [{tmp}] (id TEXT PRIMARY KEY, score REAL)")
        if pairs:
            db.executemany(
                f"INSERT OR IGNORE INTO [{tmp}] (id, score) VALUES (?, ?)",
                [(i, s) for i, s in pairs if i is not None])
        rewritten = sql[:start.start()] + tmp + sql[end_pos:]
        return algebra_materializer(db, rewritten)

    try:
        plan = compile_spec(inner)
    except SpecError as e:
        return json.dumps({"error": f"algebra() spec: {e}"})

    # Reuse the scoring engine: call the registered vec_ops UDF directly.
    try:
        if plan['prefilter']:
            row = db.execute("SELECT vec_ops(?, ?)", (plan['tokens'], plan['prefilter'])).fetchone()
        else:
            row = db.execute("SELECT vec_ops(?)", (plan['tokens'],)).fetchone()
    except Exception as e:
        return json.dumps({"error": f"algebra() execution failed: {e}"})

    if not row or not row[0]:
        results = []
    else:
        try:
            results = json.loads(row[0])
        except Exception as e:
            return json.dumps({"error": f"algebra() bad engine response: {e}"})

    if isinstance(results, dict) and 'error' in results:
        return json.dumps(results)
    if not isinstance(results, list):
        results = []

    if plan['top']:
        results = results[:plan['top']]

    tmp = f"_alg_results_{uuid.uuid4().hex[:8]}"
    db.execute(f"CREATE TEMP TABLE [{tmp}] (id TEXT PRIMARY KEY, score REAL)")
    if results:
        db.executemany(
            f"INSERT OR IGNORE INTO [{tmp}] (id, score) VALUES (?, ?)",
            [(r.get('id'), r.get('score')) for r in results if r.get('id') is not None],
        )

    rewritten = sql[:start.start()] + tmp + sql[end_pos:]
    return algebra_materializer(db, rewritten)


def _match_paren(s: str, open_idx: int):
    depth, in_quote, i = 0, False, open_idx
    while i < len(s):
        c = s[i]
        if in_quote:
            if c == "'":
                if i + 1 < len(s) and s[i + 1] == "'":
                    i += 2
                    continue
                in_quote = False
        else:
            if c == "'":
                in_quote = True
            elif c == '(':
                depth += 1
            elif c == ')':
                depth -= 1
                if depth == 0:
                    return i + 1
        i += 1
    return None


# ── compiler: arithmetic spec → (vec_ops tokens, pre-filter SQL, top) ─────────

class SpecError(ValueError):
    pass


_SIMILAR = re.compile(r'similar\(\s*(.*?)\s*\)', re.IGNORECASE)
_CENTROID = re.compile(r'centroid\(\s*(.*?)\s*\)', re.IGNORECASE)
# a signed term: optional +/-, optional coefficient, then similar(...)
_TERM = re.compile(r'([+-]?)\s*([0-9]*\.?[0-9]+)?\s*(similar|centroid)\(\s*(.*?)\s*\)', re.IGNORECASE)


def compile_spec(spec: str) -> dict:
    """Parse 'q: ... | m: ... | w: ... | top: N' into an execution plan.

    Returns {tokens, prefilter, top, compiled} where `compiled` echoes the
    arithmetic→vec_ops correspondence for inspection (the two-views property).
    """
    clauses = {}
    for part in spec.split('|'):
        part = part.strip()
        if not part:
            continue
        if ':' not in part:
            raise SpecError(f"clause missing ':' — got {part!r}")
        key, body = part.split(':', 1)
        clauses[key.strip().lower()] = body.strip()

    if 'q' not in clauses:
        raise SpecError("missing required 'q:' (query vector) clause")

    tokens, notes = _compile_q(clauses['q'])
    tokens += _compile_w(clauses.get('w', ''))
    prefilter = _compile_m(clauses.get('m', ''))
    top = _compile_top(clauses.get('top', clauses.get('limit', '')))
    if top:
        tokens.append(f"pool:{max(top * 5, 200)}")

    token_str = ' '.join(t for t in tokens if t)
    return {
        'tokens': token_str,
        'prefilter': prefilter,
        'top': top,
        'compiled': {'vec_ops': token_str, 'mask_sql': prefilter, 'notes': notes},
    }


def _compile_q(q: str) -> tuple[list[str], list[str]]:
    terms = list(_TERM.finditer(q))
    if not terms:
        raise SpecError("q: needs at least one similar(text) or centroid(ids)")
    tokens, notes = [], []
    base_set = False
    for m in terms:
        sign, coef, fn, arg = m.group(1), m.group(2), m.group(3).lower(), m.group(4)
        coef_f = float(coef) if coef else 1.0
        if fn == 'centroid':
            tokens.append(f"centroid:{arg.replace(' ', '')}")
            continue
        # similar(text)
        if sign == '-':
            tokens.append(f"suppress:{arg}")
            if coef and coef_f != 1.0:
                notes.append(f"suppress coefficient {coef_f} not honored (vec_ops fixed alpha)")
        else:
            if not base_set:
                tokens.insert(0, f"similar:{arg}")
                base_set = True
                if coef and coef_f != 1.0:
                    notes.append(f"base coefficient {coef_f} ignored (query vector is normalized)")
            else:
                # additional positive term — vec_ops has no multi-add token; fold as centroid hint
                notes.append(f"extra positive term similar({arg}) approximated; "
                             f"multi-term query-vector add is v1.1 (raw-vector compose)")
    if not base_set and not any(t.startswith('centroid:') for t in tokens):
        raise SpecError("q: needs a positive similar(text) or a centroid(ids) anchor")
    return tokens, notes


def _compile_w(w: str) -> list[str]:
    if not w:
        return []
    tokens = []
    for tok in re.split(r'[\s&]+', w.strip()):
        tok = tok.strip()
        if not tok:
            continue
        low = tok.lower()
        if low == 'diverse':
            raise SpecError(
                "diverse is a selection barrier, not a weight; use diverse(lambda, expression)")
        if low.startswith('decay'):
            m = re.search(r'decay\s*[:=]?\s*(\d+)', low)
            tokens.append(f"decay:{m.group(1) if m else '7'}")
            continue
        raise SpecError(f"unknown scalar weight {tok!r}; the legacy w: slot accepts decay only")
    return tokens


def _quote(v: str) -> str:
    return "'" + v.replace("'", "''") + "'"


def _compile_m(m: str) -> str | None:
    if not m:
        return None
    preds = []
    # Extract raw:(...) with balanced parens (SQL predicates nest parens),
    # then strip those spans before tokenizing the rest.
    m_wo_raw = m
    while True:
        hit = re.search(r'raw:\(', m_wo_raw, re.IGNORECASE)
        if not hit:
            break
        close = _match_paren(m_wo_raw, hit.end() - 1)
        if close is None:
            raise SpecError("raw:( ... ) has unbalanced parens")
        preds.append(f"({m_wo_raw[hit.end():close - 1].strip()})")
        m_wo_raw = m_wo_raw[:hit.start()] + ' ' + m_wo_raw[close:]
    for tok in re.split(r'[\s&]+', m_wo_raw.strip()):
        tok = tok.strip()
        if not tok:
            continue
        if '=' in tok and not tok.lower().startswith('recent'):
            col, val = tok.split('=', 1)
            preds.append(f"{col.strip()} = {_quote(val.strip())}")
        elif '~' in tok:
            col, val = tok.split('~', 1)
            preds.append(f"{col.strip()} LIKE {_quote('%' + val.strip() + '%')}")
        elif tok.lower().startswith('recent'):
            mm = re.search(r'recent\s*[:=]?\s*(\d+)', tok.lower())
            days = mm.group(1) if mm else '7'
            preds.append(f"timestamp >= CAST(strftime('%s','now','-{days} days') AS INTEGER)")
        else:
            raise SpecError(f"unrecognized mask predicate {tok!r} "
                            f"(use col=val, col~substr, recent:N, or raw:(sql))")
    if not preds:
        return None
    return "SELECT id FROM chunks WHERE " + " AND ".join(preds)


def _compile_top(t: str) -> int | None:
    if not t:
        return None
    m = re.search(r'(\d+)', t)
    return int(m.group(1)) if m else None


# ── registration (the only thing flex discovers) ──────────────────────────────

def register_query_materializers():
    # Default-OFF gate. The live MCP service auto-discovers every module under
    # ~/.flex/modules and joins its materializer into the chain for EVERY query.
    # algebra is research-tier and rides the hot path for all retrieval, so it
    # registers only when FLEX_ALGEBRA is explicitly set (on the service env or
    # the CLI). Unset → returns [] → discovered-but-inert, zero overhead/risk.
    # Reversible: unset the env + restart.
    if not os.environ.get("FLEX_ALGEBRA"):
        return []
    # Also claim here for hosts that enable the module after initial import.
    _claim_query_prefix()
    return [algebra_materializer]
