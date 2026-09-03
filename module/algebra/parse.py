"""Arithmetic expression front-end → IR.

Parses the real notation the thesis is about — e.g.

    E @ similar("architecture") - 0.5 * E @ similar("deployment")
    top(10, recent(90) ▷ (decay(30) ⊙ (E @ similar("auth"))))
    top(10, E @ centroid(top(20, E @ similar("jwt rotation"))))

into the canonical Top(ScoreExpr) IR. Operators (precedence high→low):
    @       matrix @ query → score                    30
    *  ⊙  & scalar scale / weight modulation / mask-and  20
    ▷       support restriction                       15
    + - ⊕   score combine / RRF fusion                10

The operators are deliberately not aliases: `*` scales by a scalar, `⊙`
modulates scores by a Weight, `▷` restricts support by a Mask, and `⊕` fuses
ranked branches with RRF. Functions include similar, centroid, top, recent,
type, decay, threshold, diverse, dedup, rrf, rescore, and the bare matrix `E`.
"""

from __future__ import annotations

import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ir import QTerm, ScoreTerm, Mask, Weight, Selector, ScoreExpr, Top, Op, Restrict, Fuse, Rescore, Keyword  # noqa: E402

_OPS = set("@*+-&,()⊙▷⊕")
_BP = {"+": 10, "-": 10, "⊕": 10, "▷": 15, "*": 20, "⊙": 20, "&": 20, "@": 30}


class ParseError(ValueError):
    pass


# ── tokenizer: name(...) becomes a single CALL token (balanced raw inner) ─────

def _tokenize(s: str):
    toks, i, n = [], 0, len(s)
    while i < n:
        c = s[i]
        if c.isspace():
            i += 1
        elif c in '"\'':
            j = i + 1
            while j < n and s[j] != c:
                j += 1
            toks.append(("STR", s[i + 1:j])); i = j + 1
        elif c.isdigit() or (c == "." and i + 1 < n and s[i + 1].isdigit()):
            j = i
            while j < n and (s[j].isdigit() or s[j] == "."):
                j += 1
            toks.append(("NUM", float(s[i:j]))); i = j
        elif c.isalpha() or c == "_":
            j = i
            while j < n and (s[j].isalnum() or s[j] == "_"):
                j += 1
            name = s[i:j]
            k = j
            while k < n and s[k].isspace():
                k += 1
            if k < n and s[k] == "(":                 # function call: grab balanced parens
                depth, m = 0, k
                while m < n:
                    if s[m] == "(":
                        depth += 1
                    elif s[m] == ")":
                        depth -= 1
                        if depth == 0:
                            break
                    m += 1
                if depth != 0:
                    raise ParseError(f"unbalanced parens after {name}(")
                toks.append(("CALL", name, s[k + 1:m])); i = m + 1
            else:
                toks.append(("NAME", name)); i = j
        elif c in _OPS:
            toks.append(("OP", c)); i += 1
        else:
            raise ParseError(f"unexpected char {c!r}")
    return toks


# ── tagged intermediate values: ('matrix',) ('query',qterms) ('score',ScoreExpr)
#    ('mask',Mask) ('weight',Weight) ('scalar',float) ('top',Top) ──────────────

def _scale(val, c):
    tag = val[0]
    if tag == "scalar":
        return ("scalar", val[1] * c)
    if tag == "query":
        return ("query", [QTerm(t.coef * c, t.kind, t.arg) for t in val[1]])
    if tag == "score":
        e = val[1]
        return ("score", ScoreExpr([ScoreTerm(st.coef * c, st.q) for st in e.terms], e.mask, e.weights, e.selectors, e.unit))
    raise ParseError(f"cannot scale a {tag}")


def _to_score(val) -> ScoreExpr:
    if val[0] == "score":
        return val[1]
    if val[0] == "query":                 # implicit E@ — a bare query IS scored
        return ScoreExpr([ScoreTerm(1.0, val[1])])
    raise ParseError(f"expected a score expression, got {val[0]}")


def _S(val):
    """Promote a bare query to a score (implicit E@). Leaves other tags alone."""
    if val[0] == "query":
        return ("score", ScoreExpr([ScoreTerm(1.0, val[1])]))
    return val


def _combine(op, a, b):
    ta, tb = a[0], b[0]
    if op == "@":
        if ta == "matrix" and tb == "query":
            return ("score", ScoreExpr([ScoreTerm(1.0, b[1])]))
        raise ParseError("'@' must be  E @ <query>")
    if op == "*":
        if ta == "scalar":
            return _scale(b, a[1])
        if tb == "scalar":
            return _scale(a, b[1])
        raise ParseError("'*' is scalar multiplication only; use ▷ for Mask support restriction or ⊙ for Weight modulation")
    if op == "▷":
        if ta != "mask":
            raise ParseError("'▷' requires Mask[K] on the left and a scored relation on the right")
        restricted = _S(b)
        mask = a[1]
        if restricted[0] == "score":
            expr = restricted[1]
            merged = Mask((expr.mask.preds if expr.mask else []) + mask.preds)
            return ("score", ScoreExpr(expr.terms, merged, expr.weights, expr.selectors, expr.unit))
        if restricted[0] == "op":
            node = restricted[1]
            if isinstance(node, Keyword):
                node.mask = Mask((node.mask.preds if node.mask else []) + mask.preds)
                return ("op", node)
            if isinstance(node, Restrict):
                merged = Mask(node.mask.preds + mask.preds)
                return ("op", Restrict(merged, node.inner, node.weights, node.selectors))
            return ("op", Restrict(mask, node))
        raise ParseError(f"cannot restrict support of a {restricted[0]}")
    if op == "⊙":
        if ta == "mask" or tb == "mask":
            raise ParseError("Mask[K] changes support; use 'm ▷ S', never 'm ⊙ S'")
        if ta == "scalar" or tb == "scalar":
            raise ParseError("'⊙' is Weight modulation only; use '*' for scalar multiplication")
        if ta == "op" or tb == "op":
            opval = a if ta == "op" else b
            other = b if ta == "op" else a
            if other[0] == "weight":
                opval[1].weights = opval[1].weights + [other[1]]
                return ("op", opval[1])
            if other[0] == "weights":
                opval[1].weights = opval[1].weights + other[1]
                return ("op", opval[1])
            raise ParseError("'⊙' requires Weight[K] and Scored[K]; use rescore(query, relation) for reranking")

        def _weight_parts(v):
            v = _S(v)
            if v[0] == "score":
                return ("score", v[1], [])
            if v[0] == "weight":
                return ("weights", None, [v[1]])
            if v[0] == "weights":
                return ("weights", None, v[1])
            raise ParseError(f"cannot modulate scores by a {v[0]}; '⊙' requires Weight[K]")

        ka, ea, wa = _weight_parts(a)
        kb, eb, wb = _weight_parts(b)
        if ka == "score" and kb == "score":
            raise ParseError("cannot ⊙ two scored relations; use '+' only for commensurable scores")
        weights = wa + wb
        if ka == "score" or kb == "score":
            expr = ea if ka == "score" else eb
            return ("score", ScoreExpr(expr.terms, expr.mask, expr.weights + weights, expr.selectors, expr.unit))
        return ("weights", weights)
    if op == "&":
        if ta == "mask" and tb == "mask":
            return ("mask", Mask(a[1].preds + b[1].preds))
        raise ParseError("'&' combines two masks")
    if op == "⊕":
        parts = []
        for value in (a, b):
            if value[0] == "op":
                node = value[1]
                if isinstance(node, Fuse) and not node.weights and not node.selectors:
                    parts.extend(node.parts)
                else:
                    parts.append(node)
            else:
                parts.append(_to_score(value))
        return ("op", Fuse(parts))
    if op in ("+", "-"):
        sign = -1.0 if op == "-" else 1.0
        if ta == "op" or tb == "op":
            node = a[1] if ta == "op" else b[1]
            if isinstance(node, Keyword):
                raise ParseError(
                    "BM25 and cosine are different score substrates; use '⊕' or rrf(...) for rank fusion")
            raise ParseError(
                f"'{op}' is commensurable score arithmetic only; use '⊕' or rrf(...) for rank fusion")
        if ta == "query" and tb == "query":     # keep query-space composition (E@(a+b))
            return ("query", a[1] + [QTerm(t.coef * sign, t.kind, t.arg) for t in b[1]])
        if ta in ("query", "score") and tb in ("query", "score"):   # mixed → score-space
            ea, eb = _to_score(a), _to_score(b)
            if ea.selectors or eb.selectors:
                raise ParseError("cannot perform score arithmetic across a selection barrier")
            if ea.unit != eb.unit:
                raise ParseError(
                    f"score units are not commensurable: {ea.unit!r} != {eb.unit!r}; use explicit fusion")
            if ea.mask != eb.mask or ea.weights != eb.weights:
                raise ParseError(
                    "score arithmetic requires identical support and weight context; compose the linear scores first")
            terms = ea.terms + [ScoreTerm(st.coef * sign, st.q) for st in eb.terms]
            return ("score", ScoreExpr(terms, ea.mask, list(ea.weights), unit=ea.unit))
        raise ParseError(f"cannot {op} a {ta} and a {tb}")
    raise ParseError(f"unknown op {op}")


# ── precedence-climbing parser ────────────────────────────────────────────────

class _P:
    def __init__(self, toks):
        self.t, self.i = toks, 0

    def peek(self):
        return self.t[self.i] if self.i < len(self.t) else None

    def next(self):
        tok = self.t[self.i]; self.i += 1; return tok

    def expr(self, min_bp=0):
        lhs = self.atom()
        while True:
            tok = self.peek()
            if not tok or tok[0] != "OP" or tok[1] not in _BP:
                break
            op = tok[1]; bp = _BP[op]
            if bp < min_bp:
                break
            self.next()
            rhs = self.expr(bp + 1)
            lhs = _combine(op, lhs, rhs)
        return lhs

    def atom(self):
        tok = self.next()
        if tok[0] == "OP" and tok[1] == "-":           # prefix minus
            return _scale(self.atom(), -1.0)
        if tok[0] == "OP" and tok[1] == "(":
            v = self.expr(0)
            close = self.next()
            if not (close[0] == "OP" and close[1] == ")"):
                raise ParseError("expected )")
            return v
        if tok[0] == "NUM":
            return ("scalar", tok[1])
        if tok[0] == "STR":
            raise ParseError("bare string not allowed outside a call")
        if tok[0] == "NAME":
            if tok[1] == "E":
                return ("matrix",)
            raise ParseError(f"unknown name {tok[1]!r} (only E)")
        if tok[0] == "CALL":
            return _call(tok[1], tok[2])
        raise ParseError(f"unexpected token {tok}")


def _call(name, raw):
    name = name.lower()
    raw = raw.strip()
    if name == "similar":
        text = raw.strip().strip("'\"")
        return ("query", [QTerm(1.0, "similar", text)])
    if name in ("kw", "keyword"):
        # FTS5/BM25 scorer (set-valued plan node). Canonical token is `kw` — the
        # name `keyword(` collides with flex's own keyword() materializer, which
        # runs earlier in the chain and would eat the literal inside algebra('…')
        # when a `top(N,` comma precedes it. `keyword` stays accepted by Python.
        text = raw.strip().strip("'\"")
        return ("op", Keyword(text))
    if name == "restrict":
        args = _split_top_args(raw)
        if len(args) != 2:
            raise ParseError("restrict(mask, relation) needs exactly two arguments")
        return _combine(
            "▷",
            _P(_tokenize(args[0])).expr(0),
            _P(_tokenize(args[1])).expr(0),
        )
    if name == "modulate":
        args = _split_top_args(raw)
        if len(args) != 2:
            raise ParseError("modulate(weight, relation) needs exactly two arguments")
        return _combine(
            "⊙",
            _P(_tokenize(args[0])).expr(0),
            _P(_tokenize(args[1])).expr(0),
        )
    if name == "rrf":
        args = _split_top_args(raw)
        if len(args) < 2:
            raise ParseError("rrf(left, right, ...) needs at least two ranked branches")
        value = _P(_tokenize(args[0])).expr(0)
        for arg in args[1:]:
            value = _combine("⊕", value, _P(_tokenize(arg)).expr(0))
        return value
    if name == "rescore":
        args = _split_top_args(raw)
        if len(args) != 2:
            raise ParseError("rescore(query, relation) needs exactly two arguments")
        query = _to_score(_P(_tokenize(args[0])).expr(0))
        inner = _parse_inner(args[1])
        if isinstance(inner, ScoreExpr):
            raise ParseError("rescore(query, relation) requires a transformed or fused relation")
        return ("op", Rescore(query=query, inner=inner))
    if name == "centroid":
        if raw.lower() == "prev":
            return ("query", [QTerm(1.0, "centroid", "prev")])
        if re.match(r"^\s*top\s*\(", raw, re.IGNORECASE) or "@" in raw:
            inner = parse(raw)                          # nested → Top (data-dependent)
            return ("query", [QTerm(1.0, "centroid", inner)])
        # centroid over a SET defined by a mask: centroid(community(120)), centroid(type(user_prompt))
        try:
            v = _P(_tokenize(raw)).expr(0)
            if v[0] == "mask":
                return ("query", [QTerm(1.0, "centroid", v[1])])   # arg = Mask
        except ParseError:
            pass
        ids = [x.strip().strip("'\"") for x in raw.split(",") if x.strip()]
        return ("query", [QTerm(1.0, "centroid", ids)])
    if name == "recent":
        return ("mask", Mask([("recent", int(float(raw)))]))
    if name == "type":
        return ("mask", Mask([("type", raw.strip().strip("'\""))]))
    if name == "minlen":
        return ("mask", Mask([("minlen", int(float(raw)))]))
    if name == "community":
        return ("mask", Mask([("community", int(float(raw)))]))
    if name == "hub":
        return ("mask", Mask([("hub", 1)]))
    if name in ("repo", "project"):
        return ("mask", Mask([("repo", raw.strip().strip("'\""))]))
    if name == "eq":                       # generic column predicate — column from @orient,
        comma = _split_top_comma(raw)      # NOT hardcoded (doc_type/section/etc resolve here)
        if comma is None:
            raise ParseError("eq(column, value) needs two args, e.g. eq(doc_type, changelog)")
        return ("mask", Mask([("eq", raw[:comma].strip().strip("'\""),
                               raw[comma + 1:].strip().strip("'\""))]))
    if name == "raw":
        return ("mask", Mask([("raw", raw.strip().strip("'\""))]))
    if name == "mask":
        return ("mask", Mask([("contract", raw.strip().strip("'\""))]))
    if name == "weight":
        return ("weight", Weight("contract", raw.strip().strip("'\"")))
    if name == "decay":
        return _modifier("decay", raw, lambda s: int(float(s)), allow_bare=True)
    if name == "threshold":
        return _modifier("threshold", raw, float)
    if name == "diverse":
        return _modifier("diverse", raw, float, 0.7)
    if name == "dedup":
        return _modifier("dedup", raw, float, 0.97)
    if name == "top":
        comma = _split_top_comma(raw)
        if comma is None:
            raise ParseError("top(n, expr) needs two args")
        n = int(float(raw[:comma].strip()))
        return ("top", Top(n, _parse_inner(raw[comma + 1:])))
    # ── time-ordered selection (the time axis's selection corner) ──
    # earliest/latest are selection barriers (peers of top/diverse/dedup) that
    # re-order a relevance-scored set by SOURCE TIME instead of score. The N is a
    # count, like top's — so they parse exactly like top, setting Top.by. The
    # relevance floor lives inside the core (a nested top(K) can't carry K here:
    # _parse_inner drops a nested top's n).
    if name in ("earliest", "latest", "chrono"):
        comma = _split_top_comma(raw)
        if comma is None:
            raise ParseError(f"{name}(n, expr) needs a count and an expr, "
                             f"e.g. {name}(15, similar(genesis))")
        n = int(float(raw[:comma].strip()))
        by = "time:desc" if name == "latest" else "time:asc"
        return ("top", Top(n, _parse_inner(raw[comma + 1:]), by=by))
    # ── Class B scope-transformers: wrap an inner sub-plan, transform its set ──
    if name == "expand":
        comma = _split_top_comma(raw)
        if comma is not None and raw[:comma].strip().replace(".", "").isdigit():
            hops = int(float(raw[:comma].strip()))
            inner = _parse_inner(raw[comma + 1:])
        else:
            hops = 1
            inner = _parse_inner(raw)
        return ("op", Op("expand", {"hops": hops}, inner))
    if name in ("coedited", "same_file"):
        return ("op", Op("coedited", {}, _parse_inner(raw)))
    if name == "same_repo":
        return ("op", Op("same_repo", {}, _parse_inner(raw)))
    if name == "window":
        comma = _split_top_comma(raw)
        if comma is not None and raw[:comma].strip().replace(".", "").isdigit():
            return ("op", Op("window", {"w": int(float(raw[:comma].strip()))}, _parse_inner(raw[comma + 1:])))
        return ("op", Op("window", {"w": 3}, _parse_inner(raw)))
    if name == "surprise":
        comma = _split_top_comma(raw)
        if comma is None:
            raise ParseError("surprise(community, expr) needs a community id and an expr")
        community = raw[:comma].strip().strip("'\"")
        return ("op", Op("surprise", {"community": community}, _parse_inner(raw[comma + 1:])))
    raise ParseError(f"unknown function {name}()")


def _split_top_comma(raw: str):
    """Index of the first top-level comma (depth 0), or None. Commas inside
    nested calls/parens (e.g. centroid(top(20, ...))) are not top-level."""
    depth = 0
    for i, c in enumerate(raw):
        if c in "([":
            depth += 1
        elif c in ")]":
            depth -= 1
        elif c == "," and depth == 0:
            return i
    return None


def _split_top_args(raw: str) -> list[str]:
    args: list[str] = []
    start = 0
    depth = 0
    for i, c in enumerate(raw):
        if c in "([":
            depth += 1
        elif c in ")]":
            depth -= 1
        elif c == "," and depth == 0:
            args.append(raw[start:i].strip())
            start = i + 1
    args.append(raw[start:].strip())
    return [arg for arg in args if arg]


def _modifier(kind, raw, cast=float, default=None, allow_bare=False):
    """Apply a typed modifier. Only genuine weights may use bare `name(arg)`
    and compose through ⊙; support-changing selectors must wrap their input."""
    comma = _split_top_comma(raw)
    if comma is None:
        if not allow_bare:
            raise ParseError(f"{kind}(arg) changes selection; use {kind}(arg, relation)")
        body = raw.strip()
        return ("weight", Weight(kind, cast(body) if body else default))
    arg = cast(raw[:comma].strip())
    inner = _parse_inner(raw[comma + 1:])
    if kind in ("threshold", "diverse", "dedup"):
        selector = Selector(kind, arg)
        if isinstance(inner, ScoreExpr):
            return ("score", ScoreExpr(
                inner.terms, inner.mask, list(inner.weights),
                list(inner.selectors) + [selector], inner.unit
            ))
        inner.selectors = list(inner.selectors) + [selector]
        return ("op", inner)
    weight = Weight(kind, arg)
    if isinstance(inner, ScoreExpr):
        return ("score", ScoreExpr(
            inner.terms, inner.mask, list(inner.weights) + [weight],
            list(inner.selectors), inner.unit
        ))
    inner.weights = list(inner.weights) + [weight]
    return ("op", inner)


def _parse_inner(s: str):
    """Parse a sub-plan that may be a score expression OR a scope-transformer Op."""
    val = _P(_tokenize(s)).expr(0)
    if val[0] == "op":
        return val[1]
    if val[0] == "top":
        return val[1].expr
    return _to_score(val)


def parse_score(s: str) -> ScoreExpr:
    """Parse an expression that must denote a score-space expression."""
    val = _P(_tokenize(s)).expr(0)
    if val[0] == "top":
        return val[1].expr     # top(...) used where a score is expected — unwrap
    return _to_score(val)


def parse(s: str) -> Top:
    """Parse a full query → Top. Bare score expr / op (no top()) defaults to top(10)."""
    val = _P(_tokenize(s)).expr(0)
    if val[0] == "top":
        return val[1]
    if val[0] == "score":
        return Top(10, val[1])
    if val[0] == "query":                 # implicit E@: bare similar()/centroid() scores
        return Top(10, _to_score(val))
    if val[0] == "op":
        return Top(10, val[1])
    raise ParseError(f"query must be a score expression, scope-transformer, or top(...), got {val[0]}")
