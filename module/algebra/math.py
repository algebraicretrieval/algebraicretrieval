"""Translate mathematical notation into the existing Algebra parser and IR.

This is a notation seam, not another planner.  It normalizes mathematical
symbols, expands explicit symbol bindings, and delegates all semantics to the
existing parse.py implementation.  A bare expression remains a bare algebraic
term; selection is preserved only when the source explicitly requests it.
"""

from __future__ import annotations

import os
import re
import sys
from typing import Mapping

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import parse as _algebra_parse  # noqa: E402


class MathError(ValueError):
    pass


_SUBSCRIPT = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
_ASSIGNMENT = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\s*=(?!=)\s*(.+)$")
_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_ROOT_SELECTION = re.compile(r"^(?:top|earliest|latest)\s*\(")
_OPAQUE_CALLS = {"similar", "kw", "keyword", "type", "repo", "project", "eq", "raw", "mask", "weight"}
_LEFT_EXPR = set("@*+-&,(⊙▷⊕·×−")
_RIGHT_EXPR = set("@*+-&,)⊙▷⊕·×−")


def _split_statements(source: str) -> list[str]:
    """Split assignments/expressions only at top-level newlines or semicolons."""
    statements: list[str] = []
    current: list[str] = []
    quote: str | None = None
    escaped = False
    depth = 0
    for char in source:
        if quote is not None:
            current.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            continue
        if char in "'\"":
            quote = char
            current.append(char)
            continue
        if char in "([{":
            depth += 1
            current.append(char)
            continue
        if char in ")]}" and depth:
            depth -= 1
            current.append(char)
            continue
        if char in ";\n" and depth == 0:
            statement = "".join(current).strip()
            if statement and not statement.lstrip().startswith(("#", "--")):
                statements.append(statement)
            current = []
            continue
        current.append(" " if char == "\n" else char)
    if quote is not None:
        raise MathError("unterminated quoted text")
    statement = "".join(current).strip()
    if statement and not statement.lstrip().startswith(("#", "--")):
        statements.append(statement)
    return statements


def _matching_paren(expression: str, open_index: int) -> int | None:
    depth = 0
    quote: str | None = None
    escaped = False
    for index in range(open_index, len(expression)):
        char = expression[index]
        if quote is not None:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            continue
        if char in "'\"":
            quote = char
        elif char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return index
    return None


def _expression_position(expression: str, start: int, end: int) -> bool:
    left = start - 1
    while left >= 0 and expression[left].isspace():
        left -= 1
    right = end
    while right < len(expression) and expression[right].isspace():
        right += 1
    return (
        (left < 0 or expression[left] in _LEFT_EXPR)
        and (right >= len(expression) or expression[right] in _RIGHT_EXPR)
    )


def _substitute(expression: str, bindings: Mapping[str, str]) -> str:
    """Replace bound expression atoms, never words inside retrieval text."""
    if not bindings:
        return expression

    out: list[str] = []
    i = 0
    quote: str | None = None
    escaped = False
    while i < len(expression):
        char = expression[i]
        if quote is not None:
            out.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            i += 1
            continue
        if char in "'\"":
            quote = char
            out.append(char)
            i += 1
            continue
        match = _IDENTIFIER.match(expression, i)
        if match:
            name = match.group(0)
            look = match.end()
            while look < len(expression) and expression[look].isspace():
                look += 1
            if name.lower() in _OPAQUE_CALLS and look < len(expression) and expression[look] == "(":
                close = _matching_paren(expression, look)
                if close is None:
                    raise MathError(f"unbalanced call after {name}(")
                out.append(expression[i:close + 1])
                i = close + 1
                continue
            replacement = bindings.get(name)
            if replacement is not None and _expression_position(expression, i, match.end()):
                out.append(f"({replacement})")
            else:
                out.append(name)
            i = match.end()
            continue
        out.append(char)
        i += 1
    return "".join(out)


def _normalize_unquoted(expression: str) -> str:
    expression = expression.translate(_SUBSCRIPT)
    expression = expression.replace("−", "-").replace("·", "*").replace("×", "*")
    expression = expression.replace("∩", "&")
    expression = re.sub(
        r"⊕\s*(?:_?\{?RRF\}?|_RRF)\b",
        "⊕",
        expression,
        flags=re.IGNORECASE,
    )
    if "∪" in expression or "¬" in expression:
        raise MathError("mask union/complement is not represented by the existing Algebra IR")
    return re.sub(
        r"τ\s*(?:_?\{?(\d+)\}?|\[(\d+)\])\s*\(",
        lambda match: f"top({match.group(1) or match.group(2)}, ",
        expression,
    )


def _normalize_operators(expression: str) -> str:
    """Normalize only outside quoted retrieval text."""
    out: list[str] = []
    current: list[str] = []
    quote: str | None = None
    escaped = False
    for char in expression:
        if quote is not None:
            out.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            continue
        if char in "'\"":
            if current:
                out.append(_normalize_unquoted("".join(current)))
                current = []
            quote = char
            out.append(char)
        else:
            current.append(char)
    if current:
        out.append(_normalize_unquoted("".join(current)))
    return "".join(out)


def _validate_assignment(name: str, expression: str) -> None:
    """Fail early on forward/self references or malformed assignment values."""
    try:
        parser = _algebra_parse._P(_algebra_parse._tokenize(expression))
        parser.expr(0)
        if parser.peek() is not None:
            raise _algebra_parse.ParseError(f"unexpected trailing token {parser.peek()}")
    except _algebra_parse.ParseError as error:
        raise MathError(f"invalid assignment {name}: {error}") from error


def transpile(source: str, bindings: Mapping[str, str] | None = None) -> str:
    """Normalize a mathematical expression to the existing Algebra syntax.

    A small sequence of named assignments is accepted for readability; the last
    expression (or last assignment value) is the returned query term.
    """
    env = dict(bindings or {})
    statements = _split_statements(source)
    if not statements:
        raise MathError("empty algebraic expression")

    current = ""
    for statement in statements:
        assignment = _ASSIGNMENT.match(statement)
        if assignment:
            name, expression = assignment.groups()
            if name in env:
                raise MathError(f"assignment {name!r} cannot rebind an existing name")
            current = _normalize_operators(_substitute(expression, env))
            _validate_assignment(name, current)
            env[name] = current
        else:
            current = _normalize_operators(_substitute(statement, env))
    return current


def parse_math(source: str, bindings: Mapping[str, str] | None = None):
    """Return existing Algebra IR for a mathematical expression.

    Bare expressions return the underlying term rather than parse.py's implicit
    Top(10) delivery wrapper. An explicit τ/top/earliest/latest remains a Top.
    """
    expression = transpile(source, bindings)
    parsed = _algebra_parse.parse(expression)
    if _ROOT_SELECTION.match(expression.lstrip()):
        return parsed
    return parsed.expr
