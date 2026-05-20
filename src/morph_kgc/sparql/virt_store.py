__author__ = "Julián Arenas-Guerrero"
__credits__ = ["Julián Arenas-Guerrero"]

__license__ = "Apache-2.0"
__maintainer__ = "Julián Arenas-Guerrero"
__email__ = "julian.arenas.guerrero@upm.es"

# ──────────────────────────────────────────────────────────────────────────────
# Standard library
# ──────────────────────────────────────────────────────────────────────────────
import hashlib
import re
import warnings
import math
from collections import defaultdict
from functools import lru_cache
from typing import Any, Iterable, Iterator, List, Optional, Tuple

# ──────────────────────────────────────────────────────────────────────────────
# Third-party
# ──────────────────────────────────────────────────────────────────────────────
import pandas as pd
from urllib.parse import unquote as _url_unquote
from rdflib import BNode, Literal, URIRef
from rdflib.namespace import XSD
from rdflib.plugins.sparql import CUSTOM_EVALS
from rdflib.plugins.sparql.evalutils import _ebv, _eval
from rdflib.plugins.sparql.sparql import CompValue, FrozenBindings, QueryContext, SPARQLError
from rdflib.store import Store
from rdflib.term import Identifier, Variable

# ──────────────────────────────────────────────────────────────────────────────
# Internal
# ──────────────────────────────────────────────────────────────────────────────
from ..args_parser import load_config_from_argument
from ..mapping.mapping_parser import retrieve_mappings
from ..materializer import _materialize_mapping_group_to_df
from .types import BGP, Triple

warnings.simplefilter(action="ignore", category=pd.errors.SettingWithCopyWarning)

# ──────────────────────────────────────────────────────────────────────────────
# Type aliases
# ──────────────────────────────────────────────────────────────────────────────
RDFTerm = URIRef | Literal | BNode | Variable
TriplePattern = tuple[RDFTerm, RDFTerm, RDFTerm]

# ──────────────────────────────────────────────────────────────────────────────
# RML vocabulary constants
# ──────────────────────────────────────────────────────────────────────────────
RML_TEMPLATE   = "http://w3id.org/rml/template"
RML_REFERENCE  = "http://w3id.org/rml/reference"
RML_CONSTANT   = "http://w3id.org/rml/constant"
RML_QUERY      = "http://w3id.org/rml/query"
RML_TABLENAME  = "http://w3id.org/rml/tableName"
RML_IRI              = "http://w3id.org/rml/IRI"
RML_LITERAL          = "http://w3id.org/rml/Literal"
RML_BLANK_NODE       = "http://w3id.org/rml/BlankNode"
RML_PARENT_TRIPLES_MAP = "http://w3id.org/rml/parentTriplesMap"

# Well-known low-selectivity predicates — extend as needed for your dataset
LOW_SELECTIVITY_PREDICATES: frozenset[str] = frozenset({
    "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
    "http://www.w3.org/2000/01/rdf-schema#label",
    "http://www.w3.org/2000/01/rdf-schema#subClassOf",
    "http://www.w3.org/2000/01/rdf-schema#subPropertyOf",
    "http://www.w3.org/2000/01/rdf-schema#domain",
    "http://www.w3.org/2000/01/rdf-schema#range",
    "http://www.w3.org/2002/07/owl#sameAs",
    "http://www.w3.org/2004/02/skos/core#broader",
    "http://www.w3.org/2004/02/skos/core#narrower",
})

# XSD boolean: RDFLib normalises "1"→"true" and "0"→"false" on Literal
# construction. Raw relational sources (e.g. GTFS) commonly store "1"/"0",
# so we expand both canonical and alternative forms in SQL IN-clauses.
_XSD_BOOLEAN          = str(XSD.boolean)
_XSD_DATE             = str(XSD.date)
_XSD_DATETIME         = str(XSD.dateTime)
_BOOLEAN_TRUE_VALUES  = frozenset({"true", "1"})
_BOOLEAN_FALSE_VALUES = frozenset({"false", "0"})

# A12 — Maximum number of values per SQL IN clause.
# SQLite raises SQLITE_LIMIT_VARIABLE_NUMBER (default 999) for larger lists;
# conservative limit of 500 gives headroom for compound WHERE clauses.
_SQL_IN_CHUNK_SIZE: int = 500

# ── B11 — Bloom Filter Pre-join Probe ─────────────────────────────────────────
# When the intermediate bindings_df grows beyond _B11_THRESHOLD rows, a Bloom
# filter is built from the join-key values and used to probe the materialised
# candidate rows of the next triple pattern BEFORE natural_join is called.
# This eliminates non-matching rows at O(n) bitarray cost rather than incurring
# the full O(n·m) merge cost for rows that the join would discard anyway.
#
# Parameters:
#   _B11_THRESHOLD  — minimum bindings_df rows to activate B11 (default 64).
#                     Below this, direct pandas merge is cheaper than BF overhead.
#   _B11_BIT_SIZE   — bit-array length in bits. Must be a power of 2 for fast
#                     modulo via bitmask. 2^17 = 131,072 bits ≈ 16 KB; fits in
#                     L1 cache for most modern CPUs.
#   _B11_HASH_COUNT — number of independent hash functions (k). Optimal k for
#                     a given (m, n) is k = (m/n) · ln(2). For m=131072 and
#                     n≤8192, k=4 gives false-positive rate < 0.01%.
_B11_THRESHOLD:  int = 64
_B11_BIT_SIZE:   int = 1 << 17      # 131,072 bits
_B11_HASH_COUNT: int = 4

# Set env var VIRT_DEBUG=1 to enable per-step SQL and row-count tracing in
# virt_eval_bgp. Useful for diagnosing empty-result issues with FILTER pushdown.
import os as _os
_VIRT_DEBUG: bool = _os.environ.get("VIRT_DEBUG", "0") == "1"


# ── B11 — Bloom filter helpers ────────────────────────────────────────────────

def _bloom_build(values: "Iterable[str]") -> bytearray:
    """
    Build a Bloom filter bit-array (as a ``bytearray``) from an iterable of
    string *values*.

    Uses ``hashlib.md5`` seeded with *k* integer prefixes as *k* independent
    hash functions.  All arithmetic uses stdlib only; no third-party dependency.

    Returns a ``bytearray`` of length ``_B11_BIT_SIZE // 8`` with the
    appropriate bits set.  The bit-array size is a power of two so that the
    modulo operation reduces to a fast bitwise AND.
    """
    bits = bytearray(_B11_BIT_SIZE >> 3)
    for v in values:
        b = v.encode()
        for seed in range(_B11_HASH_COUNT):
            h = int(hashlib.md5(seed.to_bytes(2, "little") + b).hexdigest(), 16)
            pos = h & (_B11_BIT_SIZE - 1)
            bits[pos >> 3] |= 1 << (pos & 7)
    return bits


def _bloom_probe(bits: bytearray, value: str) -> bool:
    """
    Return ``True`` if *value* **may** be in the set (possible false positive),
    ``False`` if *value* is **definitely not** in the set (guaranteed no false
    negatives).
    """
    b = value.encode()
    for seed in range(_B11_HASH_COUNT):
        h = int(hashlib.md5(seed.to_bytes(2, "little") + b).hexdigest(), 16)
        pos = h & (_B11_BIT_SIZE - 1)
        if not (bits[pos >> 3] & (1 << (pos & 7))):
            return False
    return True


def _bloom_filter_df(df: pd.DataFrame, col: str, bits: bytearray) -> pd.DataFrame:
    """
    Return the subset of *df* whose *col* values pass the Bloom filter probe.

    Uses ``Series.map`` for a vectorised Python-level scan — faster than
    ``apply`` because ``map`` avoids per-call frame overhead on the lambda.
    The result is a filtered view; the original DataFrame is not mutated.
    """
    # Guard against duplicate column labels introduced by multi-rule concat.
    # Use keep="last" to match rename_triple_columns dedup strategy: the
    # correctly-typed RDF term column always appears after the raw source column.
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated(keep="last")]
    mask = df[col].map(lambda v: _bloom_probe(bits, str(v)))
    return df[mask]


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — General utilities
# ══════════════════════════════════════════════════════════════════════════════

def is_integer_string(s: str) -> bool:
    """Return True if *s* is the string representation of a signed integer."""
    try:
        int(s.strip())
        return True
    except (ValueError, TypeError):
        return False


def keep_integer_strings_or_all(items: List[Any]) -> List[Any]:
    """
    Return only integer-string elements when the non-integer elements are
    all boolean aliases; otherwise return *items* unchanged.

    Two problems drove this design:

    1. **Boolean SQL type mismatch** — ``xsd:boolean`` literals produce
       candidate values ``["1", "true"]`` (or ``["0", "false"]``).
       Injecting ``'true'`` into a strictly-typed SQL INTEGER column raises
       ``invalid input syntax for type integer: "true"``.  When every
       non-integer value is a known boolean alias, keeping only the integer
       form (``"1"`` / ``"0"``) is safe and avoids the error.

    2. **Alphanumeric identifier loss** — stop IDs such as
       ``"000000000000000000ls"`` are not integer strings but are perfectly
       valid SQL VARCHAR values.  The old unconditional integer-only filter
       silently dropped them, so those stops never appeared in the
       ``WHERE … IN (…)`` clause and were lost from results.

    Rule: filter to integer strings **only** when every non-integer item is
    a member of the boolean alias sets ``{"true","false"}``; for all other
    mixed lists (e.g. alphanumeric identifiers alongside numeric ones) return
    the full list so no valid value is discarded.
    """
    _BOOL_ALIASES = frozenset({"true", "false"})
    integer_items = [x for x in items if isinstance(x, str) and is_integer_string(x)]
    if not integer_items:
        return list(items)
    non_integer_items = [x for x in items if not (isinstance(x, str) and is_integer_string(x))]
    # Only filter when every non-integer value is a boolean alias.
    # Alphanumeric IDs that happen to share a list with numeric IDs must be kept.
    if all(str(x).lower() in _BOOL_ALIASES for x in non_integer_items):
        return integer_items
    return list(items)


def to_rdf_term_typed(value: Any, termtype: str) -> URIRef | Literal | BNode:
    """
    Convert a Morph-KGC materialised DataFrame cell into a canonical RDFLib term.

    Morph-KGC returns proprietary subclasses such as ``LenientURIRef`` whose
    ``__eq__`` / ``eq()`` break RDFLib's SPARQL ``=`` operator (dispatched as
    ``x.eq(y)`` → ``x.__eq__(y)``).  Every branch casts to the exact base class
    to guarantee correct FILTER / ORDER BY / equality evaluation.
    """
    # ── RDFLib term subclasses: cast to canonical base class ─────────────────
    if isinstance(value, Literal):
        dt = str(value.datatype) if value.datatype else None
        lex = str(value)
        if dt in (_XSD_DATE, _XSD_DATETIME):
            lex = _sanitise_date_lexical(lex, dt)
        return Literal(lex, lang=value.language, datatype=value.datatype)
    if isinstance(value, BNode):
        return BNode(str(value))
    if isinstance(value, URIRef):          # catches LenientURIRef and any subclass
        return URIRef(str(value))
    # ── Plain string fallback ─────────────────────────────────────────────────
    s = str(value).strip()
    if termtype == RML_IRI:
        return URIRef(s[1:-1] if s.startswith("<") and s.endswith(">") else s)
    if termtype == RML_BLANK_NODE:
        return BNode(s[2:] if s.startswith("_:") else s)
    return Literal(s)


def _normalise_iri_series(series: pd.Series) -> pd.Series:
    """
    OPT-9 — Vectorised IRI column normalisation.

    Morph-KGC returns IRI columns as ``LenientURIRef`` subclasses.  Because all
    cells in a materialized IRI column share the same runtime type, we can
    branch once on the first element and apply a bulk string conversion via
    ``pd.Series.map``, which runs in pandas' C layer rather than calling a
    Python function per cell.
    """
    if series.empty:
        return series
    first = series.iloc[0]
    if isinstance(first, URIRef):
        # All cells are URIRef (or subclass): bulk-cast via str then URIRef.
        # Series.map(str) and Series.map(URIRef) both operate at C speed.
        return series.map(str).map(URIRef)
    # Fallback for plain strings (older Morph-KGC versions)
    return series.map(
        lambda v: URIRef(str(v)[1:-1]) if str(v).startswith("<") else URIRef(str(v))
    )


def _sanitise_date_lexical(lex: str, datatype_str: str) -> str:
    """
    Coerce a date/dateTime lexical form returned by a relational DB into the
    canonical form expected by rdflib / xsd:date or xsd:dateTime.

    Common problems fixed:

    * ``xsd:date`` columns return a full datetime string from the DB driver,
      e.g. ``"2017-01-19 00:00:00"`` — strip everything after the first space
      or ``T`` separator → ``"2017-01-19"``.
    * Compact ``YYYYMMDD`` form (GTFS feed_info) — insert dashes →
      ``"20170119"`` → ``"2017-01-19"``.
    * ``xsd:dateTime`` with space separator instead of ``T`` —
      replace the space with ``T`` → ``"2017-01-19T00:00:00"``.
    """
    if datatype_str == _XSD_DATE:
        # Strip time component if present
        for sep in ('T', ' '):
            if sep in lex:
                lex = lex.split(sep)[0]
        # Compact YYYYMMDD → YYYY-MM-DD
        if len(lex) == 8 and lex.isdigit():
            lex = f"{lex[:4]}-{lex[4:6]}-{lex[6:]}"
    elif datatype_str == _XSD_DATETIME:
        # Normalise space separator to T
        if ' ' in lex and 'T' not in lex:
            lex = lex.replace(' ', 'T', 1)
    return lex


def _normalise_literal_series(series: pd.Series) -> pd.Series:
    """
    OPT-9 — Vectorised Literal column normalisation.

    Literal columns require preserving datatype and language tag, so a full
    per-cell call is unavoidable.  However, we avoid the overhead of repeated
    ``isinstance`` checks inside ``to_rdf_term_typed`` by branching once on the
    first element and using a specialised lambda.

    Date/dateTime literals have their lexical form sanitised via
    ``_sanitise_date_lexical`` so that rdflib can parse and compare them
    correctly in FILTER expressions.
    """
    if series.empty:
        return series
    first = series.iloc[0]
    if isinstance(first, Literal):
        def _fix(v: Literal) -> Literal:
            dt = str(v.datatype) if v.datatype else None
            lex = str(v)
            if dt in (_XSD_DATE, _XSD_DATETIME):
                lex = _sanitise_date_lexical(lex, dt)
            return Literal(lex, lang=v.language, datatype=v.datatype)
        return series.apply(_fix)
    # Fallback: raw Python value from DB driver (datetime, date, str, numeric…)
    # Wrap as a properly typed Literal rather than an untyped string Literal,
    # so that FILTER comparisons involving xsd:date / xsd:dateTime work correctly.
    import datetime as _dt

    def _wrap_raw(v: Any) -> Literal:
        if isinstance(v, _dt.datetime):
            lex = _sanitise_date_lexical(v.isoformat(), _XSD_DATETIME)
            return Literal(lex, datatype=XSD.dateTime)
        if isinstance(v, _dt.date):
            return Literal(v.isoformat(), datatype=XSD.date)
        return Literal(str(v))

    return series.apply(_wrap_raw)


def apply_termtypes_to_df(df: pd.DataFrame, rml_tp_df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise the ``subject``, ``predicate``, and ``object`` columns of *df*
    to canonical RDFLib term objects.

    Must be called immediately after ``_materialize_mapping_group_to_df`` and
    before ``rename_triple_columns`` so that ``FrozenBindings`` always contains
    proper ``URIRef`` / ``Literal`` / ``BNode`` instances — never proprietary
    subclasses (e.g. ``LenientURIRef``) whose ``__eq__`` breaks SPARQL FILTER.

    OPT-9: IRI columns are normalised via a bulk ``Series.map(str).map(URIRef)``
    pipeline (C-layer speed) rather than a per-cell ``apply(to_rdf_term_typed)``.
    OPT-11: The DataFrame is only copied once, immediately before mutation,
    rather than defensively at the start of the function regardless of whether
    any column actually needs changing.

    Multi-rule safety: ``rml_tp_df`` may contain rows from multiple mapping
    rules with different ``object_termtype`` values (e.g. one rule maps the
    object via ``rml:template`` → IRI; another via ``rml:reference`` →
    Literal).  Using only ``iloc[0]`` would silently mis-normalise rows
    originating from the other rules.  Instead:

    - ``subject`` and ``predicate`` are always ``rml:IRI`` by the RML spec.
    - ``object`` termtype is determined by inspecting all rules: if all rules
      agree, that termtype is used; if they differ, a per-cell heuristic
      normaliser is applied as a safe fallback.
    """
    if rml_tp_df.empty or df.empty:
        return df

    # subject and predicate are always IRI per RML spec — no need to inspect rules.
    col_termtype: dict[str, str | None] = {
        "subject":   RML_IRI,
        "predicate": RML_IRI,
    }

    # Object termtype: derive from the set of distinct values across all rules.
    # NULL/NaN entries (referencing object maps) are treated as IRI.
    obj_types = (
        rml_tp_df["object_termtype"]
        .fillna(RML_IRI)
        .unique()
        .tolist()
    )
    if len(obj_types) == 1:
        col_termtype["object"] = obj_types[0]
    else:
        # Mixed termtypes across rules — use None to signal per-cell fallback.
        col_termtype["object"] = None

    # OPT-11 — copy-on-write: allocate only once, only when a column is present.
    copied = False
    for col, termtype in col_termtype.items():
        if col not in df.columns:
            continue
        if not copied:
            df = df.copy()
            copied = True
        if termtype == RML_IRI:
            # OPT-9 — bulk IRI normalisation via C-layer map pipeline.
            df[col] = _normalise_iri_series(df[col])
        elif termtype == RML_LITERAL:
            df[col] = _normalise_literal_series(df[col])
        elif termtype == RML_BLANK_NODE:
            df[col] = df[col].apply(lambda v: to_rdf_term_typed(v, RML_BLANK_NODE))
        else:
            # None → mixed termtypes: apply a safe per-cell heuristic.
            # Strings that look like IRIs (start with a scheme) become URIRef;
            # everything else becomes a plain Literal.  This is conservative:
            # the Python-level SPARQL post-filter always validates final values.
            df[col] = df[col].apply(
                lambda v: (
                    URIRef(str(v))
                    if str(v).startswith(("http://", "https://", "urn:"))
                    else Literal(v)
                )
            )
    return df


def literal_to_source_values(lit: Literal) -> list[str]:
    """
    Return the raw source string values that could represent *lit* in a
    relational database column.

    RDFLib normalises certain XSD datatypes on construction. The most
    important case is ``xsd:boolean``: ``Literal("1", datatype=xsd:boolean)``
    is stored internally as the lexical form ``"true"``, but a GTFS source
    column stores ``"1"`` or ``"0"``. This function returns both forms so
    that the generated SQL ``IN`` clause matches regardless of which
    representation the source uses.

    For all other datatypes, ``str(lit)`` gives the correct lexical form and
    a single-element list is returned.
    """
    if lit.datatype and str(lit.datatype) == _XSD_BOOLEAN:
        return sorted(
            _BOOLEAN_TRUE_VALUES if str(lit).lower() in ("true", "1")
            else _BOOLEAN_FALSE_VALUES
        )
    return [str(lit)]


def triple_pattern_variables(triple_pattern: TriplePattern) -> list[str]:
    """
    Return the ordered, deduplicated variable names in *triple_pattern*
    (subject → predicate → object order).
    """
    seen: set[str] = set()
    variables: list[str] = []
    for term in triple_pattern:
        if isinstance(term, Variable):
            name = str(term)
            if name not in seen:
                seen.add(name)
                variables.append(name)
    return variables


def bgp_variables(bgp: list[TriplePattern]) -> list[str]:
    """Return ordered, deduplicated variable names across all triples in *bgp*."""
    seen: set[str] = set()
    variables: list[str] = []
    for triple in bgp:
        for var in triple_pattern_variables(triple):
            if var not in seen:
                seen.add(var)
                variables.append(var)
    return variables


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — DataFrame join helpers
# ══════════════════════════════════════════════════════════════════════════════

def natural_join(
    left: pd.DataFrame,
    right: pd.DataFrame,
    tp: TriplePattern,
    on: list[str] | None = None,
    how: str = "inner",
) -> pd.DataFrame:
    """
    Merge *left* and *right* on their shared variable columns.

    Parameters
    ----------
    left, right:
        DataFrames whose columns are variable names.
    tp:
        The triple pattern whose variable names determine join columns when
        *on* is not provided explicitly.
    on:
        Explicit list of join columns. If ``None``, auto-detects shared
        columns that also appear as variables in *tp*. Pass ``[]`` for a
        Cartesian product.
    how:
        Pandas merge strategy (default ``"inner"``).

    OPT-10 — String-keyed merge: join key columns contain RDFLib term objects
    (``URIRef``, ``Literal``).  pandas' merge operates on Python object
    equality for object-dtype columns, bypassing the optimised C hash-join
    path.  Converting join keys to their string representations before the
    merge and restoring typed values afterward delegates the heavy hash-join
    work to pandas' C layer, while the final typed values are recovered from
    the left DataFrame (which always has the canonical typed values after
    ``apply_termtypes_to_df``).
    """
    if on is None:
        on = list(
            set(left.columns) & set(right.columns) & set(triple_pattern_variables(tp))
        )
    if not on:
        return pd.merge(left, right, how="cross")

    right_extra = [c for c in right.columns if c not in left.columns]
    right_cols  = on + right_extra

    # OPT-10 — convert join keys to str for C-layer hash-join, then restore
    # typed values from left (which holds the canonical RDFLib term objects).
    STR_SUFFIX = "__str_key__"
    str_on = [c + STR_SUFFIX for c in on]

    left_work  = left.assign(**{c + STR_SUFFIX: left[c].map(str)  for c in on})
    right_work = right[right_cols].assign(**{c + STR_SUFFIX: right[c].map(str) for c in on})

    if _VIRT_DEBUG:
        for c, sc in zip(on, str_on):
            lsample = left_work[sc].dropna().unique()[:3].tolist()
            rsample = right_work[sc].dropna().unique()[:3].tolist()
            print(f"[VIRT_DEBUG] natural_join key={c!r}: "
                  f"left samples={lsample} right samples={rsample}")
        print(f"[VIRT_DEBUG] natural_join: left={len(left)} rows, right={len(right)} rows")

    merged = pd.merge(
        left_work,
        right_work.drop(columns=on),   # drop original typed keys from right
        on=str_on,
        how=how,
    ).drop(columns=str_on)             # drop the temporary string key columns

    if _VIRT_DEBUG:
        print(f"[VIRT_DEBUG] natural_join: merged={len(merged)} rows")

    return merged


def rename_triple_columns(
    df: pd.DataFrame,
    triple_pattern: TriplePattern,
) -> pd.DataFrame:
    """
    Restrict *df* to only the ``subject``, ``predicate``, ``object`` columns
    and rename each to its corresponding variable name in *triple_pattern*.
    Columns whose position holds a concrete term (URIRef/Literal/BNode) are
    dropped.

    Keeping only these three columns is essential: morph-KGC includes raw
    source columns (e.g. ``date`` from ``calendar_dates.date``) that share
    names with SPARQL variables after renaming.  Retaining those raw columns
    causes ``FrozenBindings`` to receive untyped Python strings instead of
    properly-typed RDF Literals, breaking FILTER comparisons.
    """
    if len(triple_pattern) != 3:
        raise ValueError("triple_pattern must be a 3-tuple (s, p, o)")

    # Restrict to only the three core RDF columns.
    core_cols = [c for c in ("subject", "predicate", "object") if c in df.columns]
    df = df[core_cols]

    s, p, o = triple_pattern
    rename_map: dict[str, str] = {}
    cols_to_drop: list[str] = []

    for col, term in (("subject", s), ("predicate", p), ("object", o)):
        if col not in df.columns:
            continue
        if isinstance(term, Variable):
            rename_map[col] = str(term)
        else:
            cols_to_drop.append(col)

    result = df.rename(columns=rename_map)
    if cols_to_drop:
        result = result.drop(columns=cols_to_drop)
    return result


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — RML template / term matching
# ══════════════════════════════════════════════════════════════════════════════



def match_rml_template(rdf_term: RDFTerm, template: str) -> dict[str, str] | None:
    """
    Attempt to reverse-match *rdf_term* against an RML template.

    Returns
    -------
    dict[str, str]
        Mapping ``{reference_name: matched_value}`` on success.
    None
        If the term cannot have been produced by *template*.
    """
    pattern, references = rml_template_to_regex(template)
    m = pattern.match(str(rdf_term))
    if m is None:
        return None

    groups = m.groupdict()
    ref_count: dict[str, int] = {}
    result: dict[str, str] = {}
    for ref_name in references:
        safe = re.sub(r"\W", "_", ref_name)
        count = ref_count.get(safe, 0)
        group = safe if count == 0 else f"{safe}_{count}"
        ref_count[safe] = count + 1
        key = ref_name if ref_name not in result else f"{ref_name}#{count}"
        result[key] = groups[group]
    return result


def _position_mask(
    rml_df: pd.DataFrame,
    pat: RDFTerm,
    map_type_col: str,
    map_value_col: str,
) -> "pd.Series[bool]":
    """
    OPT-4 — Return a boolean Series marking which rules in *rml_df* are
    compatible with *pat* at the given triple-pattern position.

    Replaces the per-row Python loop + nested ``def _term_matches`` in the
    original ``match_triple_pattern`` with vectorised pandas boolean masks.
    Each map_type branch is handled as a bulk column operation:

    - ``rml:variable``  — all rules pass (unconstrained position).
    - ``rml:reference``        — all rules pass; value constraint pushed to SQL.
    - ``rml:constant``         — string equality on the ``map_value`` column (C layer).
    - ``rml:template``         — per-row regex check, only for template rows.
    - ``rml:parentTriplesMap`` — all rules pass; object IRI is derived from the
                                 parent TM's subject template which is not available
                                 at this level. morph-kgc handles the join internally.
    """
    if isinstance(pat, Variable):
        # Unconstrained position: every rule is compatible.
        return pd.Series(True, index=rml_df.index)

    pat_str   = str(pat)
    map_types = rml_df[map_type_col].fillna("").astype(str)
    map_vals  = rml_df[map_value_col].fillna("").astype(str)

    is_const  = map_types == RML_CONSTANT
    is_tmpl   = map_types == RML_TEMPLATE
    is_ref    = map_types == RML_REFERENCE
    is_parent = map_types == RML_PARENT_TRIPLES_MAP

    # rml:constant — vectorised string equality (pandas C layer, no Python loop)
    const_ok = is_const & (map_vals == pat_str)

    # rml:reference — always compatible; pushdown_bindings_to_sql adds the WHERE
    ref_ok = is_ref

    # rml:parentTriplesMap — always compatible for concrete URIRef objects.
    # The object IRI is derived from the parent TM's subject template; we cannot
    # push a WHERE condition at this level (the join column mapping is in the
    # parent TM). The rule is kept and morph-kgc materialises via its join path;
    # the bind-join on the subject position filters to the correct rows.
    parent_ok = is_parent

    # rml:template — apply regex only to the template subset to minimise overhead
    if is_tmpl.any():
        tmpl_ok = is_tmpl & map_vals.apply(
            lambda tmpl: match_rml_template(pat, tmpl) is not None
        )
    else:
        tmpl_ok = pd.Series(False, index=rml_df.index)

    return const_ok | ref_ok | parent_ok | tmpl_ok


def match_triple_pattern(tp: TriplePattern, rml_df: pd.DataFrame) -> pd.DataFrame:
    """
    Return the subset of mapping rules in *rml_df* that are compatible with
    the triple pattern *tp*.

    A mapping rule is compatible when each concrete term in the pattern is
    consistent with the rule's map type:

    - ``rml:constant``  — the constant value must equal the pattern term.
    - ``rml:template``  — the term's string form must match the template regex.
    - ``rml:reference`` — always compatible for concrete terms; the equality
      condition is injected as a SQL WHERE clause by
      ``pushdown_bindings_to_sql``.
    - Variable          — always compatible (unconstrained).

    OPT-4: The original row-by-row ``iterrows`` loop with a ``def`` re-created
    on every iteration has been replaced by ``_position_mask``, which uses
    vectorised boolean masks for constant and reference checks (handled in
    pandas' C layer) and restricts the Python-level regex loop to template rows
    only.  Copy-on-write (OPT-11) is also applied: the DataFrame is only copied
    when a mutation is actually required.
    """
    s_pat, p_pat, o_pat = tp

    # ── Stage 1: fast termtype pre-filter (vectorised column equality) ────────
    # This eliminates structurally incompatible rules before any value-level
    # work, reducing the template-matching workload in _position_mask.
    #
    # NULL/NaN termtype rows are always kept: morph-kgc does not populate
    # object_termtype for referencing object maps (rml:parentTriplesMap), which
    # produce IRI objects derived from the parent triples map's subject template.
    # Dropping NULL rows would silently prune all join-based object mappings.
    if _VIRT_DEBUG:
        print(f"[VIRT_DEBUG] match_triple_pattern: tp={tp}, {len(rml_df)} rules before Stage 1")
        if not rml_df.empty:
            preds = rml_df["predicate_map_value"].unique().tolist()
            obj_tt = rml_df["object_termtype"].unique().tolist()
            obj_mt = rml_df["object_map_type"].unique().tolist()
            print(f"  predicate_map_values={preds[:5]}")
            print(f"  object_termtype values={obj_tt}")
            print(f"  object_map_type values={obj_mt}")
    if not isinstance(s_pat, Variable):
        if isinstance(s_pat, URIRef):
            s_tt = rml_df["subject_termtype"].fillna(RML_IRI)
            rml_df = rml_df[s_tt == RML_IRI]
        elif isinstance(s_pat, BNode):
            s_tt = rml_df["subject_termtype"].fillna(RML_IRI)
            rml_df = rml_df[s_tt == RML_BLANK_NODE]

    if not isinstance(o_pat, Variable):
        if isinstance(o_pat, URIRef):
            o_tt = rml_df["object_termtype"].fillna(RML_IRI)
            if _VIRT_DEBUG:
                uniq = o_tt.unique().tolist()
                print(f"[VIRT_DEBUG] object URIRef filter: o_tt values={uniq}, RML_IRI={RML_IRI!r}")
            rml_df = rml_df[o_tt == RML_IRI]
            if _VIRT_DEBUG:
                print(f"[VIRT_DEBUG] after object URIRef filter: {len(rml_df)} rules remain")
        elif isinstance(o_pat, Literal):
            o_tt = rml_df["object_termtype"].fillna(RML_IRI)
            rml_df = rml_df[o_tt == RML_LITERAL]
        elif isinstance(o_pat, BNode):
            o_tt = rml_df["object_termtype"].fillna(RML_IRI)
            rml_df = rml_df[o_tt == RML_BLANK_NODE]

    if rml_df.empty:
        return rml_df

    # ── Stage 2: OPT-4 vectorised value-level compatibility mask ─────────────
    s_mask = _position_mask(rml_df, s_pat, "subject_map_type",   "subject_map_value")
    p_mask = _position_mask(rml_df, p_pat, "predicate_map_type", "predicate_map_value")
    o_mask = _position_mask(rml_df, o_pat, "object_map_type",    "object_map_value")

    return rml_df[s_mask & p_mask & o_mask]


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — BGP ordering
# ══════════════════════════════════════════════════════════════════════════════

def _position_weight(pos: int) -> int:
    return {0: 1, 2: 2, 1: 3}[pos]


def _is_bound(term: RDFTerm, ctx: QueryContext, bound_vars: set) -> bool:
    if not isinstance(term, Variable):
        return True
    return term in bound_vars or ctx[term] is not None



def _triple_score(
    triple: TriplePattern,
    ctx: QueryContext,
    bound_vars: set,
    rml_df: pd.DataFrame,
) -> tuple[int, float]:
    """
    Cost-based heuristic score (lower = better).

    Structure:
        (connectivity, combined_cost)

    combined_cost is a weighted sum of:
        - mapping-aware selectivity (log-normalized)
        - number of unbound variables
        - predicate selectivity penalty
        - positional heuristic
    """

    s, p, o = triple

    # ── Hard constraint: connectivity ────────────────────────────────
    connected = any(
        isinstance(t, Variable) and _is_bound(t, ctx, bound_vars)
        for t in triple
    )
    l0 = 0 if (not bound_vars or connected) else 1

    # ── Mapping-aware selectivity (log-normalized) ───────────────────
    try:
        rml_tp_df = match_triple_pattern(triple, rml_df)
        rule_count = len(rml_tp_df)
    except Exception:
        rule_count = 1000  # fallback worst case

    # log normalization (avoids dominance)
    cost_mapping = math.log(rule_count + 1)

    # ── Unbound variables ────────────────────────────────────────────
    unbound_count = sum(
        1 for term in triple if not _is_bound(term, ctx, bound_vars)
    )

    # ── Predicate selectivity ────────────────────────────────────────
    predicate_penalty = 1.0 if (
        isinstance(p, URIRef) and str(p) in LOW_SELECTIVITY_PREDICATES
    ) else 0.0

    # ── Positional heuristic ─────────────────────────────────────────
    positional_weight = 0
    for pos, term in enumerate(triple):
        if not _is_bound(term, ctx, bound_vars):
            positional_weight += _position_weight(pos)

    # ── Weighted cost (tunable constants) ────────────────────────────
    α = 1.0
    β = 1.0
    γ = 0.5
    δ = 0.2

    combined_cost = (
        α * cost_mapping
        + β * unbound_count
        + γ * predicate_penalty
        + δ * positional_weight
    )

    return (l0, combined_cost)



def order_bgp(
    ctx: QueryContext,
    bgp: list[TriplePattern],
    rml_df: pd.DataFrame,
) -> list[TriplePattern]:
    """
    Cost-based BGP ordering with connectivity constraint.
    """

    if len(bgp) <= 1:
        return list(bgp)

    bound_vars: set[Variable] = {
        t for triple in bgp for t in triple
        if isinstance(t, Variable) and ctx[t] is not None
    }

    remaining = list(bgp)
    ordered: list[TriplePattern] = []

    while remaining:
        best = min(
            range(len(remaining)),
            key=lambda i: _triple_score(
                remaining[i], ctx, bound_vars, rml_df
            ),
        )
        chosen = remaining.pop(best)
        ordered.append(chosen)

        for t in chosen:
            if isinstance(t, Variable):
                bound_vars.add(t)

    return ordered



# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — SQL bind-join pushdown
# ══════════════════════════════════════════════════════════════════════════════

# A1 + OPT-5 — Single cached template compiler.
# ``rml_template_to_regex`` (Section 3) and ``_template_to_regex_with_names``
# (Section 5) were two independent copies of the same algorithm.  They are now
# unified here under a single name.  ``rml_template_to_regex`` below is kept as a
# public alias so that external callers and ``match_rml_template`` are unaffected.
# The @lru_cache means every distinct template string is compiled exactly once for
# the lifetime of the process, regardless of how many times it appears across
# Section 3 (matching) and Section 5 (pushdown) call paths.
@lru_cache(maxsize=256)
def _compile_rml_template(template: str) -> tuple[re.Pattern, list[str]]:
    """Compile an RML template string into a named-group regex and return the
    ordered list of reference names embedded in the template.

    Escaped braces (``\\{``, ``\\}``), repeated references, and references
    containing non-identifier characters are all handled correctly.
    """
    parts: list[str] = []
    references: list[str] = []
    ref_count: dict[str, int] = {}
    i = 0
    while i < len(template):
        ch = template[i]
        if ch == "\\" and i + 1 < len(template) and template[i + 1] in ("{", "}", "\\"):
            parts.append(re.escape(template[i + 1]))
            i += 2
        elif ch == "{":
            j = i + 1
            while j < len(template) and template[j] != "}":
                j += 1
            ref = template[i + 1 : j]
            references.append(ref)
            safe = re.sub(r"\W", "_", ref)
            count = ref_count.get(safe, 0)
            ref_count[safe] = count + 1
            group = safe if count == 0 else f"{safe}_{count}"
            parts.append(f"(?P<{group}>[^/{{}}]+)")
            i = j + 1
        else:
            parts.append(re.escape(ch))
            i += 1
    return re.compile("^" + "".join(parts) + "$"), references



# A1 — Public alias: ``rml_template_to_regex`` is the name used throughout
# Section 3 (``match_rml_template``) and by any external callers.  Placing
# the alias here, after the definition, guarantees the name is resolved at
# module load time without a NameError.
rml_template_to_regex = _compile_rml_template
def _extract_references_from_term(
    term: RDFTerm, map_type: str, map_value: str
) -> dict[str, list[str]] | None:
    """
    Reverse-engineer SQL column→value pairs from a bound RDF term.

    Returns a dict mapping each column reference to a list of candidate
    source values. Multiple values arise when a datatype has several
    equivalent raw representations (e.g. ``xsd:boolean`` → ``["1", "true"]``).

    Returns
    -------
    dict[str, list[str]]
        Column-to-values mapping for SQL pushdown.
    None
        The term is structurally incompatible with the mapping rule; the
        caller should prune the rule entirely.
    """
    if isinstance(term, Variable):
        return {}
    term_str = str(term)

    if map_type == RML_TEMPLATE:
        # OPT-14 — A template without "{" is effectively a constant IRI; skip
        # regex compilation entirely and fall through to a direct string compare.
        if "{" not in map_value:
            return {} if term_str == map_value else None
        pattern, refs = _compile_rml_template(map_value)
        m = pattern.match(term_str)
        if m is None:
            return None
        ref_count: dict[str, int] = {}
        result: dict[str, list[str]] = {}
        for ref in refs:
            safe = re.sub(r"\W", "_", ref)
            count = ref_count.get(safe, 0)
            ref_count[safe] = count + 1
            group = safe if count == 0 else f"{safe}_{count}"
            # RML templates percent-encode special characters in IRI values.
            # The source column holds the raw (decoded) value, so we must
            # URL-decode the extracted group before injecting it into SQL.
            # e.g. "2017-01-18%2000%3A00%3A00" → "2017-01-18 00:00:00"
            result[ref] = [_url_unquote(m.group(group))]
        return result

    if map_type == RML_REFERENCE:
        # Expand to all source-compatible string forms.
        # xsd:boolean "true" → ["1", "true"] to match both GTFS-style and
        # canonical representations stored in the source column.
        source_vals = (
            literal_to_source_values(term)
            if isinstance(term, Literal)
            else [term_str]
        )
        return {map_value: source_vals}

    if map_type == RML_CONSTANT:
        return {}

    return {}


def _wrap_existing_query(sql: str) -> str:
    """Wrap *sql* as a subquery so a WHERE clause can safely be appended."""
    return f"SELECT * FROM ({sql.strip().rstrip(';')}) AS _subquery"


# ── B5 — FILTER pushdown helpers ─────────────────────────────────────────────

# Operator map: SPARQL relational operator → SQL operator (all 1-to-1)
_RELATIONAL_OP_MAP: dict[str, str] = {
    "=": "=", "!=": "!=", ">": ">", "<": "<", ">=": ">=", "<=": "<=",
}

# XSD numeric types safe to emit as unquoted SQL tokens (no quoting needed)
_NUMERIC_XSD_TYPES: frozenset[str] = frozenset({
    str(XSD.integer), str(XSD.int),     str(XSD.long),       str(XSD.short),
    str(XSD.byte),    str(XSD.decimal), str(XSD.float),       str(XSD.double),
    str(XSD.unsignedInt), str(XSD.unsignedLong),
    str(XSD.unsignedShort), str(XSD.unsignedByte),
})

# XSD temporal types — SQL pushdown is skipped for these entirely.
# Pushing a temporal literal comparison into SQL is unsafe when the target
# column type is unknown (e.g. INTEGER exception_type vs DATE value).
# The Python post-filter via _eval_filter_expr always handles these correctly.
_TEMPORAL_XSD_TYPES: frozenset[str] = frozenset({
    str(XSD.date),
    str(XSD.dateTime),
    str(XSD.dateTimeStamp),
    str(XSD.time),
    str(XSD.gYear),
    str(XSD.gYearMonth),
    str(XSD.duration),
    str(XSD.dayTimeDuration),
    str(XSD.yearMonthDuration),
})


def _sql_escape(value: str) -> str:
    """Escape single quotes inside a SQL string literal (SQL-99 standard)."""
    return value.replace("'", "''")


def _literal_to_sql_value(lit: Literal) -> str:
    """
    Render an rdflib Literal as a SQL value token.

    - XSD numeric types → unquoted numeric literal (safe for arithmetic)
    - xsd:boolean       → unquoted 1 / 0 (ANSI SQL compatible)
    - Everything else   → single-quoted escaped string

    Temporal literals (xsd:date, xsd:dateTime, etc.) are never passed here
    during SQL pushdown — ``filter_expr_to_sql`` returns ``None`` for any
    Literal whose datatype is in ``_TEMPORAL_XSD_TYPES``, causing the caller
    to skip SQL injection and fall back to the Python post-filter.
    """
    dt = str(lit.datatype) if lit.datatype else ""
    s  = str(lit)
    if dt in _NUMERIC_XSD_TYPES:
        if re.fullmatch(r"-?[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?", s):
            return s
    if dt == _XSD_BOOLEAN:
        return "1" if s.lower() in ("true", "1") else "0"
    return f"'{_sql_escape(s)}'"


def _extract_equalities(
    expr: Any,
) -> tuple[dict[Variable, Any], Any | None]:
    """
    Decompose a FILTER expression into equality substitutions and a residual.

    Scans *expr* for sub-expressions of the form ``?var = <IRI>`` or
    ``?var = "literal"`` (and their commutative forms ``concrete = ?var``).
    At the top level of a ``ConditionalAndExpression`` (``&&``), each such
    equality is extracted independently; the remaining sub-expressions are
    collected into the residual.

    Returns
    -------
    equalities : dict[Variable, URIRef | Literal]
        Variable-to-concrete-term substitutions found.
    residual   : algebra expression node | None
        The part of the expression that could not be folded into equalities,
        or ``None`` if the entire expression was consumed.
    """
    equalities: dict[Variable, Any] = {}

    def _try_equality(e: Any) -> bool:
        """Record e as an equality if it is ``?var = concrete``; return True."""
        if getattr(e, "name", None) != "RelationalExpression":
            return False
        if e.get("op") != "=":
            return False
        left, right = e.get("expr"), e.get("other")
        if isinstance(left, Variable) and isinstance(right, (URIRef, Literal)):
            equalities[left] = right
            return True
        if isinstance(right, Variable) and isinstance(left, (URIRef, Literal)):
            equalities[right] = left
            return True
        return False

    # Single equality expression
    if _try_equality(expr):
        return equalities, None

    # AND: split into equality and non-equality parts
    if getattr(expr, "name", None) == "ConditionalAndExpression":
        all_parts = [expr.get("expr")] + list(expr.get("other") or [])
        remainder = [p for p in all_parts if not _try_equality(p)]
        if not remainder:
            return equalities, None
        if len(remainder) == 1:
            return equalities, remainder[0]
        # Rebuild a ConditionalAndExpression for the non-equality parts
        residual = CompValue(
            "ConditionalAndExpression",
            expr=remainder[0],
            other=remainder[1:],
        )
        residual["_vars"] = getattr(expr, "_vars", set()) - set(equalities.keys())
        return equalities, residual

    # Not an equality and not an AND — return unchanged
    return {}, expr


def rewrite_filter_equalities(
    filter_expr: Any,
    bgp_triples: list[tuple],
) -> tuple[list[tuple], Any | None]:
    """
    B5-PRE — Rewrite FILTER equality constraints into concrete BGP terms.

    For every sub-expression ``?var = <IRI>`` or ``?var = "literal"`` found
    in *filter_expr* (extracted by ``_extract_equalities``), substitute the
    concrete term for the variable everywhere in *bgp_triples*.  Returns the
    rewritten triples and the residual expression (the non-equality parts that
    remain as a FILTER).

    Why this approach is superior to SQL fragment injection for equalities
    ───────────────────────────────────────────────────────────────────────
    Injecting ``?var = 'val'`` as a SQL WHERE fragment (B5) works, but
    substituting the concrete term directly into the BGP is strictly better:

    1. **Termtype pruning** — ``match_triple_pattern`` prunes rules whose
       ``subject/object_termtype`` is incompatible with the concrete term type
       (e.g. Literal vs. IRI) before any SQL is issued.  This eliminates whole
       mapping rules that a WHERE fragment alone cannot prune.

    2. **Existing pushdown path** — ``_extract_references_from_term`` already
       handles concrete non-variable terms (the normal case for ?p and ?s).
       No new SQL translation code is needed; the equality simply travels
       through the machinery that is already there.

    3. **Post-filter elimination** — when the rewrite fully consumes the FILTER
       expression (residual is ``None``), the SPARQL post-filter wrapper is
       skipped entirely.  The result is exact, not approximate.

    4. **``var = var`` support** — substituting one variable for another is
       handled naturally; SQL fragment injection has no column mapping for it.

    The residual (non-equality) expressions are handed to B5's
    ``filter_expr_to_sql`` for SQL injection, and a Python post-filter is
    applied as a safety net only if a residual remains.

    Parameters
    ----------
    filter_expr:
        The rdflib algebra FILTER expression node.
    bgp_triples:
        The triple patterns from ``part.p.triples`` (mutable list copy).

    Returns
    -------
    (new_triples, residual_expr)
        ``new_triples``    — BGP triples with equality variables substituted.
        ``residual_expr``  — Remaining FILTER expression, or ``None`` if the
                             entire FILTER was consumed by the rewrite.
    """
    equalities, residual = _extract_equalities(filter_expr)
    if not equalities:
        return bgp_triples, filter_expr

    def _subst(term: Any) -> Any:
        return equalities[term] if isinstance(term, Variable) and term in equalities else term

    new_triples = [(_subst(s), _subst(p), _subst(o)) for s, p, o in bgp_triples]
    return new_triples, residual



def _collect_filter_vars(expr: Any) -> set:
    """
    Recursively collect all ``Variable`` references from a SPARQL filter
    expression tree.  rdflib does not reliably populate ``_vars`` for all
    expression node types (e.g. ``RelationalExpression`` has ``_vars=set()``),
    so we walk the tree manually.
    """
    if isinstance(expr, Variable):
        return {expr}
    if isinstance(expr, (URIRef, Literal, str, int, float, bool, type(None))):
        return set()
    found: set = set()
    # CompValue / Expr nodes are dict-like; iterate their values
    try:
        for val in expr.values():
            found |= _collect_filter_vars(val)
    except AttributeError:
        pass
    return found

def filter_expr_to_sql(
    expr: Any,
    var_to_col: dict[Variable, str],
) -> str | None:
    """
    B5 — Translate an rdflib SPARQL algebra FILTER expression tree into a SQL
    WHERE fragment.  Returns ``None`` for any expression that cannot be safely
    translated, signalling the caller to skip pushdown for that expression.

    This function is CONSERVATIVE by design: a false ``None`` is merely
    suboptimal (falls back to Python post-filter); a wrong SQL fragment could
    silently drop valid rows.  The SPARQL post-filter is ALWAYS applied by the
    caller regardless of what this function returns.

    Supported translations
    ──────────────────────
    SPARQL                              SQL
    ────────────────────────────────    ─────────────────────────────────────
    ?x = \'val\'                          col = \'val\'
    ?x != \'val\'                         col != \'val\'
    ?x > N  /  ?x >= N                  col > N  /  col >= N
    ?x < N  /  ?x <= N                  col < N  /  col <= N
    ?x IN (v1, v2, ...)                 col IN (v1, v2, ...)
    ?x NOT IN (v1, v2, ...)             col NOT IN (v1, v2, ...)
    A && B && C                         (A AND B AND C)
    A || B || C                         (A OR B OR C)
    !BOUND(?x)                          col IS NULL
    BOUND(?x)                           col IS NOT NULL
    CONTAINS(?x, \'foo\')                col LIKE \'%foo%\'
    STRSTARTS(STR(?x), \'pre\')          col LIKE \'pre%\'
    STRENDS(STR(?x), \'.com\')           col LIKE \'%.com\'
    REGEX(?x, \'pat\')  (no flags)       col REGEXP \'pat\'
    NOT expr                            NOT (sql)

    NOT translatable (returns None): LANG, DATATYPE, isIRI, isLiteral,
    REGEX with flags, sub-select references, aggregate functions.

    SQL injection safety: Literal values are passed through ``_sql_escape``;
    column names come from the trusted RML mapping configuration only; numeric
    literals are validated by regex before emitting unquoted.
    """
    name = getattr(expr, "name", None)

    # ── Leaves ───────────────────────────────────────────────────────────────
    if isinstance(expr, Variable):
        return var_to_col.get(expr)          # None if variable has no col mapping
    if isinstance(expr, Literal):
        # Skip SQL pushdown for temporal datatypes — the target column type is
        # unknown and may be incompatible (e.g. INTEGER vs DATE).  The Python
        # post-filter via _eval_filter_expr handles temporal comparisons correctly.
        dt = str(expr.datatype) if expr.datatype else ""
        if dt in _TEMPORAL_XSD_TYPES:
            return None
        return _literal_to_sql_value(expr)
    if isinstance(expr, URIRef):
        return f"'{_sql_escape(str(expr))}'"
    if name is None:
        return None

    # ── RelationalExpression: ?var OP value ──────────────────────────────────
    if name == "RelationalExpression":
        op    = expr.get("op")
        left  = expr.get("expr")
        right = expr.get("other")

        if op == "IN":
            col  = filter_expr_to_sql(left, var_to_col)
            if col is None or not isinstance(right, list):
                return None
            vals = [filter_expr_to_sql(v, var_to_col) for v in right]
            if any(v is None for v in vals):
                return None
            return f"{col} IN ({', '.join(vals)})"

        if op == "NOT IN":
            col  = filter_expr_to_sql(left, var_to_col)
            if col is None or not isinstance(right, list):
                return None
            vals = [filter_expr_to_sql(v, var_to_col) for v in right]
            if any(v is None for v in vals):
                return None
            return f"{col} NOT IN ({', '.join(vals)})"

        sql_op = _RELATIONAL_OP_MAP.get(op)
        if sql_op is None:
            return None
        l = filter_expr_to_sql(left,  var_to_col)
        r = filter_expr_to_sql(right, var_to_col)
        if l is None or r is None:
            return None
        return f"{l} {sql_op} {r}"

    # ── Logical AND / OR ──────────────────────────────────────────────────────
    if name == "ConditionalAndExpression":
        parts = [filter_expr_to_sql(expr.get("expr"), var_to_col)]
        for other in (expr.get("other") or []):
            parts.append(filter_expr_to_sql(other, var_to_col))
        if any(p is None for p in parts):
            return None
        return "(" + " AND ".join(parts) + ")"

    if name == "ConditionalOrExpression":
        parts = [filter_expr_to_sql(expr.get("expr"), var_to_col)]
        for other in (expr.get("other") or []):
            parts.append(filter_expr_to_sql(other, var_to_col))
        if any(p is None for p in parts):
            return None
        return "(" + " OR ".join(parts) + ")"

    # ── Negation ──────────────────────────────────────────────────────────────
    if name == "UnaryNot":
        inner_expr = expr.get("expr")
        # !BOUND(?x) → col IS NULL  (special-case: more readable than NOT IS NOT NULL)
        if getattr(inner_expr, "name", None) == "Builtin_BOUND":
            col = filter_expr_to_sql(inner_expr.get("arg"), var_to_col)
            return f"{col} IS NULL" if col else None
        inner_sql = filter_expr_to_sql(inner_expr, var_to_col)
        return f"NOT ({inner_sql})" if inner_sql else None

    # ── BOUND ─────────────────────────────────────────────────────────────────
    if name == "Builtin_BOUND":
        col = filter_expr_to_sql(expr.get("arg"), var_to_col)
        return f"{col} IS NOT NULL" if col else None

    # ── String builtins ───────────────────────────────────────────────────────
    if name == "Builtin_CONTAINS":
        arg1, arg2 = expr.get("arg1"), expr.get("arg2")
        if getattr(arg1, "name", None) == "Builtin_STR":
            arg1 = arg1.get("arg")
        if not isinstance(arg2, Literal):
            return None
        col = filter_expr_to_sql(arg1, var_to_col)
        return f"{col} LIKE '%{_sql_escape(str(arg2))}%'" if col else None

    if name == "Builtin_STRSTARTS":
        arg1, arg2 = expr.get("arg1"), expr.get("arg2")
        if getattr(arg1, "name", None) == "Builtin_STR":
            arg1 = arg1.get("arg")
        if not isinstance(arg2, Literal):
            return None
        col = filter_expr_to_sql(arg1, var_to_col)
        return f"{col} LIKE '{_sql_escape(str(arg2))}%'" if col else None

    if name == "Builtin_STRENDS":
        arg1, arg2 = expr.get("arg1"), expr.get("arg2")
        if getattr(arg1, "name", None) == "Builtin_STR":
            arg1 = arg1.get("arg")
        if not isinstance(arg2, Literal):
            return None
        col = filter_expr_to_sql(arg1, var_to_col)
        return f"{col} LIKE '%{_sql_escape(str(arg2))}'" if col else None

    if name == "Builtin_REGEX":
        text    = expr.get("text")
        pattern = expr.get("pattern")
        flags   = expr.get("flags")
        # flags is a sentinel string 'flags' when absent; a real Literal when present.
        # Flag-modified REGEX is not safely translatable across SQL dialects.
        if isinstance(flags, Literal):
            return None
        if not isinstance(pattern, Literal):
            return None
        if getattr(text, "name", None) == "Builtin_STR":
            text = text.get("arg")
        col = filter_expr_to_sql(text, var_to_col)
        # REGEXP: SQLite (Python re UDF) + MySQL.  PostgreSQL callers should
        # subclass VIRTStore and override this with dialect="postgresql".
        return f"{col} REGEXP '{_sql_escape(str(pattern))}'" if col else None

    # ── STR() passthrough ─────────────────────────────────────────────────────
    if name == "Builtin_STR":
        return filter_expr_to_sql(expr.get("arg"), var_to_col)

    # ── Unary minus (numeric negation) ────────────────────────────────────────
    if name == "UnaryMinus":
        inner_sql = filter_expr_to_sql(expr.get("expr"), var_to_col)
        return f"-({inner_sql})" if inner_sql else None

    # All other nodes (LANG, DATATYPE, isIRI, isLiteral, IF, COALESCE, …)
    # cannot be safely translated — return None to trigger Python post-filter.
    return None


def build_var_to_col(
    bgp: list[tuple],
    rml_tp_df: pd.DataFrame,
) -> dict[Variable, str]:
    """
    B5 — Build a ``Variable → SQL column name`` mapping from the RML rules
    matched by *bgp*.

    Only ``rml:reference`` positions yield direct column mappings.
    ``rml:template`` positions involve composite expressions (prefix + column)
    that cannot be mapped 1-to-1 to a SQL column without expression rewriting.

    Parameters
    ----------
    bgp:
        List of ``(s, p, o)`` triple patterns (post-ordering).
    rml_tp_df:
        Morph-KGC mapping rules DataFrame, already filtered to rules relevant
        to the BGP (output of ``match_triple_pattern``).

    Returns
    -------
    dict[Variable, str]
        e.g. ``{Variable("age"): "passenger_age", Variable("name"): "name"}``
    """
    var_to_col: dict[Variable, str] = {}
    POSITIONS = [
        (0, "subject_map_type",   "subject_map_value"),
        (1, "predicate_map_type", "predicate_map_value"),
        (2, "object_map_type",    "object_map_value"),
    ]
    for tp in bgp:
        for pos_idx, type_col, value_col in POSITIONS:
            term = tp[pos_idx]
            if not isinstance(term, Variable) or rml_tp_df.empty:
                continue
            type_series  = rml_tp_df[type_col].fillna("").astype(str)
            value_series = rml_tp_df[value_col].fillna("").astype(str)
            ref_mask = type_series == RML_REFERENCE
            if not ref_mask.any():
                continue
            # Use the most common column name among reference-type rules
            mode = value_series[ref_mask].mode()
            if not mode.empty:
                var_to_col[term] = mode.iloc[0]
    return var_to_col


def filter_touches_only_bgp_vars(expr: Any, bgp_vars: set[Variable]) -> bool:
    """
    B5 — Return True if *expr* only references variables bound within the
    current BGP.

    Filters referencing variables from outer scopes or patterns not yet
    materialised in the current bind-join step cannot be pushed into the
    current SQL query — they must remain as Python post-filters.
    """
    expr_vars: set[Variable] = getattr(expr, "_vars", set())
    return expr_vars.issubset(bgp_vars)


def _inject_where(base_sql: str, conditions: list[str]) -> str:
    """Append a WHERE clause with *conditions* to *base_sql*."""
    if not conditions:
        return base_sql
    return f"{base_sql} WHERE {' AND '.join(conditions)}"


def _build_conditions(ref_values: dict[str, list[str]]) -> list[str]:
    """
    Convert a ``{column: [value, ...]}`` dict into SQL equality / IN conditions.

    Integer-valued columns are handled by ``keep_integer_strings_or_all`` to
    avoid type mismatches on strictly-typed SQL engines.

    A12 — Large value sets are split into chunks of at most ``_SQL_IN_CHUNK_SIZE``
    values and combined with OR.  Without chunking, SQLite raises
    ``SQLITE_LIMIT_VARIABLE_NUMBER`` for >999 parameters, and some MySQL
    configurations also impose a hard limit.  Each chunk is emitted as a
    separate ``col IN (...)`` expression joined by OR, which every major SQL
    engine optimises identically to a single large IN clause.
    """
    conditions: list[str] = []
    for col, vals in ref_values.items():
        unique = list(dict.fromkeys(v for v in vals if v is not None))
        if not unique:
            continue
        unique = keep_integer_strings_or_all(unique)

        if len(unique) == 1:
            conditions.append(f"{col} = '{unique[0]}'")
            continue

        # A12 — chunk into batches to respect SQL engine variable limits
        chunks = [
            unique[i : i + _SQL_IN_CHUNK_SIZE]
            for i in range(0, len(unique), _SQL_IN_CHUNK_SIZE)
        ]
        chunk_exprs = [
            f"{col} IN ({', '.join(f'{chr(39)}{v}{chr(39)}' for v in chunk)})"
            for chunk in chunks
        ]
        # Multiple chunks are OR'd; wrap in parens so the caller can AND safely
        conditions.append(
            chunk_exprs[0] if len(chunk_exprs) == 1
            else f"({' OR '.join(chunk_exprs)})"
        )
    return conditions


def _compute_ref_values_for_rule(
    positions: list[tuple],
    has_bindings: bool,
    bindings_df: pd.DataFrame | None,
) -> dict[str, list[str]] | None:
    """Compute the {column: [values]} pushdown dict for a single mapping rule.

    Returns ``None`` when the rule is impossible given current bindings:
      - a concrete term is structurally incompatible with the mapping rule, or
      - a bound join variable has **no** values compatible with the rule.

    IMPORTANT: When a join variable contains mixed bindings from multiple
    subject templates (e.g., Stops + ShapePoints), incompatibilities are
    expected. We therefore *skip* incompatible binding values and only abort
    when *all* values are incompatible for that rule.
    """
    ref_values: dict[str, list[str]] = {}

    for pat_term, map_type, map_value in positions:
        # Variable position: derive WHERE conditions from current bindings.
        if isinstance(pat_term, Variable):
            var_name = str(pat_term)
            if has_bindings and bindings_df is not None and var_name in bindings_df.columns:
                any_compatible = False
                for term_val in bindings_df[var_name].dropna().unique():
                    refs = _extract_references_from_term(term_val, map_type, map_value)
                    if refs is None:
                        # Mixed-template bindings are common; ignore values
                        # that cannot be reverse-matched against this rule.
                        continue
                    any_compatible = True
                    for ref, vals in refs.items():
                        ref_values.setdefault(ref, []).extend(
                            vals if isinstance(vals, list) else [vals]
                        )

                # Variable is bound but none of its values match this rule.
                # The rule cannot contribute to the join, so prune it.
                if not any_compatible:
                    return None
            continue

        # Concrete term position: must be compatible, otherwise prune rule.
        refs = _extract_references_from_term(pat_term, map_type, map_value)
        if refs is None:
            return None
        for ref, vals in refs.items():
            ref_values.setdefault(ref, []).extend(
                vals if isinstance(vals, list) else [vals]
            )

    return ref_values



# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5b — B7: Projection Pushdown (Column Pruning) helpers
# ══════════════════════════════════════════════════════════════════════════════

def _needed_columns(map_type: str, map_value: str, is_variable: bool) -> list[str]:
    """
    B7 — Return the SQL column names that must appear in the SELECT list for
    this (map_type, map_value, is_variable) position.

    - ``rml:reference`` + variable  → [map_value]  (the single column)
    - ``rml:template``  + variable  → all {ref} placeholders extracted from the
                                      template string (they are all needed to
                                      reconstruct the IRI/Literal value)
    - concrete term (not variable)  → []  (value already pushed as WHERE; no
                                      column value needs to travel to Python)
    - ``rml:constant``              → []  (no column involved at all)
    """
    if not is_variable:
        return []
    if map_type == RML_REFERENCE:
        return [map_value]
    if map_type == RML_TEMPLATE:
        _, refs = _compile_rml_template(map_value)
        return refs
    return []  # rml:constant or unknown


def _build_projected_sql(
    src_type: str,
    src_val: str,
    needed_cols: list[str],
    conditions: list[str],
    distinct: bool = False,
) -> tuple[str, str]:
    """
    B7 — Build a projected SQL query that SELECTs only *needed_cols* instead
    of ``SELECT *``, and appends *conditions* as a WHERE clause.

    B10 — When *distinct* is ``True``, emits ``SELECT DISTINCT col1, col2, ...``
    so the database engine eliminates duplicate rows before they are transferred
    to Python.  This is safe when *needed_cols* is the complete set of columns
    that determine RDF term identity (guaranteed by B7 which always includes all
    ``{ref}`` placeholders from ``rml:template`` positions).

    Returns a ``(new_logical_source_type, new_logical_source_value)`` pair
    ready to be written back into the mapping DataFrame.

    If *needed_cols* is empty (can happen if every variable position uses
    ``rml:constant``), falls back to ``SELECT [DISTINCT] *`` to avoid an
    invalid ``SELECT  FROM ...`` statement.

    The projected form is::

        SELECT [DISTINCT] col1, col2, ... FROM (original_sql) AS _subquery WHERE ...
    or
        SELECT [DISTINCT] col1, col2, ... FROM table WHERE ...
    """
    qualifier = "DISTINCT " if distinct else ""
    col_list  = ", ".join(dict.fromkeys(needed_cols)) if needed_cols else "*"

    if src_type == RML_QUERY:
        inner = src_val.strip().rstrip(";")
        base_sql = f"SELECT {qualifier}{col_list} FROM ({inner}) AS _subquery"
    else:  # RML_TABLENAME
        base_sql = f"SELECT {qualifier}{col_list} FROM {src_val}"

    return RML_QUERY, _inject_where(base_sql, conditions)


def pushdown_bindings_to_sql(
    triple_pattern: TriplePattern,
    rml_df: pd.DataFrame,
    bindings_df: pd.DataFrame | None = None,
    extra_sql: str | None = None,
    distinct: bool = False,
) -> pd.DataFrame:
    """
    Push intermediate variable bindings from *bindings_df* into the SQL
    logical-source queries of matching RML mapping rules.

    For every RDB-backed mapping rule in *rml_df* that is matched by
    *triple_pattern*, the function computes SQL filter conditions from:

    - Concrete terms in *triple_pattern* (constant pushdown).
    - Values of bound variables available in *bindings_df* (bind-join pushdown).

    The updated ``logical_source_value`` (SQL string) is written back into a
    copy of *rml_df* which is returned; the original is never mutated.

    B10 — Projection deduplication: when *distinct* is ``True``, emits
    ``SELECT DISTINCT`` so the SQL engine eliminates duplicate rows before
    any data crosses the driver boundary.  Requires B7 to be active (so
    the projected column list fully determines RDF term identity).

    B7 — Projection pushdown: instead of ``SELECT *``, only the columns
    actually needed to reconstruct variable-bound RDF terms are projected.
    Both ``rml:reference`` (direct column) and ``rml:template`` (all ``{ref}``
    placeholders) variable positions contribute to the projected column list.
    Filter-only columns from WHERE conditions are also included so that SQL
    engines that require projected columns to resolve filter expressions work
    correctly.

    B5 — ``extra_sql``: optional SQL WHERE fragment from ``filter_expr_to_sql``
    appended as an additional AND condition alongside bind-join conditions.
    This injects a SPARQL FILTER directly into the SQL query so the database
    engine filters rows before they are transferred to Python.

    A6 — Three-part vectorisation over the original ``iterrows`` loop:

    1. **``itertuples`` instead of ``iterrows``**: ``itertuples`` returns plain
       Python namedtuples without boxing values into a ``Series``, making row
       access 4–10× faster than ``iterrows``.

    2. **Configuration-keyed memoisation**: rules that share the same
       ``(map_type, map_value)`` triple at each position produce identical WHERE
       conditions regardless of which specific rule they belong to.  A per-call
       ``_config_cache`` dict stores already-computed ``ref_values`` dicts and
       skips ``_extract_references_from_term`` for duplicate configurations.

    3. **Batched mutations via ``loc[]``**: SQL rewrites are collected into a
       plain dict ``{row_index: new_sql}`` during the loop and applied in two
       vectorised ``loc[]`` assignments after the loop, rather than one
       ``at[]`` call per row (which re-checks copy-on-write on every write).

    OPT-11 copy-on-write is preserved: the DataFrame is only copied when at
    least one rule actually requires SQL rewriting.
    """
    s_pat, p_pat, o_pat = triple_pattern
    has_bindings = bindings_df is not None and not bindings_df.empty

    POSITIONS = [
        (s_pat, "subject_map_type",   "subject_map_value"),
        (p_pat, "predicate_map_type", "predicate_map_value"),
        (o_pat, "object_map_type",    "object_map_value"),
    ]

    # Collect mutations: index → (new_logical_source_type, new_logical_source_value)
    # We separate type changes (TABLENAME → QUERY) from value changes so we can
    # apply them with two loc[] calls instead of N at[] calls.
    new_types:  dict[Any, str] = {}
    new_values: dict[Any, str] = {}

    # Rules impossible given current bindings are pruned when bindings exist.
    rows_to_drop: list[Any] = []

    # A6 — per-call cache: (pos0_type, pos0_val, pos1_type, pos1_val, ...) → ref_values
    # Avoids recomputing _extract_references_from_term for rules with identical configs.
    _config_cache: dict[tuple, dict[str, list[str]] | None] = {}

    # A6 — itertuples is 4–10× faster than iterrows: no Series boxing,
    # values accessed as namedtuple attributes.
    for row in rml_df.itertuples():
        if getattr(row, "source_type", None) != "RDB":
            continue

        # Build the position config tuple for cache keying
        resolved_positions = []
        for pat_term, type_col, value_col in POSITIONS:
            map_type  = str(getattr(row, type_col,  None) or "")
            map_value = str(getattr(row, value_col, None) or "")
            resolved_positions.append((pat_term, map_type, map_value))

        config_key = tuple(
            (map_type, map_value)
            for _, map_type, map_value in resolved_positions
        )

        if config_key not in _config_cache:
            _config_cache[config_key] = _compute_ref_values_for_rule(
                resolved_positions, has_bindings, bindings_df
            )
        ref_values = _config_cache[config_key]

        # If the rule cannot match any of the current bindings, prune it to
        # avoid scanning an entire source table only to have the join discard
        # the rows later.
        if ref_values is None:
            if has_bindings:
                rows_to_drop.append(row.Index)
            continue

        # No constraints to inject: compatible rule but no WHERE conditions.
        if not ref_values:
            continue

        conditions = _build_conditions(ref_values)
        # B5 — append the FILTER SQL fragment as an extra AND condition
        if extra_sql:
            conditions.append(extra_sql)

        src_type = str(getattr(row, "logical_source_type", None) or "")
        src_val  = str(getattr(row, "logical_source_value", None) or "").strip()

        if src_type not in (RML_QUERY, RML_TABLENAME):
            continue

        # B10 — pass distinct flag; use SELECT * (no column pruning) so that
        # morph-kgc's internal pipeline (_preprocess_data, remove_null_values)
        # sees all mapping columns. B7 column projection is intentionally
        # disabled: morph-kgc computes `references` from the full mapping rule
        # and calls dropna(subset=references), which raises KeyError when
        # projected columns are absent from the SQL result.
        new_type, new_sql = _build_projected_sql(src_type, src_val, [], conditions, distinct=distinct)
        new_types[row.Index]  = new_type
        new_values[row.Index] = new_sql
    if not new_values and not rows_to_drop:
        # No mutations needed — return original DataFrame without any copy
        return rml_df
    # OPT-11 + A6 — single copy, then vectorised pruning and loc[] assignments
    result_df = rml_df.copy()

    # Prune rules that cannot join with the current bindings.
    if rows_to_drop:
        result_df = result_df.drop(index=rows_to_drop)

    if new_types:
        # Only assign into surviving rows.
        idx = [i for i in new_types.keys() if i in result_df.index]
        if idx:
            result_df.loc[idx, "logical_source_type"] = [new_types[i] for i in idx]

    if new_values:
        idx = [i for i in new_values.keys() if i in result_df.index]
        if idx:
            result_df.loc[idx, "logical_source_value"] = [new_values[i] for i in idx]

    return result_df


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — BGP evaluation
# ══════════════════════════════════════════════════════════════════════════════



# ── B12 — Generalized semi-join reduction (dangling tuple elimination) ─────────
# Classic distributed query processing optimization: reduce intermediate join
# inputs by iteratively applying semi-joins on shared join variables.
# Inspired by Bernstein & Chiu (1981): "Using Semi-Joins to Solve Relational Queries".

def semijoin_reduce_bgp(
    ctx: QueryContext,
    bgp: list[TriplePattern],
    rml_df: pd.DataFrame,
    config: Any,
    rounds: int = 2,
) -> list[pd.DataFrame]:
    """Return one reducer DataFrame per triple pattern.

    Each reducer contains only join-variable columns (variables that appear in
    at least two triple patterns) reduced by up to `rounds` iterations of
    pairwise semi-joins on shared variables.

    Reducers can be applied as additional filters on each triple pattern's
    bindings before expensive joins.

    Notes
    -----
    * Minimal and conservative implementation.
    * Materializes each triple pattern once to compute join-key projections.
    * If any triple pattern is empty, all reducers are returned empty.
    """

    if not bgp or len(bgp) < 2:
        return [pd.DataFrame() for _ in bgp]

    from collections import defaultdict as _dd
    var_counts: _dd[str, int] = _dd(int)
    for tp in bgp:
        for v in set(triple_pattern_variables(tp)):
            var_counts[v] += 1

    join_vars = {v for v, c in var_counts.items() if c >= 2}
    if not join_vars:
        return [pd.DataFrame() for _ in bgp]

    proj: list[pd.DataFrame] = []
    for tp in bgp:
        rml_tp_df = match_triple_pattern(tp, rml_df)
        if rml_tp_df.empty:
            return [pd.DataFrame() for _ in bgp]

        try:
            data = _materialize_mapping_group_to_df(rml_tp_df, rml_df, None, config)
        except ValueError:
            return [pd.DataFrame() for _ in bgp]

        if data.empty:
            return [pd.DataFrame() for _ in bgp]

        data = apply_termtypes_to_df(data, rml_tp_df)
        renamed = rename_triple_columns(data, tp)
        cols = [c for c in renamed.columns if c in join_vars]
        proj.append(renamed[cols].drop_duplicates() if cols else pd.DataFrame())

    for _ in range(max(0, rounds)):
        changed = False
        for i in range(len(proj)):
            if proj[i].empty:
                continue
            for j in range(len(proj)):
                if i == j or proj[j].empty:
                    continue
                shared = list(set(proj[i].columns) & set(proj[j].columns))
                if not shared:
                    continue
                before = len(proj[i])
                keys = proj[j][shared].drop_duplicates()
                proj[i] = proj[i].merge(keys, on=shared, how='inner').drop_duplicates()
                if len(proj[i]) != before:
                    changed = True
        if not changed:
            break

    return proj
def virt_eval_bgp(
    ctx: QueryContext,
    bgp: BGP,
    rml_df: pd.DataFrame,
    config: Any,
    initial_bindings_df: pd.DataFrame | None = None,
    filter_sql: str | None = None,
    distinct: bool = False,
    bloom_filter: bool = True,
    semijoin_reduction: bool = False,
    semijoin_rounds: int = 2,
) -> Iterator[FrozenBindings]:
    """
    Evaluate a BGP against the heterogeneous data sources described by
    *rml_df* using a bind-join strategy.

    Parameters
    ----------
    ctx:
        The current SPARQL query context.
    bgp:
        Ordered list of triple patterns to evaluate (use ``order_bgp`` first).
    rml_df:
        Morph-KGC mapping rules DataFrame.
    config:
        Morph-KGC configuration object.
    initial_bindings_df:
        Optional DataFrame of bindings from a preceding algebra node (e.g.
        the left side of a LeftJoin or a Join). When provided, its values are
        pushed down into the first triple pattern's SQL query.
    filter_sql:
        B5 — Optional SQL WHERE fragment produced by ``filter_expr_to_sql``.
    distinct:
        B10 — When ``True``, passes ``distinct=True`` to
        ``pushdown_bindings_to_sql`` so the first triple pattern's SQL query
        uses ``SELECT DISTINCT`` to eliminate duplicates at source level.
        Activated by the ``Distinct`` algebra handler in ``custom_eval``.
        When provided, it is injected into the first triple pattern's SQL
        query as an additional WHERE condition, pushing the FILTER evaluation
        into the database engine before rows are transferred to Python.
        The SPARQL-level post-filter is always applied by the caller regardless.
    bloom_filter:
        B11 — When ``True`` (default), a Bloom filter is built from the
        join-key values in ``bindings_df`` after each triple pattern and used
        to probe the next pattern's materialised rows before ``natural_join``.
        Rows that are definitely absent (no false negatives guaranteed) are
        dropped at O(n) bitarray cost, reducing merge input size.  Activates
        only when ``len(bindings_df) >= _B11_THRESHOLD``.  Pass ``False`` to
        disable entirely (e.g. for benchmarking or small datasets).

    Yields
    ------
    FrozenBindings
        One solution mapping per result row, with all cell values converted
        to proper RDFLib terms via ``to_rdf_term``.
    """
    # Resolve already-bound variables from ctx into the triple patterns.
    def _resolve(term: RDFTerm) -> RDFTerm:
        if isinstance(term, Variable):
            val = ctx[term]
            return val if val is not None else term
        return term

    bgp = [(_resolve(s), _resolve(p), _resolve(o)) for s, p, o in bgp]

    if not bgp:
        return

    # B12 — Optional generalized semi-join reduction
    reducers: list[pd.DataFrame] = [pd.DataFrame() for _ in bgp]
    if semijoin_reduction and len(bgp) > 1:
        try:
            reducers = semijoin_reduce_bgp(ctx, bgp, rml_df, config, rounds=semijoin_rounds)
        except Exception:
            reducers = [pd.DataFrame() for _ in bgp]

    # Helper: access a column even if duplicate labels exist (pandas returns a DataFrame)
    def _col_series(df: pd.DataFrame, name: str):
        col = df.loc[:, name]
        return col.iloc[:, -1] if isinstance(col, pd.DataFrame) else col

    if _VIRT_DEBUG:
        print(f"[VIRT_DEBUG] virt_eval_bgp called, bgp={bgp}")

    # ── First triple pattern ──────────────────────────────────────────────────
    first_tp = bgp[0]
    rml_tp_df = match_triple_pattern(first_tp, rml_df)
    if _VIRT_DEBUG:
        print(f"[VIRT_DEBUG] tp0={first_tp}: matched {len(rml_tp_df)} rules")
    if rml_tp_df.empty:
        if _VIRT_DEBUG:
            print(f"[VIRT_DEBUG] tp0: no matching rules → return")
        return
    rml_tp_df = pushdown_bindings_to_sql(first_tp, rml_tp_df, initial_bindings_df, extra_sql=filter_sql, distinct=distinct)

    try:
        data = _materialize_mapping_group_to_df(rml_tp_df, rml_df, None, config)
    except ValueError:
        # All matched rules returned empty results — pd.concat([]) raises
        # ValueError("No objects to concatenate"). Treat as empty result.
        if _VIRT_DEBUG:
            print(f"[VIRT_DEBUG] tp0={first_tp}: materialize raised ValueError (empty concat) → return")
        return
    if _VIRT_DEBUG:
        sql_vals = rml_tp_df["logical_source_value"].tolist() if not rml_tp_df.empty else []
        print(f"[VIRT_DEBUG] tp0={first_tp}")
        print(f"  SQL={sql_vals}")
        print(f"  raw rows={len(data)}, cols={list(data.columns) if not data.empty else []}")
    if data.empty:
        if _VIRT_DEBUG:
            print(f"[VIRT_DEBUG] tp0={first_tp}: data empty → return")
        return
    data = apply_termtypes_to_df(data, rml_tp_df)
    # Correctness: enforce concrete SUBJECT/OBJECT terms at row level.
    # Needed because constants can be dropped later when rename_triple_columns
    # removes non-variable positions (e.g., after FILTER equality rewrite).
    _s, _p, _o = first_tp
    if not isinstance(_s, Variable) and 'subject' in data.columns:
        data = data[_col_series(data, 'subject') == _s]
    if not isinstance(_o, Variable) and 'object' in data.columns:
        # Booleans may have lexical variants ('1'/'0' vs 'true'/'false'); compare by value.
        if isinstance(_o, Literal) and _o.datatype and str(_o.datatype) == str(XSD.boolean):
            expected = bool(_o.toPython())
            ser = _col_series(data, 'object')
            data = data[ser.apply(lambda v: isinstance(v, Literal) and v.datatype and str(v.datatype)==str(XSD.boolean) and bool(v.toPython())==expected)]
        else:
            data = data[_col_series(data, 'object') == _o]
    if data.empty:
        if _VIRT_DEBUG:
            print(f"[VIRT_DEBUG] tp0={first_tp}: data empty after constant-term filter → return")
        return
    if _VIRT_DEBUG:
        if "object" in data.columns and not data.empty:
            _sample = data["object"].dropna().iloc[0] if not data["object"].dropna().empty else None
            print(f"[VIRT_DEBUG] apply_termtypes_to_df tp0={first_tp}: object sample={_sample!r} type={type(_sample).__name__}")
    bindings_df = rename_triple_columns(data, first_tp)
    # B12 — Apply semijoin reducer for tp0 (join-variable pruning)
    if semijoin_reduction and reducers and len(reducers) > 0 and not reducers[0].empty:
        shared = list(set(bindings_df.columns) & set(reducers[0].columns))
        if shared:
            before = len(bindings_df)
            bindings_df = bindings_df.merge(reducers[0][shared].drop_duplicates(), on=shared, how='inner')
            if _VIRT_DEBUG:
                print(f"[VIRT_DEBUG] B12 semijoin tp0: before={before} after={len(bindings_df)}")
            if bindings_df.empty:
                return
    if _VIRT_DEBUG:
        print(f"  bindings_df cols={list(bindings_df.columns)} rows={len(bindings_df)}")

    # ── Remaining triple patterns (bind-join) ─────────────────────────────────
    for tp_idx, tp in enumerate(bgp[1:], start=1):
        rml_tp_df = match_triple_pattern(tp, rml_df)
        if rml_tp_df.empty:
            if _VIRT_DEBUG:
                print(f"[VIRT_DEBUG] tp={tp}: match_triple_pattern empty → return")
            return
        rml_tp_df = pushdown_bindings_to_sql(tp, rml_tp_df, bindings_df)

        try:
            data = _materialize_mapping_group_to_df(rml_tp_df, rml_df, None, config)
        except ValueError:
            if _VIRT_DEBUG:
                print(f"[VIRT_DEBUG] tp={tp}: materialize raised ValueError (empty concat) → return")
            return
        if _VIRT_DEBUG:
            sql_vals = rml_tp_df["logical_source_value"].tolist() if not rml_tp_df.empty else []
            print(f"[VIRT_DEBUG] tp={tp}")
            print(f"  SQL={sql_vals}")
            print(f"  raw rows={len(data)}, cols={list(data.columns) if not data.empty else []}")
        if data.empty:
            if _VIRT_DEBUG:
                print(f"[VIRT_DEBUG] tp={tp}: data empty → return")
            return
        data = apply_termtypes_to_df(data, rml_tp_df)
        # Correctness: enforce concrete SUBJECT/OBJECT terms at row level.
        # Needed because constants can be dropped later when rename_triple_columns
        # removes non-variable positions (e.g., after FILTER equality rewrite).
        _s, _p, _o = tp
        if not isinstance(_s, Variable) and 'subject' in data.columns:
            data = data[_col_series(data, 'subject') == _s]
        if not isinstance(_o, Variable) and 'object' in data.columns:
            # Booleans may have lexical variants ('1'/'0' vs 'true'/'false'); compare by value.
            if isinstance(_o, Literal) and _o.datatype and str(_o.datatype) == str(XSD.boolean):
                expected = bool(_o.toPython())
                ser = _col_series(data, 'object')
                data = data[ser.apply(lambda v: isinstance(v, Literal) and v.datatype and str(v.datatype)==str(XSD.boolean) and bool(v.toPython())==expected)]
            else:
                data = data[_col_series(data, 'object') == _o]
        if data.empty:
            if _VIRT_DEBUG:
                print(f"[VIRT_DEBUG] tp={tp}: data empty after constant-term filter → return")
            return
        if _VIRT_DEBUG:
            if "object" in data.columns and not data.empty:
                _sample = data["object"].dropna().iloc[0] if not data["object"].dropna().empty else None
                print(f"[VIRT_DEBUG] apply_termtypes_to_df tp={tp}: object sample={_sample!r} type={type(_sample).__name__}")
        data = rename_triple_columns(data, tp)

        # B12 — Apply semijoin reducer for this tp (join-variable pruning)
        if semijoin_reduction and reducers and tp_idx < len(reducers):
            red = reducers[tp_idx]
            if red is not None and not red.empty:
                shared = list(set(data.columns) & set(red.columns))
                if shared:
                    before = len(data)
                    data = data.merge(red[shared].drop_duplicates(), on=shared, how='inner')
                    if _VIRT_DEBUG:
                        print(f"[VIRT_DEBUG] B12 semijoin tp={tp}: before={before} after={len(data)}")
                    if data.empty:
                        return

        # ── B11 — Bloom filter pre-join probe ─────────────────────────────────
        # Build a Bloom filter from the join-key values already in bindings_df
        # and probe each row of data before the exact natural_join merge.  Rows
        # whose join key is *definitely absent* from bindings_df are eliminated
        # here at O(n) bitarray cost; false positives pass through and are
        # removed exactly by natural_join.  The optimisation activates only when
        # bindings_df is large enough that filter-build overhead is warranted.
        if bloom_filter:
            join_cols = list(
                set(bindings_df.columns)
                & set(data.columns)
                & set(triple_pattern_variables(tp))
            )
            if join_cols and len(bindings_df) >= _B11_THRESHOLD:
                join_col = join_cols[0]
                bf_bits  = _bloom_build(
                    str(v) for v in bindings_df[join_col].dropna().unique()
                )
                before = len(data)
                data   = _bloom_filter_df(data, join_col, bf_bits)
                if _VIRT_DEBUG:
                    print(
                        f"[VIRT_DEBUG] B11: col={join_col!r} "
                        f"before={before} after={len(data)} "
                        f"eliminated={before - len(data)}"
                    )
                if data.empty:
                    if _VIRT_DEBUG:
                        print("[VIRT_DEBUG] B11: data empty after Bloom probe → return")
                    return

        if _VIRT_DEBUG:
            join_key_col = list(
                set(bindings_df.columns) & set(data.columns) & set(triple_pattern_variables(tp))
            )
            if join_key_col:
                jk = join_key_col[0]
                print(f"[VIRT_DEBUG] pre-join: bindings_df[{jk!r}] nunique={bindings_df[jk].nunique()} "
                      f"sample={bindings_df[jk].dropna().unique()[:3].tolist()}")
                print(f"[VIRT_DEBUG] pre-join: data[{jk!r}] nunique={data[jk].nunique()} "
                      f"sample={data[jk].dropna().unique()[:3].tolist()}")
        bindings_df = natural_join(bindings_df, data, tp)
        if not bindings_df.empty:
            bindings_df = bindings_df.drop_duplicates()
        if _VIRT_DEBUG:
            print(f"[VIRT_DEBUG] tp={tp}: after natural_join rows={len(bindings_df)}")
        if bindings_df.empty:
            if _VIRT_DEBUG:
                print(f"[VIRT_DEBUG] tp={tp}: bindings_df empty after join → return")
            return

    # ── Deduplicate before yielding ──────────────────────────────────────────
    # A real triplestore stores each triple once; duplicate triples are
    # impossible.  morph-KGC may materialise the same triple from multiple
    # mapping rules (e.g. ?trip a gtfs:Trip from both the trips and
    # stop_times tables), producing duplicate rows in bindings_df.  Without
    # deduplication these duplicates propagate through joins and OPTIONALs,
    # producing more results than an equivalent query against a materialised
    # triplestore.  drop_duplicates() on the variable columns brings
    # behaviour in line with real triplestores.
    if not bindings_df.empty:
        bindings_df = bindings_df.drop_duplicates()

    # ── Yield FrozenBindings ──────────────────────────────────────────────────
    # OPT-8 — Replace iterrows() with to_dict("records").
    # iterrows() boxes each value into a Series and returns (index, Series)
    # pairs, incurring significant Python overhead per row.  to_dict("records")
    # produces plain dicts via pandas' C layer and is 10–50× faster for large
    # result sets.  Variable keys are pre-computed once outside the loop to
    # avoid redundant Variable() construction per row.
    var_keys = {col: Variable(col) for col in bindings_df.columns}
    for row_dict in bindings_df.to_dict("records"):
        yield FrozenBindings(
            ctx,
            {var_keys[col]: row_dict[col] for col in var_keys},
        )


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — FrozenBindings helpers
# ══════════════════════════════════════════════════════════════════════════════

def _eval_filter_expr(expr: Any, row: FrozenBindings, ctx: QueryContext) -> bool:
    """
    Evaluate a SPARQL FILTER expression against a single solution mapping.

    Per SPARQL 1.1 §17.6, any evaluation error is treated as ``False``
    (filter failure) rather than propagating an exception.
    """
    try:
        result = bool(_ebv(expr, row.forget(ctx, _except=row._d)))
        if _VIRT_DEBUG:
            print(f"[VIRT_DEBUG] _eval_filter_expr: result={result}")
        return result
    except (SPARQLError, Exception) as _filter_exc:
        if _VIRT_DEBUG:
            # Show the first variable involved in the filter expression
            _vars = getattr(expr, "_vars", set())
            _vals = {str(v): row.get(v) for v in _vars}
            print(f"[VIRT_DEBUG] _eval_filter_expr FAILED: expr={expr} vars={_vals} exc={_filter_exc!r}")
        return False


def _are_compatible(left: FrozenBindings, right: FrozenBindings) -> bool:
    """
    Return True iff *left* and *right* agree on every variable bound in both
    (SPARQL 1.1 §18.1.2 compatibility definition).
    """
    for var, val in right._d.items():
        if val is None:
            continue
        left_val = left.get(var)
        if left_val is not None and left_val != val:
            return False
    return True


def _merge_bindings(
    ctx: QueryContext, left: FrozenBindings, right: FrozenBindings
) -> FrozenBindings:
    """
    Merge two compatible solution mappings. Left values take priority.

    A8 — Replace the two-step ``dict(left._d)`` copy + for-loop with a single
    dict-comprehension union.  ``dict(left._d)`` allocates an intermediate copy
    even when no right-side values are needed; the comprehension builds the
    final dict in one pass, skipping right-side values that are None or already
    bound on the left.  For large join outputs (thousands of compatible pairs
    after the OPT-1/A2 hash-join) this noticeably reduces allocations.
    """
    merged = {
        **left._d,
        **{k: v for k, v in right._d.items()
           if v is not None and left._d.get(k) is None},
    }
    return FrozenBindings(ctx, merged)


def _rows_to_bindings_df(rows: list[FrozenBindings]) -> pd.DataFrame:
    """Convert a list of FrozenBindings into a DataFrame for SQL pushdown."""
    return pd.DataFrame([
        {str(k): v for k, v in row._d.items() if v is not None}
        for row in rows
    ])


def _values_node_to_rows(ctx: QueryContext, part: Any) -> list[FrozenBindings]:
    """
    Materialise inline VALUES data (a ``ToMultiSet`` algebra node) into a
    list of FrozenBindings.
    """
    return [FrozenBindings(ctx, row_dict) for row_dict in part.p.res]


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — VIRTStore (RDFLib Store)
# ══════════════════════════════════════════════════════════════════════════════

class VIRTStore(Store):
    """
    A read-only RDFLib ``Store`` that evaluates SPARQL queries against
    heterogeneous data sources described via RML.

    Query evaluation uses a bind-join strategy throughout: intermediate
    bindings from each solved pattern are pushed down as SQL WHERE conditions
    into subsequent source queries.

    Supported SPARQL algebra nodes
    ──────────────────────────────
    BGP           { ?s ?p ?o }       Bind-join with SQL pushdown
    Filter        FILTER(...)        Post-filter via RDFLib _ebv
    LeftJoin      OPTIONAL { }       Batch SQL pushdown from p1 into p2
    Join          BGP . BGP          Batch SQL pushdown from p1 into p2
    Union         { } UNION { }      Both branches evaluated independently
    Minus         MINUS { }          In-memory subtraction per SPARQL §18.5
    ToMultiSet    VALUES ?x { ... }  Materialised inline, pushed to Join p2
    Distinct      SELECT DISTINCT     SQL DISTINCT pushed to source (B10); Python backstop for cross-rule dupes

    All other nodes (Extend/BIND, OrderBy, Distinct, Slice, Group,
    AggregateJoin, Graph, sub-selects) are delegated to RDFLib's native
    engine, which recurses through CUSTOM_EVALS for any nested BGPs.
    """

    context_aware     = False
    formula_aware     = False
    transaction_aware = False

    _EVAL_KEY = "VIRTStore"

    def __init__(
        self,
        config_path: str,
        configuration: Any = None,
        identifier: Any = None,
        bloom_filter: bool = True,
    ) -> None:
        """
        Initialise the VIRTStore.

        Parameters
        ----------
        config_path:
            Path to the Morph-KGC configuration file.
        configuration:
            Passed through to the rdflib ``Store`` base class.
        identifier:
            Passed through to the rdflib ``Store`` base class.
        bloom_filter:
            B11 — Enable the Bloom filter pre-join probe optimisation
            (default ``True``).  When ``True``, intermediate bind-join results
            exceeding ``_B11_THRESHOLD`` rows are used to build a Bloom filter
            that eliminates non-matching rows from the next triple pattern's
            materialised data before ``natural_join`` is called.  Set to
            ``False`` to disable B11 entirely (useful for benchmarking or when
            source tables are small and BF overhead is not warranted).
        """
        super().__init__(configuration=configuration, identifier=identifier)

        config = load_config_from_argument(config_path)
        rml_df, _fnml_df, http_api_df = retrieve_mappings(config)
        config.set("CONFIGURATION", "http_api_df", http_api_df.to_csv())

        self.rml_df = rml_df.loc[
            rml_df["triples_map_type"] == "http://w3id.org/rml/TriplesMap"
        ]
        self.config       = config
        self.bloom_filter = bloom_filter

        # B12 — Semijoin reduction toggle (disabled by default)
        self.semijoin_reduction = _os.environ.get('VIRT_SEMIJOIN', '0') == '1'
        try:
            self.semijoin_rounds = int(_os.environ.get('VIRT_SEMIJOIN_ROUNDS', '2'))
        except Exception:
            self.semijoin_rounds = 2

        CUSTOM_EVALS[self._EVAL_KEY] = self._make_custom_eval()

    def __del__(self) -> None:
        CUSTOM_EVALS.pop(self._EVAL_KEY, None)

    # ── Custom eval factory ───────────────────────────────────────────────────

    def _make_custom_eval(self):
        """
        Return a ``custom_eval`` closure registered in ``CUSTOM_EVALS``.

        The closure captures ``self`` so that ``rml_df`` and ``config`` are
        always accessible without global state. Raising ``NotImplementedError``
        for unhandled nodes causes RDFLib to fall back to its native handler,
        which recurses through ``CUSTOM_EVALS`` for any nested BGPs.
        """

        def custom_eval(ctx: QueryContext, part: Any) -> Any:
            if _VIRT_DEBUG:
                print(f"[VIRT_DEBUG] custom_eval called: part.name={part.name}")

            # ── BGP ──────────────────────────────────────────────────────────
            if part.name == "BGP":
                if _VIRT_DEBUG:
                    print(f"[VIRT_DEBUG] custom_eval BGP handler: {part.triples}")
                ordered = order_bgp(ctx, part.triples, self.rml_df)
                return iter(
                    list(virt_eval_bgp(ctx, ordered, self.rml_df, self.config, bloom_filter=self.bloom_filter))
                )

            # ── FILTER ───────────────────────────────────────────────────────
            if part.name == "Filter":
                # Two-stage FILTER optimisation for BGP inner nodes:
                #
                # Stage 1 — B5-PRE: equality substitution into BGP (this pass)
                # ──────────────────────────────────────────────────────────────
                # Rewrite FILTER(?var = <IRI>) and FILTER(?var = "literal")
                # by substituting the concrete term directly into every
                # matching triple pattern position.  Superior to SQL fragment
                # injection for equalities: termtype pruning fires before any
                # SQL is issued; existing pushdown path handles concrete terms;
                # when the rewrite consumes the entire FILTER (residual None),
                # the post-filter wrapper is skipped and the result is exact.
                #
                # Stage 2 — B5: SQL fragment injection for residual expressions
                # ──────────────────────────────────────────────────────────────
                # Non-equality sub-expressions (?age > 25, CONTAINS(?name,'x'))
                # are translated to a SQL WHERE fragment via filter_expr_to_sql
                # and injected via pushdown_bindings_to_sql. Python post-filter
                # applied as safety net when a residual remains.
                #
                # B5-PRE correctness — variable binding augmentation:
                # ──────────────────────────────────────────────────────────────
                # When B5-PRE substitutes ?var → concrete in the BGP,
                # rename_triple_columns drops that position's column (it is no
                # longer a Variable). Result rows therefore contain NO binding
                # for ?var, breaking SELECT ?var and downstream joins.
                # _augment re-injects the substituted values into every result
                # row so that callers see the correct bound value.

                if _VIRT_DEBUG:
                    print(f"[VIRT_DEBUG] Filter handler reached: inner={part.p.name}, expr={part.expr}")
                if part.p.name == "BGP":
                    bgp_triples = list(part.p.triples)
                    filter_expr = part.expr
                    bgp_triples, residual = rewrite_filter_equalities(
                        filter_expr, bgp_triples
                    )
                    # Re-extract equality dict for result augmentation.
                    equalities, _ = _extract_equalities(filter_expr)
                    ordered_bgp = order_bgp(ctx, bgp_triples, self.rml_df)

                    sql_frag: str | None = None
                    if (
                        residual is not None
                        and filter_touches_only_bgp_vars(
                            residual,
                            set(bgp_variables(bgp_triples)),
                        )
                    ):
                        rml_tp_df_f = match_triple_pattern(ordered_bgp[0], self.rml_df)
                        var_to_col  = build_var_to_col(ordered_bgp, rml_tp_df_f)
                        sql_frag    = filter_expr_to_sql(residual, var_to_col)

                    inner = virt_eval_bgp(
                        ctx,
                        ordered_bgp,
                        self.rml_df,
                        self.config,
                        filter_sql=sql_frag,
                        bloom_filter=self.bloom_filter,
                        semijoin_reduction=self.semijoin_reduction,
                        semijoin_rounds=self.semijoin_rounds,
                    )

                    def _augment(
                        row: FrozenBindings,
                        _eq: dict = equalities,
                        _ctx: QueryContext = ctx,
                    ) -> FrozenBindings:
                        """Re-inject equality-substituted bindings dropped by B5-PRE."""
                        if not _eq:
                            return row
                        extra = {var: val for var, val in _eq.items()
                                 if row.get(var) is None}
                        if not extra:
                            return row
                        return FrozenBindings(_ctx, {**row._d, **extra})

                    if residual is None:
                        return (_augment(row) for row in inner)
                    return (
                        _augment(row) for row in inner
                        if _eval_filter_expr(residual, row, ctx)
                    )

                # ── Filter over LeftJoin: push down into p1 when safe ─────
                # SPARQL algebra rewrite rule (§18.6, Table 9):
                #   Filter(expr, LeftJoin(P1, P2, cond))
                #     ≡ LeftJoin(Filter(expr, P1), P2, cond)
                # when vars(expr) ∩ (vars(P2) \ vars(P1)) = ∅, i.e. no filter
                # variable is exclusively bound by the OPTIONAL clause.
                #
                # Pushing the filter into P1 allows B5-PRE equality substitution
                # and SQL-fragment pushdown to operate on P1's BGP normally,
                # rather than falling through to a Python post-filter over the
                # full LeftJoin result.
                if part.p.name == "LeftJoin":
                    lj        = part.p
                    p1_vars   = lj.p1._vars or set()
                    p2_vars   = lj.p2._vars or set()
                    opt_only  = p2_vars - p1_vars
                    expr_vars = _collect_filter_vars(part.expr)
                    if not (expr_vars & opt_only):
                        # Safe: all filter variables are bound in P1.
                        # Synthesise a new Filter node wrapping p1 and recurse.
                        from rdflib.plugins.sparql.parserutils import CompValue
                        pushed_filter = CompValue(
                            "Filter",
                            p=lj.p1,
                            expr=part.expr,
                            _vars=(p1_vars | (part.expr._vars or set())),
                        )
                        pushed_lj = CompValue(
                            "LeftJoin",
                            p1=pushed_filter,
                            p2=lj.p2,
                            expr=lj.expr,
                            _vars=lj._vars,
                        )
                        if _VIRT_DEBUG:
                            print(
                                f"[VIRT_DEBUG] Filter→LeftJoin pushdown: "
                                f"expr_vars={expr_vars}, opt_only={opt_only}"
                            )
                        return custom_eval(ctx, pushed_lj)

                # Non-BGP inner node (e.g. nested Join/Union) — no pushdown;
                # evaluate inner normally and apply the full SPARQL post-filter.
                inner = custom_eval(ctx, part.p)
                return (
                    row for row in inner
                    if _eval_filter_expr(part.expr, row, ctx)
                )

            # ── OPTIONAL (LeftJoin) ──────────────────────────────────────────
            if part.name == "LeftJoin":
                return _eval_left_join(ctx, part)

            # ── JOIN ─────────────────────────────────────────────────────────
            if part.name == "Join":
                return _eval_join(ctx, part)

            # ── UNION ────────────────────────────────────────────────────────
            if part.name == "Union":
                return _eval_union(ctx, part)

            # ── MINUS ────────────────────────────────────────────────────────
            if part.name == "Minus":
                return _eval_minus(ctx, part)

            # ── VALUES (ToMultiSet) ──────────────────────────────────────────
            if part.name == "ToMultiSet":
                return iter(_values_node_to_rows(ctx, part))

            # ── DISTINCT (B10) ───────────────────────────────────────────────
            if part.name == "Distinct":
                # B10 — If the inner node is a plain BGP, push DISTINCT into
                # the SQL query so the database engine deduplicates before any
                # data crosses the driver boundary.  For all other inner nodes
                # (Filter, Join, Union, …) fall back to Python-level
                # deduplication via a seen-set, which is still correct.
                if part.p.name == "BGP":
                    ordered = order_bgp(ctx, part.p.triples, self.rml_df)
                    rows = list(
                        virt_eval_bgp(
                            ctx, ordered, self.rml_df, self.config,
                            distinct=True,
                            bloom_filter=self.bloom_filter,
                        semijoin_reduction=self.semijoin_reduction,
                        semijoin_rounds=self.semijoin_rounds,
                        )
                    )
                    # Python-level dedup as correctness backstop for
                    # cross-rule duplicates (multiple rules producing the
                    # same RDF triple from different tables).
                    seen: set[frozenset] = set()
                    deduped: list[FrozenBindings] = []
                    for row in rows:
                        key = frozenset(
                            (k, str(v)) for k, v in row._d.items() if v is not None
                        )
                        if key not in seen:
                            seen.add(key)
                            deduped.append(row)
                    return iter(deduped)
                # Non-BGP inner — evaluate normally, deduplicate in Python
                inner_rows = list(custom_eval(ctx, part.p))
                seen_nb: set[frozenset] = set()
                deduped_nb: list[FrozenBindings] = []
                for row in inner_rows:
                    key = frozenset(
                        (k, str(v)) for k, v in row._d.items() if v is not None
                    )
                    if key not in seen_nb:
                        seen_nb.add(key)
                        deduped_nb.append(row)
                return iter(deduped_nb)

            # ── Everything else → RDFLib native ─────────────────────────────
            raise NotImplementedError()

        # ── OPTIONAL (LeftJoin) handler ───────────────────────────────────────

        def _eval_left_join(ctx: QueryContext, part: Any):
            left_rows = list(custom_eval(ctx, part.p1))
            if not left_rows:
                return

            left_df = _rows_to_bindings_df(left_rows)
            ordered_p2 = order_bgp(ctx, part.p2.triples, self.rml_df)
            right_rows = list(
                virt_eval_bgp(
                    ctx, ordered_p2, self.rml_df, self.config,
                    initial_bindings_df=left_df,
                    bloom_filter=self.bloom_filter,
                )
            )

            if not right_rows:
                # No OPTIONAL matches at all — yield all left rows unchanged.
                yield from left_rows
                return

            # OPT-1 — Hash-join for OPTIONAL: replace the O(|left|×|right|)
            # nested-loop compatibility check with an O(|left|+|right|) hash
            # lookup.
            #
            # Strategy:
            #  1. Find the variables bound in both left and right solution sets
            #     (the "join variables").
            #  2. Build a hash index: (join_var_values_tuple) → [right rows].
            #  3. For each left row, look up candidates in O(1) rather than
            #     scanning all right rows.
            #
            # When there are no shared variables the full Cartesian product is
            # correct per SPARQL semantics; we fall back to the nested loop only
            # in that degenerate case.
            left_bound  = {v for r in left_rows  for v, val in r._d.items() if val is not None}
            right_bound = {v for r in right_rows for v, val in r._d.items() if val is not None}
            join_vars   = sorted(left_bound & right_bound)  # sorted for stable key tuples

            if join_vars:
                # Build right-side hash index keyed by join variable values.
                right_index: defaultdict[tuple, list[FrozenBindings]] = defaultdict(list)
                for rr in right_rows:
                    key = tuple(rr._d.get(v) for v in join_vars)
                    right_index[key].append(rr)

                for left_row in left_rows:
                    key = tuple(left_row._d.get(v) for v in join_vars)
                    candidates = right_index.get(key, [])
                    matched = []
                    for right_row in candidates:
                        # Candidates share join-variable values by construction;
                        # _are_compatible handles any remaining variables.
                        if not _are_compatible(left_row, right_row):
                            continue
                        merged = _merge_bindings(ctx, left_row, right_row)
                        try:
                            if _ebv(part.expr, merged.forget(ctx)):
                                matched.append(merged)
                        except (SPARQLError, Exception):
                            pass
                    yield from matched if matched else [left_row]
            else:
                # No shared variables — degenerate Cartesian case; nested loop
                # is correct and unavoidable.
                for left_row in left_rows:
                    matched = []
                    for right_row in right_rows:
                        if not _are_compatible(left_row, right_row):
                            continue
                        merged = _merge_bindings(ctx, left_row, right_row)
                        try:
                            if _ebv(part.expr, merged.forget(ctx)):
                                matched.append(merged)
                        except (SPARQLError, Exception):
                            pass
                    yield from matched if matched else [left_row]

        # ── JOIN handler ──────────────────────────────────────────────────────

        def _eval_join(ctx: QueryContext, part: Any):
            left_rows = list(custom_eval(ctx, part.p1))
            if not left_rows:
                return

            if part.p2.name == "BGP":
                left_df = _rows_to_bindings_df(left_rows)
                ordered_p2 = order_bgp(ctx, part.p2.triples, self.rml_df)
                right_rows = list(
                    virt_eval_bgp(
                        ctx, ordered_p2, self.rml_df, self.config,
                        initial_bindings_df=left_df,
                        bloom_filter=self.bloom_filter,
                        semijoin_reduction=self.semijoin_reduction,
                        semijoin_rounds=self.semijoin_rounds,
                    )
                )
            else:
                right_rows = list(custom_eval(ctx, part.p2))

            if not right_rows:
                return

            # A2 — Hash-join for JOIN: apply the same O(|left|+|right|) strategy
            # used for OPTIONAL (OPT-1) to eliminate the O(|left|×|right|) nested
            # loop.  Build a hash index over right_rows keyed by the shared
            # variable values, then look up each left row in O(1).
            #
            # When there are no shared variables the result is a Cartesian product;
            # the nested loop is retained as the only correct strategy for that case.
            left_bound  = {v for r in left_rows  for v, val in r._d.items() if val is not None}
            right_bound = {v for r in right_rows for v, val in r._d.items() if val is not None}
            join_vars   = sorted(left_bound & right_bound)  # sorted for stable key tuples

            if join_vars:
                # Build right-side hash index: join-key tuple → [matching right rows]
                right_index: defaultdict[tuple, list[FrozenBindings]] = defaultdict(list)
                for rr in right_rows:
                    key = tuple(rr._d.get(v) for v in join_vars)
                    right_index[key].append(rr)

                for left_row in left_rows:
                    key = tuple(left_row._d.get(v) for v in join_vars)
                    for right_row in right_index.get(key, []):
                        # Candidates share join-variable values by construction;
                        # _are_compatible handles any remaining free variables.
                        if _are_compatible(left_row, right_row):
                            yield _merge_bindings(ctx, left_row, right_row)
            else:
                # No shared variables — Cartesian product is required by SPARQL semantics
                for left_row in left_rows:
                    for right_row in right_rows:
                        if _are_compatible(left_row, right_row):
                            yield _merge_bindings(ctx, left_row, right_row)

        # ── UNION handler ─────────────────────────────────────────────────────

        def _eval_union(ctx: QueryContext, part: Any):
            yield from custom_eval(ctx, part.p1)
            yield from custom_eval(ctx, part.p2)

        # ── MINUS handler ─────────────────────────────────────────────────────

        def _eval_minus(ctx: QueryContext, part: Any):
            """SPARQL MINUS per §18.5: remove left rows compatible with any right row,
            but only when the two patterns share at least one variable."""
            left_rows  = list(custom_eval(ctx, part.p1))
            right_rows = list(custom_eval(ctx, part.p2))

            if not right_rows:
                yield from left_rows
                return

            left_vars  = {v for row in left_rows  for v, val in row._d.items() if val is not None}
            right_vars = {v for row in right_rows for v, val in row._d.items() if val is not None}
            shared_vars = left_vars & right_vars

            for left_row in left_rows:
                if not shared_vars or not any(_are_compatible(left_row, r) for r in right_rows):
                    yield left_row

        return custom_eval

    # ── Mandatory Store interface (read-only) ─────────────────────────────────

    def triples(self, pattern: TriplePattern, context: Any = None) -> Iterable[tuple]:
        return iter([])

    def __len__(self, context: Any = None) -> int:
        return 0

    def add(self, _, context=None, quoted=False) -> None:
        raise TypeError("VIRTStore is read-only")

    def addN(self, quads) -> None:
        raise TypeError("VIRTStore is read-only")

    def remove(self, _, context=None) -> None:
        raise TypeError("VIRTStore is read-only")

    def create(self, configuration) -> None:
        raise TypeError("VIRTStore is read-only")

    def destroy(self, configuration) -> None:
        raise TypeError("VIRTStore is read-only")

    def commit(self) -> None:
        raise TypeError("VIRTStore is read-only")

    def rollback(self) -> None:
        raise TypeError("VIRTStore is read-only")
