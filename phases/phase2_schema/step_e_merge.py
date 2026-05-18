"""
phases/phase2_schema/step_e_merge.py
--------------------------------------
Step E: Merge duplicate fields across runs.

1. Count how often each field appears across runs
2. Build a representative object for each unique field
3. Find and merge fields that mean the same thing (water_temp, temp_water)
   using Jaccard + string similarity with antonym-aware conflict detection

Output saved to: outputs/schema/step_e_merged.json
"""

import json
import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from itertools import combinations


# ── Config ────────────────────────────────────────────────────────────────────

AUTO_MERGE_THRESHOLD = 0.90
REVIEW_THRESHOLD     = 0.55
CONFIRM_THRESHOLD    = 0.62
MIN_SHARED_WORDS     = 1

ANTONYM_GROUPS = [
    {"start", "end", "stop"},
    {"started", "completed"},
    {"open", "closed"},
    {"min", "max"},
    {"low", "high"},
    {"before", "after"},
    {"in", "out"},
    {"input", "output"},
]


# ── Public entry point ────────────────────────────────────────────────────────

def run(cfg: dict, normalized_runs: list[list[dict]]) -> dict:
    """
    Merge duplicate fields across all runs.

    Args:
        cfg:             result of get_machine_config()
        normalized_runs: list of normalized runs from step_d

    Returns:
        dict with keys:
            canonical_map  — {original_name: canonical_name}
            groups         — {canonical_name: [original_names]}
            representatives — {canonical_name: full field dict}
            field_counts   — {field_name: count across runs}
    """
    output_path = cfg["step_e_merged"]

    # Load from disk if already done
    if output_path.exists():
        print("  ✓ step_e already done — loading from disk")
        with open(output_path) as f:
            return json.load(f)

    print("  Running step_e — merging duplicate fields...")

    # Count fields across runs
    field_counts    = _build_field_counts(normalized_runs)
    representatives = _build_representatives(normalized_runs)

    print(f"    Unique fields before merge: {len(field_counts)}")

    # Find and apply merges
    canonical_map, groups, candidates = _apply_merges(field_counts)

    # Count merges
    merged = {k: v for k, v in groups.items() if len(v) > 1}
    print(f"    Merged groups: {len(merged)}")
    for canonical, members in merged.items():
        print(f"      {canonical} ← {members}")

    print(f"    Unique fields after merge: {len(groups)}")

    # Map representatives to canonical names
    canonical_representatives = {}
    for canonical in groups:
        if canonical in representatives:
            canonical_representatives[canonical] = representatives[canonical]
        else:
            # Find first available member representative
            for member in groups[canonical]:
                if member in representatives:
                    canonical_representatives[canonical] = representatives[member]
                    break

    result = {
        "canonical_map":     canonical_map,
        "groups":            groups,
        "representatives":   canonical_representatives,
        "field_counts":      dict(field_counts),
    }

    # Save to disk
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"    Saved → {output_path.name}")

    return result


# ── Field counting ────────────────────────────────────────────────────────────

def _build_field_counts(normalized_runs: list[list[dict]]) -> Counter:
    """Count each field once per run."""
    counts = Counter()

    for run in normalized_runs:
        seen_this_run = set()
        for field in run:
            name = field["normalized_name"]
            if name and name not in seen_this_run:
                counts[name] += 1
                seen_this_run.add(name)

    return counts


def _build_representatives(normalized_runs: list[list[dict]]) -> dict:
    """Keep one representative full field object per normalized name."""
    representatives = {}

    for run in normalized_runs:
        for field in run:
            name = field["normalized_name"]
            if name and name not in representatives:
                representatives[name] = field

    return representatives


# ── Similarity ────────────────────────────────────────────────────────────────

def _normalize_field_name(name: str) -> str:
    if not isinstance(name, str):
        return name
    name = name.lower().strip()
    name = re.sub(r"[\s\-]+", "_", name)
    name = re.sub(r"[^a-z0-9_]", "", name)
    name = re.sub(r"_+", "_", name)
    return name.strip("_")


def _tokenize(name: str) -> list[str]:
    normalized = _normalize_field_name(name)
    return normalized.split("_") if normalized else []


def _has_conflicting_tokens(name1: str, name2: str) -> bool:
    t1 = set(_tokenize(name1))
    t2 = set(_tokenize(name2))
    for group in ANTONYM_GROUPS:
        if (t1 & group) and (t2 & group) and (t1 & group) != (t2 & group):
            return True
    return False


def _jaccard(tokens1: list, tokens2: list) -> float:
    s1, s2 = set(tokens1), set(tokens2)
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    return len(s1 & s2) / len(s1 | s2)


def _string_similarity(name1: str, name2: str) -> float:
    return SequenceMatcher(None, name1, name2).ratio()


def _field_similarity(name1: str, name2: str) -> float:
    t1 = _tokenize(name1)
    t2 = _tokenize(name2)
    return 0.6 * _jaccard(t1, t2) + 0.4 * _string_similarity(
        _normalize_field_name(name1),
        _normalize_field_name(name2)
    )


def _classify(name1: str, name2: str) -> str:
    if _has_conflicting_tokens(name1, name2):
        return "ignore"

    shared = set(_tokenize(name1)) & set(_tokenize(name2))
    score  = _field_similarity(name1, name2)

    if score >= AUTO_MERGE_THRESHOLD:
        return "auto_merge"
    elif score >= REVIEW_THRESHOLD and len(shared) >= MIN_SHARED_WORDS:
        return "review"
    return "ignore"


# ── Union-Find ────────────────────────────────────────────────────────────────

class _UnionFind:
    def __init__(self, items):
        self.parent = {item: item for item in items}
        self.rank   = {item: 0 for item in items}

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1


def _pick_canonical(names: list, field_counts: dict) -> str:
    """Pick the most observed name, breaking ties by preferring shorter names."""
    return max(names, key=lambda n: (field_counts.get(n, 0), -len(n)))


# ── Merge ─────────────────────────────────────────────────────────────────────

def _apply_merges(field_counts: Counter) -> tuple:
    """Find and apply merges, returning canonical_map, groups, candidates."""
    field_names = list(field_counts.keys())
    candidates  = []

    for f1, f2 in combinations(field_names, 2):
        label = _classify(f1, f2)
        if label == "ignore":
            continue

        score = _field_similarity(f1, f2)
        candidates.append({
            "field1":          f1,
            "field2":          f2,
            "score":           round(score, 3),
            "classification":  label,
            "shared_words":    sorted(set(_tokenize(f1)) & set(_tokenize(f2))),
        })

    uf = _UnionFind(field_counts.keys())

    for c in candidates:
        if c["classification"] == "auto_merge":
            uf.union(c["field1"], c["field2"])
        elif c["classification"] == "review" and c["score"] >= CONFIRM_THRESHOLD:
            uf.union(c["field1"], c["field2"])

    groups = defaultdict(list)
    for name in field_counts:
        groups[uf.find(name)].append(name)

    canonical_map  = {}
    final_groups   = {}
    for root, members in groups.items():
        canonical = _pick_canonical(members, field_counts)
        final_groups[canonical] = sorted(members)
        for name in members:
            canonical_map[name] = canonical

    return canonical_map, final_groups, candidates