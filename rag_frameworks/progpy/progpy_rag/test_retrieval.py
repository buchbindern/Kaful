"""
test_retrieval.py

Run this after ingestion to validate retrieval quality.
Tests the exact queries your pipeline will use in production.

Usage:
    python test_retrieval.py
"""

from dotenv import load_dotenv
load_dotenv()

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from vector_store import ChromaDBStore

store = ChromaDBStore(persist_dir="./data/chroma", openai_api_key=os.environ.get("OPENAI_API_KEY"))

print(f"Collection has {store.count()} chunks\n")
print("=" * 60)

# These simulate what your pipeline will actually query
TEST_CASES = [
    {
        "description": "Per-component generation: thermal component",
        "query": "thermal degradation model state equations progpy",
        "filter": {"framework": "progpy"},
    },
    {
        "description": "Per-component generation: wear degradation",
        "query": "wear degradation component model grinder mechanical",
        "filter": {"framework": "progpy", "pattern": "component"},
    },
    {
        "description": "Integration step: wiring components together",
        "query": "CompositeModel connect components signals",
        "filter": {"framework": "progpy", "pattern": "composite"},
    },
    {
        "description": "Parameter estimation from real data",
        "query": "estimate_params fitting model to sensor data",
        "filter": {"framework": "progpy"},
    },
    {
        "description": "Running simulation to threshold / EOL prediction",
        "query": "simulate_to_threshold end of life prediction",
        "filter": {"framework": "progpy", "pattern": "degradation"},
    },
    {
        "description": "State estimator for online tracking",
        "query": "particle filter kalman state estimation",
        "filter": {"framework": "progpy"},
    },
]

for case in TEST_CASES:
    print(f"\n🔍 {case['description']}")
    print(f"   Query:  {case['query']}")
    print(f"   Filter: {case['filter']}")

    results = store.query(case["query"], filter=case["filter"], top_k=3)

    for i, chunk in enumerate(results):
        score = chunk.metadata.get("_score", "?")
        name = chunk.metadata.get("name", "?")
        source = chunk.metadata.get("source", "?")
        ctype = chunk.metadata.get("type", "?")
        domain = chunk.metadata.get("domain", "?")
        print(f"\n   [{i+1}] score={score:.3f} | {ctype} | domain={domain}")
        print(f"        name: {name}")
        print(f"        source: {source}")
        print(f"        preview: {chunk.text[:120].replace(chr(10), ' ')}...")