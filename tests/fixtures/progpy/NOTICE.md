# ProgPy excerpts: notice and modification record

`framework_hits.json` contains excerpts of **ProgPy** (Python Prognostics Packages,
NASA Ames Research Center; https://github.com/nasa/progpy and
https://nasa.github.io/progpy), which is released under the **NASA Open Source
Agreement version 1.3**. Copies of the agreement, as shipped with
`progpy==1.7.1`, are in `licenses/`. These excerpts are distributed under that
agreement; the rest of this repository is a Larger Work (NOSA §3I) and is not
governed by it.

Copyright © 2021 United States Government as represented by the Administrator of
the National Aeronautics and Space Administration. No copyright is claimed in the
United States under Title 17, U.S. Code. All Other Rights Reserved.

## Modification record (NOSA §3C)

- **Date:** 2026-11-02
- **Originator:** the Kaful project (owner of this repository)
- **Alterations:** for each of the 10 queries in `utils/progpy_rag.FRAMEWORK_QUERIES`,
  the top 4 chunks of a local index of ProgPy source docstrings, example files and
  documentation pages were selected by BM25 ranking (`scripts/freeze_progpy_hits.py`)
  and stored as JSON. Selection removes all other content; the selected chunk texts
  are not otherwise edited. One hit (`manual:ukf_usage_pattern`) is this project's
  own text from `rag_frameworks/progpy/progpy_rag/ingestion/manual_chunks.py`, not ProgPy.
- **Consent:** the originator consents to characterising these alterations as a
  Modification, derived directly from Original Software provided by Government Agency.

The excerpts are used only as test fixtures. Nothing here implies endorsement by
NASA (NOSA §3E).
