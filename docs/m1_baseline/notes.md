## Findings

1. **Phase 1 re-extraction dominates any run that rebuilds the index.**
   - eversys: $2.18, 37% of a cold run.
   - CNC: $15.85, 72% of a cold run, over 1,153 pages; the Fanuc manual alone is 750 of them.
   - Measured PDF input is about 1,940 tokens/page (eversys) and 2,140 (CNC). That is within the
     1,500–3,000 per page Anthropic documents.
   - CNC is not safe to rerun without `--from`: phase 1 would extract the 181 unindexed chunks, for
     about $12.79.
2. **6b component codegen is the largest cost of a phase 2–6 rerun on eversys.**
   - $1.44, 38% of the rerun; $0.81 at the low bound.
   - Each of the 8 calls re-sends the same ProgPy framework context, roughly 20–50k tokens.
   - That identical block is the obvious prompt-caching candidate for M5.
3. **2b and 2c re-send the same retrieved context 7 times (2 context calls + 5 extraction runs).**
   - eversys: about 12k tokens per call, $0.43 for the two steps together.
   - CNC: about 92k tokens per call, $2.17. CNC's saved 2a context is 281k characters (23 chunks
     averaging about 12k each), about 6× the eversys context.
4. **Per-component work costs about $0.19 (eversys) and $0.34 (CNC) per in-scope component**,
   summed over 4b, 6a and 6b. 4b also runs for components that triage later drops (eversys: 4 of
   12, about $0.16). Moving scope before the per-component work (M8) removes that.
5. **PLAN M1's expected drivers mostly hold:**
   - eversys cold: phase 1, 6b, 4b;
   - CNC cold: phase 1, 2c, 6b.
   - 2c ranks only 3rd on eversys reruns, but it is the top rerun cost on CNC because of the large
     context.
6. **Embeddings are negligible:** under $0.01 per run in every scenario.

## Scenarios

- **Cold run:** phase 1 extracts every chunk of every manual in `queries.manuals` (fresh index or
  `--force`), then phases 2–6 run with no checkpoints.
- **Rerun `--from phase2`:** the index is kept, and phases 2–6 run with their checkpoints cleared.
- **Rerun with checkpoints intact:** every LLM step loads from disk. The only remaining API traffic
  is the 10 ProgPy query embeddings in 6b, because `get_framework_context()` runs before 6b's cache
  check. That costs well under $0.001.
- **Phase 7** makes no API calls.

## Method

**Phase 1.** Each chunk request is rebuilt exactly as `utils/rag.py` sends it: the PDF slice as
a base64 `document` block plus the extraction prompt. Its input is measured with `count_tokens`.
The output is the stored extraction text in Chroma, also measured with `count_tokens`.
- eversys: the 75 chunk PDFs behind the live index (built 2026-05-15).
- cnc_spindle: the chunks a cold run would cut today, produced by the real chunker.

**Phases 2–6.** These are replayed by the pipeline's own step code against a scratch copy of the
outputs, with `call_claude` replaced by a stub. The stub captures each exact request (system, user
message, model, `max_tokens`, temperature) and returns the saved output. Input tokens are measured
with `count_tokens` on the captured request. How exact the output tokens are depends on the step:

| Output basis | Steps | Accuracy |
|---|---|---|
| exact | 2c (raw responses are saved) | exact |
| saved code | 3c, 6b, 6d composite | code fences are stripped before saving, so a few tokens low |
| reformatted | 2b, 3b, 4a, 4b, 5a, 5b, 6a, 6d nominal | the parsed JSON, re-dumped with `indent=2`; the model's own whitespace is unknown, so roughly ±10% |
| estimated | 4c | units and confidences are saved, but each 120-character reasoning string is assumed |

**Stand-in retrieval.** These retrieved chunks were never saved:
- 4a: `k=4` per comprehension query;
- 4b: 3 queries × `k=3` per component;
- 4c: `k=3` per missing-unit signal;
- the ProgPy context: 10 queries × `k=4`, plus the hand-written rules.

They are re-created with a local BM25 ranking over the same Chroma stores. Querying the real
stores would need OpenAI embedding calls, which this baseline does not make. BM25 favours long
chunks: on the ProgPy store its picks average about 5.1k characters, against a store mean of
2.4k and a median of 0.6k. The tables therefore give the stand-in figure and a "low" figure that
assumes the same number of chunks at the store's mean size.

**Token counts and embeddings.**
- **Anthropic counts:** `count_tokens` on `claude-sonnet-4-6` / `claude-opus-4-6`. The docs say
  the count "is an estimate. In some cases, the actual number of input tokens used when creating a
  message might differ by a small amount."
- **Embeddings:** tokens are taken as characters ÷ 4. They are under $0.01 per run on every
  scenario.

**Not included.**
- **Rejected phase-1 extractions:** they are billed but were never logged; the new ledger now
  records them.
- **429/529 retries:** they are not billed.
- **6b syntax retries:** these are shown separately as "+6b if all retries".

## Reliability

**eversys_coffee_machine**
- Phase 1 is measured end to end: inputs with `count_tokens` on the real PDF slices, outputs from
  the stored extractions.
- **Chunker drift.** Today's chunker cuts the manual into 70 chunks (157 pages), not 75 (162
  pages). A cold run today would therefore cost slightly less in phase 1 than the figure shown.
- **Main uncertainty:** 4a/4b/4c and the 6b/6d ProgPy context, which use the stand-in retrieval
  (see "low").

**cnc_spindle.** The index has drifted, so several of these numbers are unreliable.
- **Phase-1 output for ISO 10816 and the Fanuc manual is estimated.**
  - The live index holds only 48 chunks, all MTConnect (20 from part 1 and 28 of part 2's 46).
  - The Fanuc and ISO extractions from the May run are gone, so their output tokens are estimated
    from the output/input ratio of the chunks whose extraction is stored. The phase-1 range column
    shows the spread.
  - Phase-1 *input* is measured for all 229 chunks.
- **4a, 4b and 4c inputs are unreliable.**
  - The stand-in retrieval can only search today's 48-chunk MTConnect index.
  - The May run that produced the saved 4a/4b outputs searched an index that also held the Fanuc
    manual and ISO 10816.
  - The prompt sizes are the right order of magnitude, but the chunks are wrong.
- **2a/3a contexts are exact for the May run, but a cold run today would retrieve different
  chunks.**
  - The saved contexts are what was sent then.
  - A cold run today would first rebuild the index from all four manuals (Fanuc is 750 of the
    1,153 pages), then retrieve from it.
- **The phase-1 cost of a plain rerun (no `--from`) is not $0.** Phase 1 always runs without
  `--from`, and it extracts every chunk whose ID is not already indexed. The IDs line up for the 48
  MTConnect chunks, so roughly the remaining 181 chunks (MTConnect part 2's tail, ISO and Fanuc)
  would be extracted.
- **Manual list:** `queries.py` lists `m110_03_safety.pdf` (not on disk; it is skipped) and omits
  `nsk_bearing_selection_mounting.pdf`, as OVERVIEW §5 notes.
- **Phases 2–6 outputs:** taken from the May checkpoints (8 components in 4b, 7 in 6a/6b).

## Sources (verified 2026-11-02)

- Anthropic pricing: https://platform.claude.com/docs/en/about-claude/pricing.md
  - Sonnet 4.6: $3 input / $15 output per MTok.
  - Opus 4.6: $5 / $25.
  - Haiku 4.5: $1 / $5.
  - Cache writes ×1.25 (5-minute) or ×2 (1-hour); cache reads ×0.1; Batch API 50% off.
  - No long-context surcharge for the 4.6 models.
- Anthropic PDF support: https://platform.claude.com/docs/en/build-with-claude/pdf-support.md
  - "Each page typically uses 1,500–3,000 tokens", plus image costs, because each page is
    converted to an image.
- Anthropic token counting: https://platform.claude.com/docs/en/build-with-claude/token-counting.md
  - "Token counting is free to use but subject to requests per minute rate limits."
  - It supports base64 PDFs.
- OpenAI pricing: https://developers.openai.com/api/docs/pricing (the redirect target of
  https://platform.openai.com/docs/pricing)
  - text-embedding-3-large: $0.13 per 1M tokens.
  - text-embedding-3-small: $0.02 per 1M tokens.
