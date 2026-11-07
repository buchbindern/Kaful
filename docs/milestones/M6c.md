# M6c. Follow-ups (batch polling retries, PLAN entries)

## The problem it solved
A batch run polls the API for its status until the batch is done. One dropped connection or
server error during that wait crashed the whole run, although the batch itself was fine.

## What was built
- **Polling retries:** batch status polling retries connection, timeout, server and
  rate-limit errors, up to `batch_poll_max_errors` (10) failures in a row. A successful poll
  resets the count. After that the run stops cleanly with `BatchPending`, the batch id saved,
  so a rerun collects the results without paying twice. Other API errors still fail at once.
- **PLAN.md:** M15 gains a Dockerfile for CI and deployment. Parked: container or microVM
  sandbox isolation, needed only if multi-tenant or unattended production.

## The most important files
- `kaful/llm/batch.py`: `wait_for_batch`, with the error count and the clean stop.
- `kaful/llm/anthropic_api.py`: which API errors count as transient.
- `kaful/llm/transport.py`: `TransientTransportError`.
- `tests/unit/test_kaful_llm_gateway.py`: the two polling tests, written first.

## How to see it working
- `pytest -q tests/unit/test_kaful_llm_gateway.py -k polling`.

## Checks
- The two new tests failed before the fix and pass after it.
- `pytest -q` 418 passed; mypy clean. $0 spent.

## Decisions you made
- Retry transient polling errors instead of crashing; stop cleanly after a run of failures.

## Open issues
- None.
