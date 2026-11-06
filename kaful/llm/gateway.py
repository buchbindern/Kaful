"""The LLM gateway: the only way pipeline code calls a model.

For every call it:
1. renders the prompt into a request (cached segments get `cache_control`; the
   output model becomes `output_config.format`; model and effort come from config);
2. answers from the response cache when it can (cost $0, ledger `response_cache=hit`);
3. otherwise checks the run budget against the call's worst case *before* sending,
   sends it through the transport, charges the actual cost and stores the response;
4. validates the reply against the prompt's output model and, on failure, retries
   once with the validation errors appended; a second failure raises LLMOutputError;
5. writes one ledger row per attempt.

`run_many` does the same for many calls; in batch mode the uncached requests go out
as one Message Batch (50% price), and results are matched back by custom_id.
"""

import base64
import json
import logging
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ValidationError

from kaful.config import LLMSettings, ModelChoice, ReuseRule
from kaful.llm import batch as batches
from kaful.llm.anthropic_api import output_format
from kaful.llm.budget import BudgetExceeded, RunBudget, estimate_input_tokens
from kaful.llm.cache import ResponseStore, StoredRecord, StoredResponse, redact_request, usage_fields
from kaful.llm.hashing import prompt_hash
from kaful.llm.ledger import LEDGER_SCHEMA_VERSION, Ledger, LedgerRow, ResponseCacheUse, Validation
from kaful.llm.pricing import CacheTtl, PricingTable, TokenUsage, load_pricing
from kaful.llm.transport import Completion, Transport, TransportError
from kaful.prompts.registry import Prompt, PromptError, SegmentName

log = logging.getLogger("kaful.llm.gateway")

CACHE_CONTROL: dict[str, dict[str, str]] = {
    "5m": {"type": "ephemeral"},
    "1h": {"type": "ephemeral", "ttl": "1h"},
}
END_TURN = "end_turn"
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
MAX_ERROR_LINES = 20
RETRY_MESSAGE = (
    "Your previous reply did not validate against the required output schema:\n{errors}\n\n"
    "Reply again with the complete corrected JSON object only."
)


class LLMOutputError(RuntimeError):
    """The model's reply was unusable after the one allowed retry."""

    def __init__(self, message: str, last_text: str | None = None) -> None:
        super().__init__(message)
        self.last_text = last_text


class OfflineMissError(LookupError):
    """No cached response and no transport to ask (offline gateway)."""


@dataclass(frozen=True)
class CallScope:
    """Where a call belongs: ledger attribution and the artifact it feeds."""

    phase: str
    step: str
    component: str | None = None
    artifact: str | None = None


@dataclass(frozen=True)
class Call:
    prompt: Prompt
    inputs: Mapping[str, str]
    scope: CallScope
    sample: int = 0
    images: Mapping[str, bytes] = field(default_factory=dict)  # PNG bytes by declared image name
    choice: ModelChoice | None = None  # overrides the configured model choice (an escalation)
    route: str | None = None  # why this call runs on this model, for the ledger


@dataclass(frozen=True)
class LLMResult:
    output: BaseModel
    text: str
    prompt_sha256: str
    attempts: int
    cost_usd: float
    reused_from: tuple[int, str] | None = None  # (older version, its request's prompt_sha256) when reused


@dataclass
class _Obtained:
    """A response plus how it was obtained, logged once its validation is known."""

    completion: Completion
    use: ResponseCacheUse
    cost_usd: float
    unbilled_usd: float | None
    estimated_usd: float | None = None
    latency_ms: int | None = None
    batch_id: str | None = None
    custom_id: str | None = None
    reused_from: tuple[int, str] | None = None


def image_block(png: bytes) -> dict[str, Any]:
    if not png.startswith(PNG_SIGNATURE):
        raise PromptError("image attachments must be PNG")
    return {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                        "data": base64.b64encode(png).decode("ascii")}}


def parse_output(model: type[BaseModel], text: str) -> BaseModel:
    """Validate a reply; tolerate a ```json fence around it."""
    body = text.strip()
    if body.startswith("```"):
        body = body.split("\n", 1)[1] if "\n" in body else ""
        body = body.rsplit("```", 1)[0]
    return model.model_validate_json(body)


def describe_errors(exc: ValidationError) -> str:
    lines = [f"- {'.'.join(str(p) for p in e['loc']) or '(root)'}: {e['msg']}" for e in exc.errors()]
    more = len(lines) - MAX_ERROR_LINES
    return "\n".join(lines[:MAX_ERROR_LINES] + ([f"- ... {more} more"] if more > 0 else []))


@dataclass
class Gateway:
    settings: LLMSettings
    ledger: Ledger
    budget: RunBudget
    transport: Transport | None
    cache: ResponseStore | None
    batch_state_dir: Path | None = None
    run_id: str | None = None
    machine_id: str | None = None
    pricing: PricingTable = field(default_factory=load_pricing)
    sleep: Callable[[float], None] = time.sleep
    clock: Callable[[], float] = time.monotonic
    now: Callable[[], datetime] = lambda: datetime.now(UTC)
    prompt_versions: dict[str, int] = field(default_factory=dict)
    _fresh: dict[tuple[str, int], _Obtained] = field(default_factory=dict)
    _token_counts: dict[str, int] = field(default_factory=dict)

    # ── Requests ──────────────────────────────────────────────────────────────

    def request_for(self, prompt: Prompt, inputs: Mapping[str, str],
                    images: Mapping[str, bytes] | None = None,
                    choice: ModelChoice | None = None) -> dict[str, Any]:
        """The Messages API request for one prompt rendering (images go just before the task).
        `choice` overrides the configured model choice."""
        choice = choice or self.settings.choice(prompt.id)
        given = sorted(images or {})
        if given != list(prompt.spec.images):
            raise PromptError(f"{prompt.id}: images {given} != declared {list(prompt.spec.images)}")
        rendered = prompt.render(inputs)
        segments = prompt.spec.segments

        def block(text: str, name: SegmentName) -> dict[str, Any]:
            if not text.strip():
                raise PromptError(f"{prompt.id}: the {name} segment rendered empty")
            cache = segments[name].cache
            return {"type": "text", "text": text, **({"cache_control": CACHE_CONTROL[cache]} if cache != "none" else {})}

        content = [block(rendered.context, "context")] if rendered.context is not None else []
        content += [image_block(images[name]) for name in prompt.spec.images] if images else []
        content.append(block(rendered.task, "task"))
        output_config: dict[str, Any] = {"format": output_format(prompt.output_model)}
        if choice.effort is not None:
            output_config["effort"] = choice.effort
        return {
            "model": choice.model,
            "max_tokens": prompt.spec.max_tokens,
            "system": [block(rendered.system, "system")],
            "messages": [{"role": "user", "content": content}],
            "output_config": output_config,
        }

    @staticmethod
    def retry_request(request: dict[str, Any], reply: str, errors: str) -> dict[str, Any]:
        messages = [*request["messages"], {"role": "assistant", "content": reply},
                    {"role": "user", "content": RETRY_MESSAGE.format(errors=errors)}]
        return {**request, "messages": messages}

    # ── One call ──────────────────────────────────────────────────────────────

    def call(self, call: Call) -> LLMResult:
        """Run one call (cache, budget, transport, validation, one retry)."""
        request = self.request_for(call.prompt, call.inputs, call.images, call.choice)
        result, retry, cost = self._attempt(call, request, attempt=1)
        if result is not None:
            return result
        assert retry is not None
        result, second, _ = self._attempt(call, retry, attempt=2, prior_cost=cost)
        if result is None:
            last = second["messages"][-2]["content"] if second else None
            raise LLMOutputError(f"{call.prompt.id} ({call.scope.step}): reply invalid after one retry", last)
        return result

    def _attempt(
        self, call: Call, request: dict[str, Any], attempt: int, prior_cost: float = 0.0
    ) -> tuple[LLMResult | None, dict[str, Any] | None, float]:
        """One attempt: obtain, validate, log.

        Returns (result, None, cost so far) or (None, retry request, cost so far)."""
        phash = prompt_hash(request)
        obtained = self._obtain(call, request, phash, attempt)
        cost = prior_cost + obtained.cost_usd
        completion = obtained.completion
        if completion.stop_reason != END_TURN:
            self._log(call, request, phash, obtained, attempt, "invalid", status="error",
                      error=f"stop_reason={completion.stop_reason}")
            raise LLMOutputError(f"{call.prompt.id}: stop_reason {completion.stop_reason!r}", completion.text)
        try:
            output = parse_output(call.prompt.output_model, completion.text)
        except ValidationError as exc:
            self._log(call, request, phash, obtained, attempt, "invalid", status="retry")
            return None, self.retry_request(request, completion.text, describe_errors(exc)), cost
        self._log(call, request, phash, obtained, attempt, "valid", status="ok")
        return LLMResult(output, completion.text, phash, attempt, cost, obtained.reused_from), None, cost

    def _obtain(self, call: Call, request: dict[str, Any], phash: str, attempt: int) -> _Obtained:
        """The response for `request`: a fresh batch result, the cache, or the transport."""
        self.prompt_versions[call.prompt.id] = call.prompt.version
        fresh = self._fresh.pop((phash, call.sample), None)
        if fresh is not None:
            return fresh
        stored = self._cached(call, phash)
        if stored is not None:
            completion = stored.completion()
            return _Obtained(completion, "hit", 0.0, self._cost(completion))
        reused = self._reused(call, request)
        if reused is not None:
            completion = reused[0].completion()
            return _Obtained(completion, "hit", 0.0, self._cost(completion), reused_from=reused[1])
        if self.transport is None:
            raise OfflineMissError(f"no cached response for {call.prompt.id} v{call.prompt.version} "
                                   f"({call.scope.step}) prompt_sha256={phash} sample={call.sample}")
        estimated = self.worst_case(call.prompt, request, batch=False)
        what = f"{call.prompt.id} ({call.scope.phase}/{call.scope.step})"
        try:
            self.budget.check(estimated, what)
        except BudgetExceeded as exc:
            self._log(call, request, phash, _nothing(request, estimated), attempt, "not_checked",
                      status="refused_budget", error=str(exc))
            raise
        started = self.clock()
        try:
            completion = self.transport.send(request, self.settings.timeout_for(call.prompt.id))
        except TransportError as exc:
            self._log(call, request, phash, _nothing(request, estimated), attempt, "not_checked",
                      status="error", error=str(exc))
            raise
        latency = round((self.clock() - started) * 1000)
        return self._accept(call, request, phash, completion, estimated, latency)

    def _accept(self, call: Call, request: dict[str, Any], phash: str, completion: Completion,
                estimated: float | None, latency: int | None, batch_id: str | None = None,
                custom_id: str | None = None) -> _Obtained:
        """Charge a response from the transport and store it in the cache."""
        cost = self._cost(completion, batch=batch_id is not None) if completion.billed else 0.0
        self.budget.charge(cost)
        if completion.stop_reason == END_TURN:
            self._store(call, request, phash, completion, "batch" if batch_id else "recorded")
        use: ResponseCacheUse = "miss" if completion.billed else "replay"
        unbilled = None if completion.billed else self._cost(completion)
        return _Obtained(completion, use, cost, unbilled, estimated, latency, batch_id, custom_id)

    def _reused(self, call: Call, request: dict[str, Any]) -> tuple[StoredResponse, tuple[int, str]] | None:
        """An older version's response under the prompt's ReuseRule: the same request with the
        older max_tokens, stored, and ended normally. None if there is no such response."""
        rule: ReuseRule | None = self.settings.reuse.get(call.prompt.id)
        if rule is None or self.cache is None or not self.settings.cache_reads or len(request["messages"]) != 1:
            return None  # retry requests (with a reply and the errors) are never reused
        older = prompt_hash({**request, "max_tokens": rule.max_tokens})
        stored = self.cache.get(call.prompt.id, rule.version, older, call.sample)
        if stored is None or stored.stop_reason != END_TURN:
            return None
        return stored, (rule.version, older)

    def _cached(self, call: Call, phash: str) -> StoredResponse | None:
        if self.cache is None or not self.settings.cache_reads:
            return None
        return self.cache.get(call.prompt.id, call.prompt.version, phash, call.sample)

    def _store(self, call: Call, request: dict[str, Any], phash: str, completion: Completion,
               source: Literal["recorded", "batch"]) -> None:
        if self.cache is None:
            return
        existing = self.cache.record(call.prompt.id, call.prompt.version, phash)
        if (len(existing.responses) if existing else 0) != call.sample:
            return  # an earlier sample is missing (or this one is stored); keep the order intact
        record = StoredRecord(
            prompt_sha256=phash, prompt_id=call.prompt.id, prompt_version=call.prompt.version,
            phase=call.scope.phase, step=call.scope.step, component=call.scope.component,
            source=source, request=redact_request(request),
        )
        response = StoredResponse(text=completion.text, model=completion.model, stop_reason=completion.stop_reason,
                                  usage=usage_fields(completion.usage), recorded_at=self._ts())
        self.cache.put(record, call.sample, response)

    # ── Cost ─────────────────────────────────────────────────────────────────

    def _cost(self, completion: Completion, batch: bool = False) -> float:
        return self.pricing.cost(completion.model, completion.usage, batch)

    def input_tokens(self, request: dict[str, Any]) -> int:
        """count_tokens when configured and possible (cached by hash), else the chars estimate."""
        if self.settings.token_estimate == "count_tokens" and self.transport and self.transport.can_count_tokens:
            phash = prompt_hash(request)
            if phash not in self._token_counts:
                try:
                    self._token_counts[phash] = self.transport.count_tokens(request)
                except TransportError as exc:
                    log.warning("count_tokens failed; using the chars estimate", extra={"fields": {"error": str(exc)}})
                    return estimate_input_tokens(request, self.settings)
            return self._token_counts[phash]
        return estimate_input_tokens(request, self.settings)

    def worst_case(self, prompt: Prompt, request: dict[str, Any], batch: bool) -> float:
        ttls = {seg.cache for seg in prompt.spec.segments.values()}
        cache_write: CacheTtl | None = "1h" if "1h" in ttls else "5m" if "5m" in ttls else None
        return self.pricing.worst_case_cost(request["model"], self.input_tokens(request),
                                            request["max_tokens"], cache_write, batch)

    # ── Many calls / batches ──────────────────────────────────────────────────

    def run_many(self, calls: Sequence[Call], mode: Literal["sync", "batch"] | None = None) -> list[LLMResult]:
        """Run calls in order. Batch mode (default: when every prompt says `batch`) sends
        uncached requests as batches sliced to fit the budget, then one retry round."""
        use_batch = mode == "batch" or (mode is None and all(c.prompt.spec.mode == "batch" for c in calls))
        if not use_batch:
            return [self.call(c) for c in calls]
        pending = {i: self.request_for(c.prompt, c.inputs, c.images, c.choice) for i, c in enumerate(calls)}
        try:
            return self._run_batched(calls, pending)
        except BaseException:
            self._ledger_unconsumed(calls, pending)
            raise

    def _ledger_unconsumed(self, calls: Sequence[Call], requests: Mapping[int, dict[str, Any]]) -> None:
        """Write a ledger row for every collected (and charged) batch result the run never got
        to validate, e.g. because a later slice was refused: a paid call is always ledgered."""
        for i in sorted(requests):
            phash = prompt_hash(requests[i])
            obtained = self._fresh.pop((phash, calls[i].sample), None)
            if obtained is not None:
                self._log(calls[i], requests[i], phash, obtained, 1, "not_checked", status="ok",
                          error="collected but not validated: the run stopped first (response cached if complete)")

    def _run_batched(self, calls: Sequence[Call], pending: dict[int, dict[str, Any]]) -> list[LLMResult]:
        results: dict[int, LLMResult] = {}
        costs = dict.fromkeys(pending, 0.0)
        for attempt in (1, 2):
            self._prefetch([(calls[i], pending[i]) for i in sorted(pending)])
            next_pending: dict[int, dict[str, Any]] = {}
            for i in sorted(pending):
                if not self._available(calls[i], pending[i]):
                    next_pending[i] = pending[i]  # errored in the batch: resend as is
                    continue
                result, retry, costs[i] = self._attempt(calls[i], pending[i], attempt, costs[i])
                if result is not None:
                    results[i] = result
                elif retry is not None:
                    next_pending[i] = retry
            pending.clear()
            pending.update(next_pending)  # in place: run_many ledgers what a stop leaves unconsumed
            if not pending:
                return [results[i] for i in range(len(calls))]
        failed = ", ".join(f"#{i} {calls[i].prompt.id}/{calls[i].scope.component or '-'}" for i in sorted(pending))
        raise LLMOutputError(f"batch calls still failing after one retry round: {failed}")

    def _available(self, call: Call, request: dict[str, Any]) -> bool:
        phash = prompt_hash(request)
        return ((phash, call.sample) in self._fresh or self._cached(call, phash) is not None
                or self._reused(call, request) is not None)

    def _prefetch(self, pairs: Sequence[tuple[Call, dict[str, Any]]]) -> None:
        """Send the requests that are neither cached nor fresh; collect them into `_fresh`.

        Batches already submitted by an earlier run are collected first. The rest go out in
        slices, in custom-id order: each slice is the longest run of requests whose summed
        worst case fits the remaining budget, so a large job runs under a small budget once
        earlier slices have been charged at their actual cost. A request that does not fit
        on its own raises BudgetExceeded (nothing more is sent; collected slices stay cached).
        """
        misses: dict[str, tuple[Call, dict[str, Any], str]] = {}
        for call, request in pairs:
            phash = prompt_hash(request)
            if not self._available(call, request):
                misses.setdefault(batches.custom_id(phash, call.sample), (call, request, phash))
        if not misses:
            return
        if self.transport is None or self.batch_state_dir is None:
            raise OfflineMissError(f"{len(misses)} batch request(s) not cached and no transport/state dir")
        worst = {c: self.worst_case(misses[c][0].prompt, misses[c][1], batch=True) for c in sorted(misses)}
        remaining = sorted(misses)
        for state in batches.saved_states(self.batch_state_dir):
            if set(state.custom_ids) <= set(remaining):
                self._collect(list(state.custom_ids), misses, worst)
                remaining = [c for c in remaining if c not in state.custom_ids]
        while remaining:
            size = _fitting_prefix([worst[c] for c in remaining], self.budget.remaining_usd)
            self._collect(remaining[:max(size, 1)], misses, worst)  # size 0: reserve() raises
            remaining = remaining[max(size, 1):]

    def _collect(self, ids: list[str], misses: Mapping[str, tuple[Call, dict[str, Any], str]],
                 worst: Mapping[str, float]) -> None:
        """Reserve one slice's worst case, submit (or resume) it, wait, and accept its results."""
        assert self.transport is not None and self.batch_state_dir is not None
        reserved = sum(worst[c] for c in ids)
        self.budget.reserve(reserved, f"batch of {len(ids)} request(s)")
        try:
            batch_id = self._submit_or_resume(ids, misses)
            batches.wait_for_batch(self.transport, batch_id, self.settings.batch_poll_s,
                                   self.settings.batch_max_wait_s, self.sleep, self.clock)
            items = self.transport.batch_results(batch_id)
        finally:
            self.budget.release(reserved)
        for item in sorted(items, key=lambda it: it.custom_id):
            call, request, phash = misses[item.custom_id]
            if item.completion is None:
                failed = _nothing(request, worst[item.custom_id], batch_id, item.custom_id)
                self._log(call, request, phash, failed, 1, "not_checked", status="error",
                          error=f"batch {item.outcome}: {item.error}")
                continue
            self._fresh[(phash, call.sample)] = self._accept(call, request, phash, item.completion,
                                                            worst[item.custom_id], None, batch_id, item.custom_id)
        batches.clear_state(self.batch_state_dir, ids)

    def _submit_or_resume(self, ids: list[str], misses: Mapping[str, tuple[Call, dict[str, Any], str]]) -> str:
        assert self.transport is not None and self.batch_state_dir is not None
        state = batches.find_state(self.batch_state_dir, ids)
        if state is not None:
            log.info("resuming batch", extra={"fields": {"batch_id": state.batch_id, "requests": len(ids)}})
            return state.batch_id
        batch_id = self.transport.submit_batch([(c, misses[c][1]) for c in ids])
        batches.save_state(self.batch_state_dir, batches.BatchState(batch_id=batch_id, custom_ids=tuple(ids),
                                                                    submitted_at=self._ts()))
        first = misses[ids[0]][0]
        self.ledger.append(LedgerRow(
            schema_version=LEDGER_SCHEMA_VERSION, ts=self._ts(), run_id=self.run_id, machine_id=self.machine_id,
            event="llm_batch_submit", phase=first.scope.phase, step=first.scope.step, provider="anthropic",
            status="submitted", batch_id=batch_id, n_texts=len(ids), cost_usd=0.0,
        ))
        return batch_id

    # ── Ledger ───────────────────────────────────────────────────────────────

    def _ts(self) -> str:
        return self.now().isoformat(timespec="milliseconds")

    def _log(self, call: Call, request: dict[str, Any], phash: str, obtained: _Obtained, attempt: int,
             validation: Validation, status: Literal["ok", "retry", "error", "refused_budget"],
             error: str | None = None) -> None:
        u = obtained.completion.usage
        self.ledger.append(LedgerRow(
            schema_version=LEDGER_SCHEMA_VERSION, ts=self._ts(), run_id=self.run_id, machine_id=self.machine_id,
            event="llm_call", phase=call.scope.phase, step=call.scope.step, component=call.scope.component,
            provider="anthropic", model=obtained.completion.model or request["model"], attempt=attempt,
            status=status, input_tokens=u.input_tokens, output_tokens=u.output_tokens,
            cache_creation_input_tokens=u.cache_write_5m_tokens + u.cache_write_1h_tokens,
            cache_creation_1h_tokens=u.cache_write_1h_tokens, cache_read_input_tokens=u.cache_read_tokens,
            cache_hit=u.cache_read_tokens > 0, cost_usd=obtained.cost_usd, pricing_as_of=self.pricing.as_of,
            latency_ms=obtained.latency_ms, max_tokens=request["max_tokens"], prompt_sha256=phash,
            request_chars=len(json.dumps(request["messages"], ensure_ascii=False)),
            response_chars=len(obtained.completion.text), error=error, prompt_id=call.prompt.id,
            prompt_version=call.prompt.version, sample=call.sample, response_cache=obtained.use,
            validation=validation, estimated_cost_usd=obtained.estimated_usd,
            unbilled_cost_usd=obtained.unbilled_usd, batch_id=obtained.batch_id, custom_id=obtained.custom_id,
            output_artifact=call.scope.artifact, route=call.route,
            reused_from=None if obtained.reused_from is None
            else f"{call.prompt.id} v{obtained.reused_from[0]} {obtained.reused_from[1]}",
        ))


def _nothing(request: dict[str, Any], estimated: float | None, batch_id: str | None = None,
             custom_id: str | None = None) -> _Obtained:
    """A placeholder for an attempt that produced no response (refused, failed, errored)."""
    empty = Completion(text="", model=request["model"], stop_reason=None, usage=TokenUsage())
    return _Obtained(empty, "miss", 0.0, None, estimated, None, batch_id, custom_id)


def _fitting_prefix(costs: Sequence[float], available_usd: float) -> int:
    """How many leading items fit in `available_usd` together."""
    total = 0.0
    for n, cost in enumerate(costs):
        total += cost
        if total > available_usd:
            return n
    return len(costs)
