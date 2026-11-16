"""Step f: observation-model priors from machine.toml [[observation]] and [[observation_channel]] (M12).

For each reviewed linear observation of a full-model component (kaful.observation.forms):
- `<channel>_y_limit` on the component (role observation): the reading when that component alone is at D_limit;
- `<channel>_y_healthy` on `machine` (role observation): the channel's reading on a healthy machine;
- `<channel>_noise_sd` on `machine` (role noise): declared, or a flagged default whose median is
  observation_noise_fraction of the channel's smallest |y_limit - y_healthy| (prior medians).
A threshold_event observation needs no parameter. Values are in the channel's unit (the SensorContract's).

A value that cites a page has its quote checked by code (hard rule 4): a `manual` value whose quote is not in the
page's raw text (verbatim or near) becomes `deployment`, and the downgrade is recorded. Confidence is computed by
code: exp(-log-sd) times the evidence weight of the quote's verification class, and times
judgement_confidence_base for an `llm_judgement` value, which is also flagged (rule 15).
Gate failures: an observation whose channel is not in the contract or has no unit, a cited page that is not in the
index, and a y_limit whose 5-95% interval overlaps y_healthy's (the reading's direction would be undetermined).
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

from kaful.artifacts.base import Evidence, SourceKind
from kaful.artifacts.distributions import Lognormal
from kaful.artifacts.document_index import DocumentIndex
from kaful.artifacts.param_set import ParamRole, Parameter
from kaful.artifacts.sensor_contract import SensorContract
from kaful.config import DeclaredValue, MachineConfig, ParameterizeSettings, UnderstandSettings
from kaful.evidence import verify_quote

Z95 = 1.6448536269514722
RAW_TEXT_CLASSES = ("verbatim", "near")  # where a manual quote must be found (rule 4)
FLAG_NOISE = "observation noise is a default (parameterize.observation_noise_fraction); data can estimate it"
FLAG_TRANSCRIBED = "an observation value was read from a figure (the page's LLM layer), checked by eye"
FLAG_DOWNGRADED = "an observation value's manual quote is not in the cited page's raw text: source deployment"
FLAG_JUDGED = "an observation value is a judgement (no document states it)"


@dataclass
class ObservationPriors:
    params: list[Parameter] = field(default_factory=list)
    flags: dict[str, list[str]] = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)

    def flag(self, component: str, text: str) -> None:
        if text not in self.flags.setdefault(component, []):
            self.flags[component].append(text)


def lognormal(v: DeclaredValue, p: ParameterizeSettings) -> Lognormal:
    if v.value is not None:
        return Lognormal(median=v.value, sigma_log=p.declared_spec_sigma)
    assert v.low is not None and v.high is not None
    return Lognormal(median=math.sqrt(v.low * v.high), sigma_log=math.log(v.high / v.low) / (2 * Z95))


def _evidence(v: DeclaredValue, index: DocumentIndex, u: UnderstandSettings) -> Evidence | None:
    pages = {(pg.doc_id, pg.page): pg for pg in index.pages}
    page = pages.get((v.doc_id or "", v.page or 0))
    if page is None:
        return None
    assert v.quote is not None
    verdict = verify_quote(v.quote, page, pages.get((page.doc_id, page.page + 1)), u.evidence_near_ratio,
                           u.evidence_candidate_overlap)
    return Evidence(doc_id=page.doc_id, page=page.page, page_label=page.label, quote=v.quote, verification=verdict)


def value_param(name: str, component: str, role: ParamRole, unit: str, v: DeclaredValue, what: str,
                index: DocumentIndex, p: ParameterizeSettings, u: UnderstandSettings,
                out: ObservationPriors, owner: str) -> None:
    """Append the parameter for declared value `v` (or a gate failure) to `out`; `owner` gets its flags."""
    prior = lognormal(v, p)
    evidence: tuple[Evidence, ...] = ()
    if v.quote is not None:
        ev = _evidence(v, index, u)
        if ev is None:
            out.failures.append(f"{what}: cited page {v.doc_id} p{v.page} is not in the document index")
            return
        evidence = (ev,)
    source: SourceKind = v.source
    downgraded: SourceKind | None = None
    if source == "manual" and evidence[0].verification not in RAW_TEXT_CLASSES:
        source, downgraded = "deployment", "manual"
        out.flag(owner, FLAG_DOWNGRADED)
    ref = f"{what}: {v.ref}" if source == "deployment" else v.ref
    if v.transcribed:
        out.flag(owner, FLAG_TRANSCRIBED)
    weight = u.evidence_weights[evidence[0].verification] if evidence else 1.0
    if source == "llm_judgement":
        weight *= p.judgement_confidence_base
        out.flag(owner, FLAG_JUDGED)
    out.params.append(Parameter(name=name, component=component, role=role, unit=unit, prior=prior, source=source,
                                ref=ref, confidence=math.exp(-prior.sigma_log) * weight, evidence=evidence,
                                downgraded_from=downgraded))


def _interval(x: Lognormal) -> tuple[float, float]:
    return x.quantile(0.05), x.quantile(0.95)


def observation_priors(machine: MachineConfig, contract: SensorContract, index: DocumentIndex,
                       components: Sequence[str], p: ParameterizeSettings,
                       u: UnderstandSettings) -> ObservationPriors:
    out, modelled = ObservationPriors(), set(components)
    names = {c.name for c in contract.channels}
    linear: dict[str, list[tuple[str, Lognormal]]] = {}
    for d in sorted(machine.observation, key=lambda d: (d.channel, d.component)):
        what = f"machine.toml [[observation]] {d.component} -> {d.channel}"
        if d.component not in modelled:
            continue  # build lists observations of components without a full model
        if d.channel not in names:
            out.failures.append(f"{what}: channel not in the SensorContract")
            continue
        unit = contract.channel(d.channel).unit
        if d.form != "linear":
            continue
        if not unit:
            out.failures.append(f"{what}: the channel has no unit")
            continue
        assert d.y_limit is not None
        before = len(out.params)
        value_param(f"{d.channel}_y_limit", d.component, "observation", unit, d.y_limit, what, index, p, u, out,
                    d.component)
        if len(out.params) > before:
            linear.setdefault(d.channel, []).append((d.component, lognormal(d.y_limit, p)))
    for ch in sorted(machine.observation_channel, key=lambda c: c.channel):
        if ch.channel not in linear:
            continue
        unit, owners = contract.channel(ch.channel).unit or "1", [c for c, _ in linear[ch.channel]]
        what = f"machine.toml [[observation_channel]] {ch.channel}"
        healthy = lognormal(ch.y_healthy, p)
        value_param(f"{ch.channel}_y_healthy", "machine", "observation", unit, ch.y_healthy, what, index, p, u, out,
                    owners[0])
        for component, limit in linear[ch.channel]:
            (hl, hh), (ll, lh) = _interval(healthy), _interval(limit)
            if ll <= hh and hl <= lh:
                out.failures.append(f"{what}: {component}'s y_limit 5-95% [{ll:.4g}, {lh:.4g}] overlaps y_healthy's "
                                    f"[{hl:.4g}, {hh:.4g}]; the reading's direction is undetermined")
        if ch.noise_sd is not None:
            value_param(f"{ch.channel}_noise_sd", "machine", "noise", unit, ch.noise_sd, what, index, p, u, out,
                        owners[0])
            continue
        span = min(abs(limit.median - healthy.median) for _, limit in linear[ch.channel])
        prior = Lognormal(median=p.observation_noise_fraction * span, sigma_log=p.observation_noise_sigma)
        out.params.append(Parameter(
            name=f"{ch.channel}_noise_sd", component="machine", role="noise", unit=unit, prior=prior,
            source="default", confidence=math.exp(-prior.sigma_log),
            ref=f"default: parameterize.observation_noise_fraction {p.observation_noise_fraction:g} of the smallest "
                f"|y_limit - y_healthy| ({span:.4g} {unit})"))
        for owner in owners:
            out.flag(owner, FLAG_NOISE)
    out.params.sort(key=lambda x: x.param_id)
    return out
