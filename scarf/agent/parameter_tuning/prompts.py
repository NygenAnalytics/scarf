from collections.abc import Sequence

from .contracts import ParameterCandidate, _default_parameter_candidates


def get_default_parameter_candidates() -> list[ParameterCandidate]:
    """Return a small one-factor candidate set around Scarf defaults."""

    return _default_parameter_candidates()


def build_initial_parameter_candidates(
    candidates: Sequence[ParameterCandidate],
    *,
    pair_harmony: bool,
) -> list[ParameterCandidate]:
    """Build deterministic initial branches from caller-authorized parameters."""

    initial: list[ParameterCandidate] = []
    for candidate in candidates:
        if pair_harmony and candidate.useHarmony:
            raise ValueError(
                "Initial seed candidates must not set useHarmony when the "
                "experimental handoff controls Harmony pairing"
            )
        initial.append(candidate)
        if pair_harmony:
            payload = candidate.model_dump()
            payload.update(
                {
                    "candidateId": f"{candidate.candidateId}_harmony",
                    "useHarmony": True,
                }
            )
            initial.append(ParameterCandidate.model_validate(payload))
    return initial
