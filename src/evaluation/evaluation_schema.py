"""Evaluation response schema, validation, and weighted scoring."""

from __future__ import annotations

import copy
import re
from typing import Any


SCORE_WEIGHTS = {
    "hook_strength": 5,
    "emotional_impact": 4,
    "cliffhanger_strength": 4,
    "standalone_comprehensibility": 3,
    "story_coherence": 2,
    "promotional_potential": 2,
}
CLASSIFICATION_THRESHOLDS = (
    (80, "STRONG"),
    (65, "PROMISING"),
    (45, "WEAK"),
    (0, "REJECT"),
)
CONFIDENCE_LEVELS = ("low", "medium", "high")
SEMANTIC_EVIDENCE_STATUSES = (
    "SUPPORTED",
    "PARTIALLY_SUPPORTED",
    "UNSUPPORTED",
)

_CLAIM_STOP_WORDS = {
    "a", "an", "and", "as", "at", "before", "by", "during", "for", "from",
    "in", "into", "is", "it", "its", "of", "on", "or", "that", "the",
    "their", "them", "then", "there", "this", "through", "to", "toward",
    "under", "up", "upon", "was", "while", "with", "without", "after",
    "her", "him", "his", "she", "he", "they", "we", "you", "i", "me",
    "our", "your", "who", "what", "when", "where", "why", "how",
    "clip", "candidate", "moment", "event", "shot", "scene",
}

_CLAIM_SYNONYMS = {
    "advance": "walk",
    "approach": "walk",
    "approaches": "walk",
    "approaching": "walk",
    "advances": "walk",
    "advancing": "walk",
    "judgmental": "judge",
    "judging": "judge",
    "whispers": "whisper",
    "whispering": "whisper",
    "shocked": "shock",
    "shaking": "shake",
    "devastated": "devastation",
    "betrayed": "betrayal",
    "faces": "face",
    "leaves": "leave",
    "leaving": "leave",
}

OPENAI_RESPONSE_SCHEMA: dict[str, Any] = {
    "name": "promo_candidate_evaluation",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "candidate_id": {"type": "string"},
            "episode_number": {"type": "integer"},
            "start_timestamp": {"type": "string"},
            "end_timestamp": {"type": "string"},
            "duration_seconds": {"type": "number"},
            "scores": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    name: {"type": "integer", "minimum": 0, "maximum": 5}
                    for name in SCORE_WEIGHTS
                },
                "required": list(SCORE_WEIGHTS),
            },
            "hook": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "description": {"type": "string"},
                    "within_first_3_seconds": {"type": "boolean"},
                    "evidence": {"type": "array", "items": {"$ref": "#/$defs/evidence"}},
                },
                "required": ["description", "within_first_3_seconds", "evidence"],
            },
            "emotional_impact_reasoning": {"type": "string"},
            "cliffhanger": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "description": {"type": "string"},
                    "evidence": {"type": "array", "items": {"$ref": "#/$defs/evidence"}},
                },
                "required": ["description", "evidence"],
            },
            "standalone_comprehensibility_reasoning": {"type": "string"},
            "story_coherence_reasoning": {"type": "string"},
            "promotional_potential_reasoning": {"type": "string"},
            "strengths": {"type": "array", "items": {"type": "string"}},
            "weaknesses": {"type": "array", "items": {"type": "string"}},
            "human_review_required": {"type": "boolean"},
            "confidence": {"type": "string", "enum": list(CONFIDENCE_LEVELS)},
            "uncertainties": {"type": "array", "items": {"type": "string"}},
        },
        "required": [
            "candidate_id",
            "episode_number",
            "start_timestamp",
            "end_timestamp",
            "duration_seconds",
            "scores",
            "hook",
            "emotional_impact_reasoning",
            "cliffhanger",
            "standalone_comprehensibility_reasoning",
            "story_coherence_reasoning",
            "promotional_potential_reasoning",
            "strengths",
            "weaknesses",
            "human_review_required",
            "confidence",
            "uncertainties",
        ],
        "$defs": {
            "evidence": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "event_id": {"type": "string"},
                    "timestamp": {"type": "string"},
                    "excerpt": {"type": "string"},
                },
                "required": ["event_id", "timestamp", "excerpt"],
            }
        },
    },
}

PROPOSAL_RESPONSE_SCHEMA: dict[str, Any] = copy.deepcopy(OPENAI_RESPONSE_SCHEMA)
PROPOSAL_RESPONSE_SCHEMA["name"] = "promo_proposal_evaluation"
_proposal_schema = PROPOSAL_RESPONSE_SCHEMA["schema"]
_proposal_properties = _proposal_schema["properties"]
for _candidate_field in (
    "candidate_id",
    "episode_number",
    "start_timestamp",
    "end_timestamp",
    "duration_seconds",
):
    _proposal_properties.pop(_candidate_field)
_proposal_properties.update(
    {
        "promo_id": {"type": "string"},
        "promo_category": {
            "type": "string",
            "enum": [
                "SINGLE_SEGMENT",
                "SAME_EPISODE_MULTI_SEGMENT",
                "CROSS_EPISODE_MULTI_SEGMENT",
            ],
        },
        "total_duration_seconds": {"type": "number"},
    }
)
_proposal_schema["required"] = [
    "promo_id",
    "promo_category",
    "total_duration_seconds",
    *[
        field
        for field in _proposal_schema["required"]
        if field
        not in {
            "candidate_id",
            "episode_number",
            "start_timestamp",
            "end_timestamp",
            "duration_seconds",
        }
    ],
]
_proposal_evidence = _proposal_schema["$defs"]["evidence"]
_proposal_evidence["properties"] = {
    "segment_id": {"type": "string"},
    "playback_order": {"type": "integer"},
    "episode_id": {"type": "string"},
    "event_id": {"type": "string"},
    "source_timestamp": {"type": "string"},
    "playback_timestamp": {"type": "string"},
    "excerpt": {"type": "string"},
}
_proposal_evidence["required"] = list(_proposal_evidence["properties"])


def gemini_response_schema() -> dict[str, Any]:
    """Return a Gemini-compatible JSON Schema with local references inlined."""
    schema = copy.deepcopy(OPENAI_RESPONSE_SCHEMA["schema"])
    definitions = schema.pop("$defs")

    def inline_references(value: Any) -> Any:
        if isinstance(value, dict):
            reference = value.get("$ref")
            if isinstance(reference, str) and reference.startswith("#/$defs/"):
                name = reference.rsplit("/", maxsplit=1)[-1]
                if name not in definitions:
                    raise ValueError(f"Unknown schema reference: {reference}")
                return inline_references(copy.deepcopy(definitions[name]))
            return {
                key: inline_references(item)
                for key, item in value.items()
                if key != "additionalProperties"
            }
        if isinstance(value, list):
            return [inline_references(item) for item in value]
        return value

    return inline_references(schema)


def gemini_proposal_response_schema() -> dict[str, Any]:
    """Return the assembled-promo response schema in Gemini's supported form."""
    schema = copy.deepcopy(PROPOSAL_RESPONSE_SCHEMA["schema"])
    definitions = schema.pop("$defs")

    def inline_references(value: Any) -> Any:
        if isinstance(value, dict):
            reference = value.get("$ref")
            if isinstance(reference, str) and reference.startswith("#/$defs/"):
                name = reference.rsplit("/", maxsplit=1)[-1]
                if name not in definitions:
                    raise ValueError(f"Unknown schema reference: {reference}")
                return inline_references(copy.deepcopy(definitions[name]))
            return {
                key: inline_references(item)
                for key, item in value.items()
                if key != "additionalProperties"
            }
        if isinstance(value, list):
            return [inline_references(item) for item in value]
        return value

    return inline_references(schema)


class EvaluationValidationError(ValueError):
    """Raised when a model evaluation does not match the candidate/schema."""


def _semantic_tokens(text: str) -> set[str]:
    tokens: set[str] = set()
    for token in re.findall(r"[a-z0-9']+", text.casefold()):
        if token in _CLAIM_STOP_WORDS or len(token) < 3:
            continue
        token = _CLAIM_SYNONYMS.get(token, token)
        if token.endswith("ies") and len(token) > 4:
            token = f"{token[:-3]}y"
        elif token.endswith("ing") and len(token) > 5:
            token = token[:-3]
        elif token.endswith("ed") and len(token) > 4:
            token = token[:-2]
        elif token.endswith("s") and len(token) > 4:
            token = token[:-1]
        tokens.add(token)
    return tokens


def _semantic_evidence_status(description: str, evidence: list[dict[str, Any]]) -> str:
    """Conservatively compare claim-bearing words to the cited source excerpts.

    SUPPORTED requires complete normalized claim-token coverage. This is a
    screening heuristic, not a proof of natural-language entailment.
    """
    claim_tokens = _semantic_tokens(description)
    evidence_tokens = _semantic_tokens(
        " ".join(citation["excerpt"] for citation in evidence)
    )
    if not claim_tokens or not evidence_tokens:
        return "UNSUPPORTED"
    coverage = len(claim_tokens & evidence_tokens) / len(claim_tokens)
    if coverage == 1.0:
        return "SUPPORTED"
    if coverage >= 0.25:
        return "PARTIALLY_SUPPORTED"
    return "UNSUPPORTED"


def calculate_final_score(scores: dict[str, Any]) -> int:
    """Calculate the rubric's exact integer 0–100 weighted score."""
    if not isinstance(scores, dict):
        raise EvaluationValidationError("scores must be an object")
    if set(scores) != set(SCORE_WEIGHTS):
        raise EvaluationValidationError("scores must contain exactly the six rubric dimensions")
    for name, value in scores.items():
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 5:
            raise EvaluationValidationError(f"{name} must be an integer from 0 to 5")
    return sum(scores[name] * weight for name, weight in SCORE_WEIGHTS.items())


def classification_for_score(score: int) -> str:
    """Return the rubric class for a valid weighted score."""
    if isinstance(score, bool) or not isinstance(score, int) or not 0 <= score <= 100:
        raise EvaluationValidationError(f"Score outside supported range: {score!r}")
    for threshold, classification in CLASSIFICATION_THRESHOLDS:
        if score >= threshold:
            return classification
    raise EvaluationValidationError(f"Score outside supported range: {score}")


def _candidate_identity(candidate: dict[str, Any]) -> tuple[str, int, str, str, float]:
    episode = candidate.get("episode")
    time_range = candidate.get("time_range")
    if not isinstance(episode, dict) or not isinstance(time_range, dict):
        raise EvaluationValidationError("Candidate is missing episode or time_range metadata")
    return (
        str(candidate.get("candidate_id", "")),
        episode.get("episode_number"),
        time_range.get("start", {}).get("raw"),
        time_range.get("end", {}).get("raw"),
        time_range.get("duration_seconds"),
    )


def _validate_evidence(
    evidence: Any,
    *,
    candidate_events: dict[str, dict[str, Any]],
    candidate_start_seconds: float,
    candidate_end_seconds: float,
    target_window: tuple[float, float] | None = None,
    allow_outside_target_window: bool = False,
    path: str,
) -> None:
    if not isinstance(evidence, list):
        raise EvaluationValidationError(f"{path} must be a list")
    has_target_window_evidence = False
    for index, item in enumerate(evidence):
        if not isinstance(item, dict) or set(item) != {"event_id", "timestamp", "excerpt"}:
            raise EvaluationValidationError(f"{path}[{index}] has invalid evidence fields")
        if not all(
            isinstance(item[field], str)
            for field in ("event_id", "timestamp", "excerpt")
        ):
            raise EvaluationValidationError(f"{path}[{index}] evidence values must be strings")
        event = candidate_events.get(item["event_id"])
        if event is None:
            raise EvaluationValidationError(
                f"{path}[{index}] references unknown event ID {item['event_id']!r}"
            )
        timestamp = item["timestamp"]
        allowed_timestamps = {event.get("start", {}).get("raw"), event.get("end", {}).get("raw")}
        if timestamp not in allowed_timestamps:
            raise EvaluationValidationError(
                f"{path}[{index}] timestamp must match the cited event's start or end"
            )
        seconds = _timestamp_seconds(timestamp)
        if not candidate_start_seconds <= seconds <= candidate_end_seconds:
            raise EvaluationValidationError(f"{path}[{index}] timestamp is outside the candidate")
        event_start = _timestamp_seconds(event["start"]["raw"])
        event_end = _timestamp_seconds(event["end"]["raw"])
        if target_window is not None:
            window_start, window_end = target_window
            overlaps_target_window = event_start < window_end and event_end > window_start
            has_target_window_evidence = has_target_window_evidence or overlaps_target_window
            if not overlaps_target_window and not allow_outside_target_window:
                raise EvaluationValidationError(
                    f"{path}[{index}] event does not overlap the required evaluation window"
                )
        if not isinstance(item["excerpt"], str):
            raise EvaluationValidationError(f"{path}[{index}].excerpt must be a string")
        if not item["excerpt"].strip():
            raise EvaluationValidationError(f"{path}[{index}].excerpt must not be empty")
        content = event.get("content")
        source_text = content.get("raw") if isinstance(content, dict) else content
        if not isinstance(source_text, str) or item["excerpt"] not in source_text:
            raise EvaluationValidationError(
                f"{path}[{index}].excerpt is not present in the cited event content"
            )
    if (
        evidence
        and target_window is not None
        and allow_outside_target_window
        and not has_target_window_evidence
    ):
        raise EvaluationValidationError(
            f"{path} must include evidence overlapping the required evaluation window"
        )


def _timestamp_seconds(timestamp: str) -> float:
    """Parse candidate-format MM:SS.hh timestamps for evidence window checks."""
    try:
        minutes, remainder = timestamp.split(":", maxsplit=1)
        seconds, hundredths = remainder.split(".", maxsplit=1)
        if (
            len(minutes) != 2
            or len(seconds) != 2
            or len(hundredths) != 2
            or int(seconds) >= 60
        ):
            raise ValueError
        return int(minutes) * 60 + int(seconds) + int(hundredths) / 100
    except (AttributeError, ValueError) as error:
        raise EvaluationValidationError(f"Invalid timestamp: {timestamp!r}") from error


def preprocess_evidence_citations(
    evaluation: Any,
    candidate: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    """Remove out-of-range citations only when same-claim evidence independently remains.

    This function never substitutes an event endpoint for the timestamp supplied
    by the model. The regular validator remains responsible for final validation.
    """
    if not isinstance(evaluation, dict):
        raise EvaluationValidationError("Evaluation response must be a JSON object")
    candidate_identity = _candidate_identity(candidate)
    candidate_start = _timestamp_seconds(candidate_identity[2])
    candidate_end = _timestamp_seconds(candidate_identity[3])
    candidate_events: dict[str, dict[str, Any]] = {}
    for wrapped_event in candidate.get("events", []):
        event = wrapped_event.get("event") if isinstance(wrapped_event, dict) else None
        if isinstance(event, dict) and isinstance(event.get("id"), str):
            candidate_events[event["id"]] = event

    normalized = copy.deepcopy(evaluation)
    repairs: list[dict[str, str]] = []
    for section_name in ("hook", "cliffhanger"):
        section = normalized.get(section_name)
        if not isinstance(section, dict) or not isinstance(section.get("evidence"), list):
            continue
        evidence = section["evidence"]
        prepared: list[tuple[dict[str, Any], bool, bool]] = []
        for index, citation in enumerate(evidence):
            path = f"{section_name}.evidence[{index}]"
            if not isinstance(citation, dict) or set(citation) != {
                "event_id",
                "timestamp",
                "excerpt",
            }:
                raise EvaluationValidationError(f"{path} has invalid evidence fields")
            if not all(
                isinstance(citation[field], str)
                for field in ("event_id", "timestamp", "excerpt")
            ):
                raise EvaluationValidationError(f"{path} evidence values must be strings")
            event = candidate_events.get(citation["event_id"])
            if event is None:
                raise EvaluationValidationError(
                    f"{path} references unknown event ID {citation['event_id']!r}"
                )
            timestamp = citation["timestamp"]
            allowed_timestamps = {
                event.get("start", {}).get("raw"),
                event.get("end", {}).get("raw"),
            }
            if timestamp not in allowed_timestamps:
                raise EvaluationValidationError(
                    f"{path} timestamp must match the cited event's start or end"
                )
            if not citation["excerpt"].strip():
                raise EvaluationValidationError(f"{path}.excerpt must not be empty")
            content = event.get("content")
            source_text = content.get("raw") if isinstance(content, dict) else content
            if not isinstance(source_text, str) or citation["excerpt"] not in source_text:
                raise EvaluationValidationError(
                    f"{path}.excerpt is not present in the cited event content"
                )

            seconds = _timestamp_seconds(timestamp)
            in_candidate = candidate_start <= seconds <= candidate_end
            supports_required_window = True
            if section_name == "hook" and section.get("within_first_3_seconds") is True:
                hook_end = min(candidate_end, candidate_start + 3.0)
                event_start = _timestamp_seconds(event["start"]["raw"])
                event_end = _timestamp_seconds(event["end"]["raw"])
                supports_required_window = event_start < hook_end and event_end > candidate_start
            prepared.append((citation, in_candidate, supports_required_window))

        independently_valid = [
            citation
            for citation, in_candidate, supports_required_window in prepared
            if in_candidate and supports_required_window
        ]
        has_independent_support = bool(independently_valid)
        if has_independent_support and any(
            not in_candidate for _, in_candidate, _ in prepared
        ):
            description = section.get("description")
            if not isinstance(description, str) or _semantic_evidence_status(
                description, independently_valid
            ) != "SUPPORTED":
                raise EvaluationValidationError(
                    f"{section_name} contains out-of-candidate evidence and the remaining "
                    "citations do not independently support its description"
                )
        kept: list[dict[str, Any]] = []
        for citation, in_candidate, supports_required_window in prepared:
            if in_candidate:
                kept.append(citation)
                continue
            if not has_independent_support:
                raise EvaluationValidationError(
                    f"{section_name}.evidence timestamp is outside the candidate and no "
                    "independently valid citation supports the same claim"
                )
            repairs.append(
                {
                    "section": section_name,
                    "event_id": citation["event_id"],
                    "timestamp": citation["timestamp"],
                    "reason": (
                        "Removed out-of-candidate timestamp because another independently "
                        "valid citation in the same claim evidence group remains; no "
                        "endpoint substitution was made."
                    ),
                }
            )
        section["evidence"] = kept
    return normalized, repairs


def validate_evaluation(
    evaluation: Any,
    candidate: dict[str, Any],
) -> dict[str, Any]:
    """Validate model output and return it with Python-authoritative score fields."""
    if not isinstance(evaluation, dict):
        raise EvaluationValidationError("Evaluation response must be a JSON object")
    required = set(OPENAI_RESPONSE_SCHEMA["schema"]["required"])
    optional_fields = {
        "final_score",
        "classification",
        "model_reported_final_score",
        "model_reported_classification",
    }
    if not required.issubset(evaluation):
        missing = sorted(required - set(evaluation))
        raise EvaluationValidationError(f"Invalid fields; missing={missing}, extra=[]")
    extra = sorted(set(evaluation) - required - optional_fields)
    if extra:
        raise EvaluationValidationError(f"Invalid fields; missing=[], extra={extra}")
    expected_identity = _candidate_identity(candidate)
    observed_identity = (
        evaluation["candidate_id"],
        evaluation["episode_number"],
        evaluation["start_timestamp"],
        evaluation["end_timestamp"],
        evaluation["duration_seconds"],
    )
    if isinstance(evaluation["episode_number"], bool) or not isinstance(
        evaluation["episode_number"], int
    ):
        raise EvaluationValidationError("episode_number must be an integer")
    if isinstance(evaluation["duration_seconds"], bool) or not isinstance(
        evaluation["duration_seconds"], (int, float)
    ):
        raise EvaluationValidationError("duration_seconds must be numeric")
    if not all(
        isinstance(evaluation[field], str)
        for field in ("candidate_id", "start_timestamp", "end_timestamp")
    ):
        raise EvaluationValidationError("Candidate IDs and timestamps must be strings")
    if observed_identity != expected_identity:
        raise EvaluationValidationError(
            f"Candidate identity mismatch: expected {expected_identity}, got {observed_identity}"
        )

    final_score = calculate_final_score(evaluation["scores"])
    expected_classification = classification_for_score(final_score)
    reported_score = (
        evaluation["model_reported_final_score"]
        if "model_reported_final_score" in evaluation
        else evaluation.get("final_score")
    )
    reported_classification = (
        evaluation["model_reported_classification"]
        if "model_reported_classification" in evaluation
        else evaluation.get("classification")
    )

    required_strings = (
        "emotional_impact_reasoning",
        "standalone_comprehensibility_reasoning",
        "story_coherence_reasoning",
        "promotional_potential_reasoning",
    )
    for field in required_strings:
        if not isinstance(evaluation[field], str) or not evaluation[field].strip():
            raise EvaluationValidationError(f"{field} must be a non-empty string")
    if not isinstance(evaluation["strengths"], list) or not all(
        isinstance(item, str) for item in evaluation["strengths"]
    ):
        raise EvaluationValidationError("strengths must be a list of strings")
    if not isinstance(evaluation["weaknesses"], list) or not all(
        isinstance(item, str) for item in evaluation["weaknesses"]
    ):
        raise EvaluationValidationError("weaknesses must be a list of strings")
    if not isinstance(evaluation["uncertainties"], list) or not all(
        isinstance(item, str) for item in evaluation["uncertainties"]
    ):
        raise EvaluationValidationError("uncertainties must be a list of strings")
    if not isinstance(evaluation["human_review_required"], bool):
        raise EvaluationValidationError("human_review_required must be a boolean")
    if evaluation["confidence"] not in CONFIDENCE_LEVELS:
        raise EvaluationValidationError(f"confidence must be one of {CONFIDENCE_LEVELS}")

    hook = evaluation["hook"]
    cliffhanger = evaluation["cliffhanger"]
    if not isinstance(hook, dict) or set(hook) - {
        "description",
        "within_first_3_seconds",
        "evidence",
        "semantic_evidence_status",
    } or not {"description", "within_first_3_seconds", "evidence"}.issubset(hook):
        raise EvaluationValidationError("hook has invalid structure")
    if not isinstance(hook["description"], str) or not isinstance(
        hook["within_first_3_seconds"], bool
    ):
        raise EvaluationValidationError("hook description/within_first_3_seconds have invalid types")
    if (
        not isinstance(cliffhanger, dict)
        or set(cliffhanger) - {"description", "evidence", "semantic_evidence_status"}
        or not {"description", "evidence"}.issubset(cliffhanger)
    ):
        raise EvaluationValidationError("cliffhanger has invalid structure")
    if not isinstance(cliffhanger["description"], str):
        raise EvaluationValidationError("cliffhanger description must be a string")

    candidate_start = _timestamp_seconds(expected_identity[2])
    candidate_end = _timestamp_seconds(expected_identity[3])
    candidate_events: dict[str, dict[str, Any]] = {}
    for item in candidate.get("events", []):
        event = item.get("event") if isinstance(item, dict) else None
        if isinstance(event, dict) and isinstance(event.get("id"), str):
            candidate_events[event["id"]] = event
    hook_window = (candidate_start, min(candidate_end, candidate_start + 3.0))
    ending_window = (max(candidate_start, candidate_end - 3.0), candidate_end)
    _validate_evidence(
        hook["evidence"],
        candidate_events=candidate_events,
        candidate_start_seconds=candidate_start,
        candidate_end_seconds=candidate_end,
        target_window=hook_window if hook["within_first_3_seconds"] else None,
        path="hook.evidence",
    )
    _validate_evidence(
        cliffhanger["evidence"],
        candidate_events=candidate_events,
        candidate_start_seconds=candidate_start,
        candidate_end_seconds=candidate_end,
        target_window=ending_window,
        allow_outside_target_window=True,
        path="cliffhanger.evidence",
    )
    normalized = {
        key: value
        for key, value in evaluation.items()
        if key
        not in {
            "final_score",
            "classification",
            "model_reported_final_score",
            "model_reported_classification",
        }
    }
    normalized["final_score"] = final_score
    normalized["classification"] = expected_classification
    normalized["hook"] = dict(hook)
    normalized["cliffhanger"] = dict(cliffhanger)
    hook_status = _semantic_evidence_status(
        hook["description"],
        hook["evidence"],
    )
    cliffhanger_status = _semantic_evidence_status(
        cliffhanger["description"],
        cliffhanger["evidence"],
    )
    normalized["hook"]["semantic_evidence_status"] = hook_status
    normalized["cliffhanger"]["semantic_evidence_status"] = cliffhanger_status
    if hook_status != "SUPPORTED" or cliffhanger_status != "SUPPORTED":
        normalized["human_review_required"] = True
    if "model_reported_final_score" in evaluation or "final_score" in evaluation:
        normalized["model_reported_final_score"] = reported_score
    if "model_reported_classification" in evaluation or "classification" in evaluation:
        normalized["model_reported_classification"] = reported_classification
    return normalized


def _timestamp_hundredths(timestamp: Any) -> int:
    if not isinstance(timestamp, str):
        raise EvaluationValidationError(f"Invalid timestamp: {timestamp!r}")
    match = re.fullmatch(r"(\d{2}):(\d{2})\.(\d{2})", timestamp)
    if match is None or int(match.group(2)) >= 60:
        raise EvaluationValidationError(f"Invalid timestamp: {timestamp!r}")
    return int(match.group(1)) * 6_000 + int(match.group(2)) * 100 + int(match.group(3))


def _format_timestamp_hundredths(value: int) -> str:
    minutes, remainder = divmod(value, 6_000)
    seconds, hundredths = divmod(remainder, 100)
    return f"{minutes:02d}:{seconds:02d}.{hundredths:02d}"


def _numeric_hundredths(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvaluationValidationError(f"{field} must be numeric")
    scaled = value * 100
    rounded = round(scaled)
    if abs(scaled - rounded) > 1e-6:
        raise EvaluationValidationError(f"{field} must have hundredth-second precision")
    return rounded


def normalize_proposal(
    proposal: Any,
    *,
    source_events: dict[tuple[str, str], dict[str, Any]],
    source_candidates: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Validate a constructed proposal and derive its assembled playback timeline.

    Source timestamps remain episode-relative and unchanged. Playback timestamps
    are derived independently for each segment from its source start and offset.
    """
    if not isinstance(proposal, dict):
        raise EvaluationValidationError("Proposal must be a JSON object")
    promo_id = proposal.get("promo_id")
    category = proposal.get("promo_category")
    segments = proposal.get("segments")
    if not isinstance(promo_id, str) or not promo_id:
        raise EvaluationValidationError("Proposal is missing promo_id")
    if category not in {
        "SINGLE_SEGMENT",
        "SAME_EPISODE_MULTI_SEGMENT",
        "CROSS_EPISODE_MULTI_SEGMENT",
    }:
        raise EvaluationValidationError(f"Invalid proposal category: {category!r}")
    if not isinstance(segments, list) or not segments:
        raise EvaluationValidationError(f"{promo_id} has no segments")
    if category == "SINGLE_SEGMENT" and len(segments) != 1:
        raise EvaluationValidationError("SINGLE_SEGMENT proposals must contain one segment")
    if category != "SINGLE_SEGMENT" and len(segments) < 2:
        raise EvaluationValidationError("Multi-segment proposals must contain at least two segments")

    normalized_segments: list[dict[str, Any]] = []
    playback_cursor = 0
    all_event_refs: list[str] = []
    episode_numbers: list[int] = []
    seen_segment_ids: set[str] = set()
    for index, segment in enumerate(segments, start=1):
        path = f"{promo_id}.segments[{index - 1}]"
        if not isinstance(segment, dict):
            raise EvaluationValidationError(f"{path} must be an object")
        segment_id = segment.get("segment_id")
        if not isinstance(segment_id, str) or not segment_id:
            raise EvaluationValidationError(f"{path}.segment_id is required")
        if segment_id in seen_segment_ids:
            raise EvaluationValidationError(f"Duplicate segment_id: {segment_id}")
        seen_segment_ids.add(segment_id)
        segment_order = segment.get("segment_order")
        if isinstance(segment_order, bool) or segment_order != index:
            raise EvaluationValidationError(f"{path}.segment_order must be {index}")
        episode_id = segment.get("episode_id")
        episode_number = segment.get("episode_number")
        if (
            not isinstance(episode_id, str)
            or isinstance(episode_number, bool)
            or not isinstance(episode_number, int)
        ):
            raise EvaluationValidationError(f"{path} has invalid source episode identity")
        episode_numbers.append(episode_number)
        try:
            source_start = _timestamp_hundredths(segment.get("start_timestamp"))
            source_end = _timestamp_hundredths(segment.get("end_timestamp"))
        except EvaluationValidationError as error:
            raise EvaluationValidationError(f"{path} has invalid source boundaries") from error
        if source_start >= source_end:
            raise EvaluationValidationError(f"{path} source start must precede source end")
        duration = source_end - source_start
        if _numeric_hundredths(segment.get("duration_seconds"), field=f"{path}.duration_seconds") != duration:
            raise EvaluationValidationError(f"{path} source duration does not match its boundaries")
        minimum_segment = 2_000 if category == "SINGLE_SEGMENT" else 800
        maximum_segment = 6_000 if category == "SINGLE_SEGMENT" else 2_200
        if not minimum_segment <= duration <= maximum_segment:
            raise EvaluationValidationError(f"{path} duration is outside its category bounds")
        stored_playback_start = _numeric_hundredths(
            segment.get("playback_offset_seconds"),
            field=f"{path}.playback_offset_seconds",
        )
        stored_playback_end = _numeric_hundredths(
            segment.get("playback_end_seconds"),
            field=f"{path}.playback_end_seconds",
        )
        if stored_playback_start != playback_cursor:
            raise EvaluationValidationError(f"{path} playback start offset is inconsistent")
        if stored_playback_end != playback_cursor + duration:
            raise EvaluationValidationError(f"{path} playback end offset is inconsistent")

        source_event_ids = segment.get("source_event_ids")
        if (
            not isinstance(source_event_ids, list)
            or not source_event_ids
            or not all(isinstance(item, str) for item in source_event_ids)
            or len(source_event_ids) != len(set(source_event_ids))
        ):
            raise EvaluationValidationError(f"{path}.source_event_ids must be unique non-empty IDs")
        event_wrappers = segment.get("events")
        if not isinstance(event_wrappers, list) or not event_wrappers:
            raise EvaluationValidationError(f"{path} is missing embedded source events")
        wrapped_ids: list[str] = []
        normalized_events: list[dict[str, Any]] = []
        for event_index, wrapped in enumerate(event_wrappers):
            event_path = f"{path}.events[{event_index}]"
            if not isinstance(wrapped, dict) or not isinstance(wrapped.get("event"), dict):
                raise EvaluationValidationError(f"{event_path} is malformed")
            event = wrapped["event"]
            event_id = event.get("id")
            if not isinstance(event_id, str):
                raise EvaluationValidationError(f"{event_path} has no event ID")
            if wrapped.get("source_event_id") != event_id:
                raise EvaluationValidationError(f"{event_path} source event identity mismatch")
            if wrapped.get("source_event_ref") != event_id:
                raise EvaluationValidationError(f"{event_path} source event reference mismatch")
            wrapped_ids.append(event_id)
            source_record = source_events.get((episode_id, event_id))
            if source_record is None:
                raise EvaluationValidationError(
                    f"{event_path} references unknown source event {episode_id}:{event_id}"
                )
            actual_event = source_record.get("event")
            if not isinstance(actual_event, dict) or source_record.get("episode_number") != episode_number:
                raise EvaluationValidationError(f"{event_path} source episode does not match parsed data")
            for timestamp_field in ("start", "end"):
                embedded_raw = event.get(timestamp_field, {}).get("raw")
                actual_raw = actual_event.get(timestamp_field, {}).get("raw")
                if embedded_raw != actual_raw:
                    raise EvaluationValidationError(
                        f"{event_path} {timestamp_field} differs from parsed source"
                    )
            embedded_content = event.get("content", {})
            actual_content = actual_event.get("content", {})
            embedded_text = (
                embedded_content.get("raw")
                if isinstance(embedded_content, dict)
                else embedded_content
            )
            actual_text = (
                actual_content.get("raw")
                if isinstance(actual_content, dict)
                else actual_content
            )
            if embedded_text != actual_text:
                raise EvaluationValidationError(f"{event_path} content differs from parsed source")

            event_start = _timestamp_hundredths(actual_event.get("start", {}).get("raw"))
            event_end = _timestamp_hundredths(actual_event.get("end", {}).get("raw"))
            if event_start >= source_end or event_end <= source_start:
                raise EvaluationValidationError(f"{event_path} does not overlap its source segment")
            clipped_start = max(event_start, source_start)
            clipped_end = min(event_end, source_end)
            playback_event_start = playback_cursor + clipped_start - source_start
            playback_event_end = playback_cursor + clipped_end - source_start
            source_start_eligible = source_start <= event_start <= source_end
            source_end_eligible = source_start <= event_end <= source_end
            playback_timestamp_for_source_start = (
                _format_timestamp_hundredths(playback_cursor + event_start - source_start)
                if source_start_eligible
                else None
            )
            playback_timestamp_for_source_end = (
                _format_timestamp_hundredths(playback_cursor + event_end - source_start)
                if source_end_eligible
                else None
            )
            normalized_events.append(
                {
                    "event_id": event_id,
                    "episode_id": episode_id,
                    "event": actual_event,
                    "source_start_timestamp": actual_event["start"]["raw"],
                    "source_end_timestamp": actual_event["end"]["raw"],
                    "source_start_endpoint_eligible": source_start_eligible,
                    "playback_timestamp_for_source_start": playback_timestamp_for_source_start,
                    "source_end_endpoint_eligible": source_end_eligible,
                    "playback_timestamp_for_source_end": playback_timestamp_for_source_end,
                    "clipped_playback_overlap_start_timestamp": _format_timestamp_hundredths(
                        playback_event_start
                    ),
                    "clipped_playback_overlap_end_timestamp": _format_timestamp_hundredths(
                        playback_event_end
                    ),
                    "playback_start_hundredths": playback_event_start,
                    "playback_end_hundredths": playback_event_end,
                }
            )
        if set(wrapped_ids) != set(source_event_ids):
            raise EvaluationValidationError(f"{path} embedded events do not match source_event_ids")
        all_event_refs.extend(source_event_ids)

        candidate_ids = segment.get("source_candidate_ids")
        if (
            not isinstance(candidate_ids, list)
            or not all(isinstance(candidate_id, str) for candidate_id in candidate_ids)
        ):
            raise EvaluationValidationError(f"{path}.source_candidate_ids must be a list of IDs")
        for candidate_id in candidate_ids:
            source_candidate = source_candidates.get(candidate_id)
            if (
                source_candidate is None
                or source_candidate.get("episode", {}).get("episode_id") != episode_id
            ):
                raise EvaluationValidationError(
                    f"{path} references candidate from the wrong or unknown episode"
                )
            candidate_event_ids = {
                wrapped.get("event", {}).get("id")
                for wrapped in source_candidate.get("events", [])
                if isinstance(wrapped, dict)
            }
            if not set(source_event_ids).issubset(candidate_event_ids):
                raise EvaluationValidationError(
                    f"{path} source candidate does not contain all segment events"
                )

        normalized_segments.append(
            {
                "segment_id": segment_id,
                "playback_order": index,
                "episode_id": episode_id,
                "episode_number": episode_number,
                "source_start_timestamp": segment["start_timestamp"],
                "source_end_timestamp": segment["end_timestamp"],
                "source_duration_seconds": duration / 100,
                "playback_start_offset_seconds": playback_cursor / 100,
                "playback_end_offset_seconds": (playback_cursor + duration) / 100,
                "playback_start_timestamp": _format_timestamp_hundredths(playback_cursor),
                "playback_end_timestamp": _format_timestamp_hundredths(playback_cursor + duration),
                "source_candidate_ids": list(candidate_ids),
                "source_event_ids": list(source_event_ids),
                "events": normalized_events,
            }
        )
        playback_cursor += duration

    if category == "SAME_EPISODE_MULTI_SEGMENT":
        if len(set(segment["episode_id"] for segment in normalized_segments)) != 1:
            raise EvaluationValidationError("Same-episode proposal contains multiple episodes")
        for previous, current in zip(normalized_segments, normalized_segments[1:]):
            if _timestamp_hundredths(current["source_start_timestamp"]) < _timestamp_hundredths(
                previous["source_end_timestamp"]
            ):
                raise EvaluationValidationError(
                    "Same-episode segments overlap or are out of source order"
                )
    if category == "CROSS_EPISODE_MULTI_SEGMENT":
        if len(set(segment["episode_id"] for segment in normalized_segments)) < 2:
            raise EvaluationValidationError("Cross-episode proposal must span multiple episodes")
        if any(right <= left for left, right in zip(episode_numbers, episode_numbers[1:])):
            raise EvaluationValidationError("Cross-episode segments must follow episode-number order")
    total_duration = _numeric_hundredths(
        proposal.get("total_duration_seconds"),
        field="total_duration_seconds",
    )
    if total_duration != playback_cursor:
        raise EvaluationValidationError("Proposal total duration does not match segment durations")
    if not 2_000 <= total_duration <= 6_000:
        raise EvaluationValidationError("Assembled promo duration must be between 20 and 60 seconds")
    proposal_source_ids = proposal.get("source_event_ids")
    if (
        not isinstance(proposal_source_ids, list)
        or len(proposal_source_ids) != len(set(proposal_source_ids))
        or set(proposal_source_ids) != set(all_event_refs)
    ):
        raise EvaluationValidationError("Proposal source event IDs do not match its segments")
    proposal_candidate_ids = proposal.get("source_candidate_ids")
    segment_candidate_ids = list(
        dict.fromkeys(
            candidate_id
            for segment in normalized_segments
            for candidate_id in segment["source_candidate_ids"]
        )
    )
    if (
        not isinstance(proposal_candidate_ids, list)
        or len(proposal_candidate_ids) != len(set(proposal_candidate_ids))
        or set(proposal_candidate_ids) != set(segment_candidate_ids)
    ):
        raise EvaluationValidationError("Proposal source candidate IDs do not match its segments")

    return {
        "evaluation_kind": "assembled_promo",
        "promo_id": promo_id,
        "promo_category": category,
        "total_duration_seconds": playback_cursor / 100,
        "segments": normalized_segments,
        "source_candidate_ids": list(proposal.get("source_candidate_ids", [])),
        "source_event_ids": list(dict.fromkeys(all_event_refs)),
        "narrative_rationale": proposal.get("narrative_rationale", ""),
        "source_proposal_fingerprint_payload": proposal,
    }


def validate_proposal_evaluation(
    evaluation: Any,
    proposal: dict[str, Any],
) -> dict[str, Any]:
    """Validate assembled-promo evidence against source and playback coordinates."""
    if not isinstance(evaluation, dict):
        raise EvaluationValidationError("Proposal evaluation must be a JSON object")
    required = set(PROPOSAL_RESPONSE_SCHEMA["schema"]["required"])
    optional_fields = {
        "final_score",
        "classification",
        "model_reported_final_score",
        "model_reported_classification",
    }
    if not required.issubset(evaluation):
        raise EvaluationValidationError(
            f"Invalid proposal evaluation fields; missing={sorted(required - set(evaluation))}"
        )
    extra = sorted(set(evaluation) - required - optional_fields)
    if extra:
        raise EvaluationValidationError(f"Invalid proposal evaluation fields; extra={extra}")
    expected_identity = (
        proposal["promo_id"],
        proposal["promo_category"],
        proposal["total_duration_seconds"],
    )
    observed_identity = (
        evaluation.get("promo_id"),
        evaluation.get("promo_category"),
        evaluation.get("total_duration_seconds"),
    )
    if observed_identity != expected_identity:
        raise EvaluationValidationError(
            f"Proposal identity mismatch: expected {expected_identity}, got {observed_identity}"
        )
    total_duration = _numeric_hundredths(
        evaluation["total_duration_seconds"],
        field="total_duration_seconds",
    )
    final_score = calculate_final_score(evaluation["scores"])
    expected_classification = classification_for_score(final_score)
    reported_score = evaluation.get("model_reported_final_score", evaluation.get("final_score"))
    reported_classification = evaluation.get(
        "model_reported_classification",
        evaluation.get("classification"),
    )

    for field in (
        "emotional_impact_reasoning",
        "standalone_comprehensibility_reasoning",
        "story_coherence_reasoning",
        "promotional_potential_reasoning",
    ):
        if not isinstance(evaluation[field], str) or not evaluation[field].strip():
            raise EvaluationValidationError(f"{field} must be a non-empty string")
    for field in ("strengths", "weaknesses", "uncertainties"):
        if not isinstance(evaluation[field], list) or not all(
            isinstance(item, str) for item in evaluation[field]
        ):
            raise EvaluationValidationError(f"{field} must be a list of strings")
    if not isinstance(evaluation["human_review_required"], bool):
        raise EvaluationValidationError("human_review_required must be a boolean")
    if evaluation["confidence"] not in CONFIDENCE_LEVELS:
        raise EvaluationValidationError(f"confidence must be one of {CONFIDENCE_LEVELS}")

    segments_by_id = {segment["segment_id"]: segment for segment in proposal["segments"]}
    hook = evaluation["hook"]
    cliffhanger = evaluation["cliffhanger"]
    if (
        not isinstance(hook, dict)
        or set(hook) - {
            "description",
            "within_first_3_seconds",
            "evidence",
            "semantic_evidence_status",
        }
        or not {"description", "within_first_3_seconds", "evidence"}.issubset(hook)
    ):
        raise EvaluationValidationError("hook has invalid proposal evidence structure")
    if not isinstance(hook["description"], str) or not isinstance(
        hook["within_first_3_seconds"], bool
    ):
        raise EvaluationValidationError("hook description/within_first_3_seconds have invalid types")
    if (
        not isinstance(cliffhanger, dict)
        or set(cliffhanger) - {"description", "evidence", "semantic_evidence_status"}
        or not {"description", "evidence"}.issubset(cliffhanger)
    ):
        raise EvaluationValidationError("cliffhanger has invalid proposal evidence structure")
    if not isinstance(cliffhanger["description"], str):
        raise EvaluationValidationError("cliffhanger description must be a string")

    def validate_proposal_citations(
        evidence: Any,
        *,
        section_name: str,
        target_start: int,
        target_end: int,
    ) -> None:
        if not isinstance(evidence, list):
            raise EvaluationValidationError(f"{section_name}.evidence must be a list")
        for index, citation in enumerate(evidence):
            path = f"{section_name}.evidence[{index}]"
            citation_fields = {
                "segment_id",
                "playback_order",
                "episode_id",
                "event_id",
                "source_timestamp",
                "playback_timestamp",
                "excerpt",
            }
            if not isinstance(citation, dict) or set(citation) != citation_fields:
                raise EvaluationValidationError(f"{path} has invalid evidence fields")
            if not all(isinstance(citation[field], str) for field in citation_fields - {"playback_order"}):
                raise EvaluationValidationError(f"{path} evidence values must be strings")
            playback_order = citation["playback_order"]
            if isinstance(playback_order, bool) or not isinstance(playback_order, int):
                raise EvaluationValidationError(f"{path}.playback_order must be an integer")
            segment = segments_by_id.get(citation["segment_id"])
            if segment is None:
                raise EvaluationValidationError(f"{path} references unknown segment")
            if (
                playback_order != segment["playback_order"]
                or citation["episode_id"] != segment["episode_id"]
            ):
                raise EvaluationValidationError(f"{path} segment/episode identity mismatch")
            matching_events = [
                event_record
                for event_record in segment["events"]
                if event_record["event_id"] == citation["event_id"]
            ]
            if len(matching_events) != 1:
                raise EvaluationValidationError(f"{path} event does not belong to this segment")
            event_record = matching_events[0]
            source_timestamp = citation["source_timestamp"]
            source_value = _timestamp_hundredths(source_timestamp)
            if source_timestamp not in {
                event_record["source_start_timestamp"],
                event_record["source_end_timestamp"],
            }:
                raise EvaluationValidationError(f"{path} source timestamp is not an event endpoint")
            source_segment_start = _timestamp_hundredths(segment["source_start_timestamp"])
            source_segment_end = _timestamp_hundredths(segment["source_end_timestamp"])
            if not source_segment_start <= source_value <= source_segment_end:
                raise EvaluationValidationError(f"{path} source timestamp is outside its segment")
            mapped_playback = (
                _timestamp_hundredths(segment["playback_start_timestamp"])
                + source_value
                - source_segment_start
            )
            playback_timestamp = citation["playback_timestamp"]
            if _timestamp_hundredths(playback_timestamp) != mapped_playback:
                raise EvaluationValidationError(f"{path} playback timestamp mapping is incorrect")
            if not target_start <= mapped_playback <= target_end:
                raise EvaluationValidationError(f"{path} citation is outside the {section_name} window")
            excerpt = citation["excerpt"]
            source_content = event_record["event"].get("content", {})
            source_text = (
                source_content.get("raw")
                if isinstance(source_content, dict)
                else source_content
            )
            if (
                not excerpt.strip()
                or not isinstance(source_text, str)
                or excerpt not in source_text
            ):
                raise EvaluationValidationError(f"{path} excerpt is not present in the source event")

    opening_end = min(total_duration, 300)
    ending_start = max(0, total_duration - 300)
    validate_proposal_citations(
        hook["evidence"],
        section_name="hook",
        target_start=0,
        target_end=opening_end,
    )
    validate_proposal_citations(
        cliffhanger["evidence"],
        section_name="cliffhanger",
        target_start=ending_start,
        target_end=total_duration,
    )
    if hook["within_first_3_seconds"] and not hook["evidence"]:
        raise EvaluationValidationError("Hook claims an opening hook without evidence")
    if cliffhanger["evidence"] and total_duration < 0:
        raise EvaluationValidationError("Invalid proposal duration")

    normalized = {
        key: value
        for key, value in evaluation.items()
        if key
        not in {
            "final_score",
            "classification",
            "model_reported_final_score",
            "model_reported_classification",
        }
    }
    normalized["final_score"] = final_score
    normalized["classification"] = expected_classification
    normalized["hook"] = dict(hook)
    normalized["cliffhanger"] = dict(cliffhanger)
    for section_name in ("hook", "cliffhanger"):
        section = normalized[section_name]
        section["semantic_evidence_status"] = _semantic_evidence_status(
            section["description"],
            section["evidence"],
        )
        if section["semantic_evidence_status"] != "SUPPORTED":
            normalized["human_review_required"] = True
    if "model_reported_final_score" in evaluation or "final_score" in evaluation:
        normalized["model_reported_final_score"] = reported_score
    if "model_reported_classification" in evaluation or "classification" in evaluation:
        normalized["model_reported_classification"] = reported_classification
    return normalized
