"""Evaluate prepared promo candidates independently with resumable storage."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.evaluation.evaluation_prompts import (
    build_candidate_prompt,
    build_proposal_prompt,
    proposal_response_schema,
)
from src.evaluation.evaluation_schema import (
    EvaluationValidationError,
    normalize_proposal,
    preprocess_evidence_citations,
    validate_evaluation,
    validate_proposal_evaluation,
)
from src.evaluation.llm_client import (
    LLMClientError,
    LLMConfig,
    LLMQuotaExceededError,
    create_provider,
)
from src.evaluation.ranking import build_proposal_ranking, build_ranking


def _read_object(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def _candidate_key(candidate_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", candidate_id)


def _fingerprint(candidate: dict[str, Any]) -> str:
    payload = json.dumps(candidate, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_candidates(input_directory: Path) -> list[dict[str, Any]]:
    """Load prepared candidate records from every episode JSON."""
    if not input_directory.is_dir():
        raise NotADirectoryError(f"Prepared input directory does not exist: {input_directory}")
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    files = sorted(path for path in input_directory.glob("*.json") if path.is_file())
    if not files:
        raise ValueError(f"No prepared episode files found in {input_directory}")
    for path in files:
        episode_file = _read_object(path)
        if episode_file.get("preparation", {}).get("schema_version") is None:
            raise ValueError(f"Prepared input marker missing in {path}")
        records = episode_file.get("candidates")
        if not isinstance(records, list):
            raise ValueError(f"Candidate list missing in {path}")
        for candidate in records:
            if not isinstance(candidate, dict) or not isinstance(candidate.get("candidate_id"), str):
                raise ValueError(f"Malformed candidate record in {path}")
            candidate_id = candidate["candidate_id"]
            if candidate_id in seen:
                raise ValueError(f"Duplicate candidate ID across input files: {candidate_id}")
            seen.add(candidate_id)
            # Prompt formatting is an input-shape check in both dry-run and live runs.
            build_candidate_prompt(candidate)
            candidates.append(candidate)
    return candidates


def _load_completed(
    validated_directory: Path,
    candidate: dict[str, Any],
    fingerprint: str,
    *,
    provider: str,
    model: str,
) -> dict[str, Any] | None:
    path = validated_directory / f"{_candidate_key(candidate['candidate_id'])}.json"
    if not path.is_file():
        return None
    record = _read_object(path)
    if record.get("source_candidate_sha256") != fingerprint:
        raise ValueError(
            f"Existing evaluation for {candidate['candidate_id']} belongs to different input; "
            "use --force to replace it."
        )
    if record.get("provider") != provider or record.get("model") != model:
        raise ValueError(
            f"Existing evaluation for {candidate['candidate_id']} used a different provider/model; "
            "use a separate output directory or --force."
        )
    evaluation = record.get("evaluation")
    return validate_evaluation(evaluation, candidate)


def _save_raw_response(
    *,
    output_directory: Path,
    candidate: dict[str, Any],
    fingerprint: str,
    raw_text: str,
    response_metadata: dict[str, Any],
) -> None:
    key = _candidate_key(candidate["candidate_id"])
    _atomic_json(
        output_directory / "raw_responses" / f"{key}.json",
        {
            "candidate_id": candidate["candidate_id"],
            "source_candidate_sha256": fingerprint,
            "provider_metadata": response_metadata,
            "raw_response": raw_text,
            "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )


def _save_validated_result(
    *,
    output_directory: Path,
    candidate: dict[str, Any],
    fingerprint: str,
    evaluation: dict[str, Any],
    provider: str,
    model: str,
) -> None:
    key = _candidate_key(candidate["candidate_id"])
    _atomic_json(
        output_directory / "validated" / f"{key}.json",
        {
            "candidate_id": candidate["candidate_id"],
            "source_candidate_sha256": fingerprint,
            "provider": provider,
            "model": model,
            "evaluation": evaluation,
            "validated_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )


def _save_failures(output_directory: Path, failures: list[dict[str, Any]]) -> None:
    _atomic_json(output_directory / "failed_candidates.json", {"failures": failures})


def _record_evidence_repairs(
    output_directory: Path,
    candidate_id: str,
    repairs: list[dict[str, str]],
) -> None:
    if not repairs:
        return
    path = output_directory / "evidence_repairs.json"
    records: list[dict[str, Any]] = []
    if path.is_file():
        existing = _read_object(path).get("repairs")
        if not isinstance(existing, list) or not all(
            isinstance(item, dict) for item in existing
        ):
            raise ValueError(f"Malformed evidence repair log: {path}")
        records = existing
    records.append(
        {
            "candidate_id": candidate_id,
            "citations": repairs,
            "logged_at_utc": datetime.now(timezone.utc).isoformat(),
        }
    )
    _atomic_json(path, {"repairs": records})


def _read_failures(output_directory: Path) -> list[dict[str, Any]]:
    path = output_directory / "failed_candidates.json"
    if not path.is_file():
        return []
    value = _read_object(path).get("failures")
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ValueError(f"Malformed failure log: {path}")
    return value


def _refresh_reports(
    output_directory: Path,
    evaluations: list[dict[str, Any]],
    *,
    expected: int,
    failed: int,
    provider: str | None,
    model: str | None,
    dry_run: bool = False,
) -> None:
    _atomic_json(output_directory / "ranked_candidates.json", build_ranking(evaluations))
    _atomic_json(
        output_directory / "evaluation_summary.json",
        {
            "status": "dry_run" if dry_run else ("complete" if len(evaluations) == expected else "incomplete"),
            "provider": provider,
            "model": model,
            "free_tier_confirmation_required_for_gemini": provider == "gemini",
            "free_tier_confirmed_by_operator": (
                bool(os.getenv("PROMO_GEMINI_FREE_TIER_CONFIRMED") == "YES")
                if provider == "gemini"
                else None
            ),
            "expected_candidate_count": expected,
            "validated_evaluation_count": len(evaluations),
            "failed_candidate_count": failed,
            "evaluations_are_real_model_outputs": not dry_run,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )


def run_evaluations(
    *,
    input_directory: Path,
    output_directory: Path,
    config: LLMConfig | None,
    dry_run: bool = False,
    limit: int | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Evaluate candidates incrementally or validate their inputs without scoring."""
    candidates = load_candidates(input_directory)
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be a positive integer")
        candidates = candidates[:limit]
    output_directory.mkdir(parents=True, exist_ok=True)

    if dry_run:
        # A dry run deliberately does not create evaluation records or rankings.
        summary = {
            "status": "dry_run",
            "provider": config.provider if config else None,
            "model": config.model if config else None,
            "free_tier_confirmed_by_operator": (
                config.free_tier_confirmed if config and config.provider == "gemini" else None
            ),
            "candidates_validated": len(candidates),
            "evaluations_are_real_model_outputs": False,
            "note": "No API calls and no evaluation scores were produced.",
        }
        _atomic_json(output_directory / "dry_run_summary.json", summary)
        return summary

    if config is None:
        raise ValueError("Provider configuration is required outside dry-run mode")
    client = create_provider(config)
    validated_directory = output_directory / "validated"
    raw_directory = output_directory / "raw_responses"
    validated_directory.mkdir(parents=True, exist_ok=True)
    raw_directory.mkdir(parents=True, exist_ok=True)

    completed: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        fingerprint = _fingerprint(candidate)
        evaluation_path = validated_directory / f"{_candidate_key(candidate['candidate_id'])}.json"
        if evaluation_path.exists() and force:
            evaluation_path.unlink()
            raw_path = raw_directory / evaluation_path.name
            if raw_path.exists():
                raw_path.unlink()
            continue
        existing = _load_completed(
            validated_directory,
            candidate,
            fingerprint,
            provider=config.provider,
            model=config.model,
        )
        if existing is not None:
            completed[candidate["candidate_id"]] = existing

    failures_by_id = {
        item.get("candidate_id"): item
        for item in _read_failures(output_directory)
        if isinstance(item.get("candidate_id"), str)
    }
    _save_failures(output_directory, list(failures_by_id.values()))
    quota_exhausted = False
    pending = [
        candidate
        for candidate in candidates
        if candidate["candidate_id"] not in completed
    ]
    for batch_start in range(0, len(pending), config.batch_size):
        batch = pending[batch_start : batch_start + config.batch_size]
        for candidate in batch:
            candidate_id = candidate["candidate_id"]
            fingerprint = _fingerprint(candidate)
            try:
                raw_text, response_metadata = client.evaluate(candidate)
                _save_raw_response(
                    output_directory=output_directory,
                    candidate=candidate,
                    fingerprint=fingerprint,
                    raw_text=raw_text,
                    response_metadata=response_metadata,
                )
                try:
                    evaluation = json.loads(raw_text)
                except json.JSONDecodeError as error:
                    raise EvaluationValidationError("Model response was not valid JSON") from error
                evaluation, evidence_repairs = preprocess_evidence_citations(
                    evaluation, candidate
                )
                _record_evidence_repairs(
                    output_directory,
                    candidate_id,
                    evidence_repairs,
                )
                evaluation = validate_evaluation(evaluation, candidate)
                returned_model = response_metadata.get("returned_model")
                if (
                    returned_model is not None
                    and (
                        not isinstance(returned_model, str)
                        or (
                            returned_model != config.model
                            and not returned_model.startswith(f"{config.model}-")
                        )
                    )
                ):
                    raise EvaluationValidationError(
                        "Provider returned a model different from the explicitly requested model"
                    )
                _save_validated_result(
                    output_directory=output_directory,
                    candidate=candidate,
                    fingerprint=fingerprint,
                    evaluation=evaluation,
                    provider=config.provider,
                    model=config.model,
                )
                completed[candidate_id] = evaluation
                failures_by_id.pop(candidate_id, None)
            except LLMQuotaExceededError as error:
                failures_by_id[candidate_id] = {
                    "candidate_id": candidate_id,
                    "error_type": type(error).__name__,
                    "message": str(error),
                }
                quota_exhausted = True
            except (LLMClientError, EvaluationValidationError, ValueError) as error:
                failures_by_id[candidate_id] = {
                    "candidate_id": candidate_id,
                    "error_type": type(error).__name__,
                    "message": str(error),
                }
            _save_failures(output_directory, list(failures_by_id.values()))
            _refresh_reports(
                output_directory,
                list(completed.values()),
                expected=len(candidates),
                failed=len(failures_by_id),
                provider=config.provider,
                model=config.model,
            )
            if quota_exhausted:
                break
        if quota_exhausted:
            break
    _refresh_reports(
        output_directory,
        list(completed.values()),
        expected=len(candidates),
        failed=len(failures_by_id),
        provider=config.provider,
        model=config.model,
    )
    return {
        "status": "complete"
        if not quota_exhausted and not failures_by_id and len(completed) == len(candidates)
        else "incomplete",
        "expected": len(candidates),
        "validated": len(completed),
        "failed": len(failures_by_id),
        "skipped_resumed": len(candidates) - len(pending),
        "stopped_for_quota": quota_exhausted,
        "provider": config.provider,
        "model": config.model,
    }


def load_proposals(
    *,
    proposal_file: Path,
    parsed_directory: Path,
    candidate_directory: Path,
) -> list[dict[str, Any]]:
    """Load shortlisted proposals and verify every reference against source JSON."""
    proposal_document = _read_object(proposal_file)
    raw_proposals = proposal_document.get("proposals")
    if not isinstance(raw_proposals, list) or not raw_proposals:
        raise ValueError(f"No proposals found in {proposal_file}")

    source_events: dict[tuple[str, str], dict[str, Any]] = {}
    for path in sorted(parsed_directory.glob("*.json")):
        episode_document = _read_object(path)
        episode_id = episode_document.get("episode_id")
        episode_number = episode_document.get("episode_number")
        if isinstance(episode_number, bool) or not isinstance(episode_number, int):
            metadata = episode_document.get("metadata", {})
            number_value = metadata.get("episode_number", {}).get("value")
            match = re.search(r"\d+", number_value) if isinstance(number_value, str) else None
            episode_number = int(match.group()) if match else None
        if not isinstance(episode_id, str) or not isinstance(episode_number, int):
            raise ValueError(f"Parsed episode identity is missing in {path}")
        for scene in episode_document.get("scenes", []):
            for event in scene.get("events", []):
                event_id = event.get("id")
                if not isinstance(event_id, str):
                    raise ValueError(f"Malformed source event in {path}")
                key = (episode_id, event_id)
                if key in source_events:
                    raise ValueError(f"Duplicate source event ID {event_id!r} in {episode_id}")
                source_events[key] = {
                    "event": event,
                    "episode_number": episode_number,
                }

    source_candidates: dict[str, dict[str, Any]] = {}
    for path in sorted(candidate_directory.glob("*.json")):
        if path.name == "summary.json":
            continue
        candidate_document = _read_object(path)
        candidate_records = candidate_document.get("candidates")
        if not isinstance(candidate_records, list):
            raise ValueError(f"Candidate list is missing in {path}")
        for candidate in candidate_records:
            candidate_id = candidate.get("candidate_id")
            if not isinstance(candidate_id, str) or candidate_id in source_candidates:
                raise ValueError(f"Invalid or duplicate source candidate in {path}")
            source_candidates[candidate_id] = candidate

    proposals: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw_proposal in raw_proposals:
        if not isinstance(raw_proposal, dict) or raw_proposal.get("validation_status") != "VALID":
            raise ValueError("Proposal input contains a malformed or non-VALID proposal")
        promo_id = raw_proposal.get("promo_id")
        if not isinstance(promo_id, str) or promo_id in seen:
            raise ValueError(f"Invalid or duplicate promo_id: {promo_id!r}")
        seen.add(promo_id)
        normalized = normalize_proposal(
            raw_proposal,
            source_events=source_events,
            source_candidates=source_candidates,
        )
        build_proposal_prompt(normalized)
        proposal_response_schema()
        proposals.append(normalized)
    return proposals


def _proposal_key(promo_id: str) -> str:
    return _candidate_key(promo_id)


def _load_completed_proposal(
    validated_directory: Path,
    proposal: dict[str, Any],
    fingerprint: str,
    *,
    provider: str,
    model: str,
) -> dict[str, Any] | None:
    path = validated_directory / f"{_proposal_key(proposal['promo_id'])}.json"
    if not path.is_file():
        return None
    record = _read_object(path)
    if record.get("source_proposal_sha256") != fingerprint:
        raise ValueError(
            f"Existing evaluation for {proposal['promo_id']} belongs to different input"
        )
    if record.get("provider") != provider or record.get("model") != model:
        raise ValueError(
            f"Existing evaluation for {proposal['promo_id']} used a different provider/model"
        )
    evaluation = validate_proposal_evaluation(record.get("evaluation"), proposal)
    return {**record, "evaluation": evaluation}


def _save_proposal_raw_response(
    *,
    output_directory: Path,
    proposal: dict[str, Any],
    fingerprint: str,
    raw_text: str,
    response_metadata: dict[str, Any],
) -> None:
    _atomic_json(
        output_directory / "raw_responses" / f"{_proposal_key(proposal['promo_id'])}.json",
        {
            "promo_id": proposal["promo_id"],
            "source_proposal_sha256": fingerprint,
            "provider_metadata": response_metadata,
            "raw_response": raw_text,
            "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )


def _save_validated_proposal(
    *,
    output_directory: Path,
    proposal: dict[str, Any],
    fingerprint: str,
    evaluation: dict[str, Any],
    provider: str,
    model: str,
) -> dict[str, Any]:
    record = {
        "promo_id": proposal["promo_id"],
        "promo_category": proposal["promo_category"],
        "total_duration_seconds": proposal["total_duration_seconds"],
        "segment_ids": [segment["segment_id"] for segment in proposal["segments"]],
        "source_episode_ids": list(
            dict.fromkeys(segment["episode_id"] for segment in proposal["segments"])
        ),
        "source_candidate_ids": proposal["source_candidate_ids"],
        "source_event_ids": proposal["source_event_ids"],
        "source_proposal_sha256": fingerprint,
        "provider": provider,
        "model": model,
        "evaluation": evaluation,
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_json(
        output_directory / "validated" / f"{_proposal_key(proposal['promo_id'])}.json",
        record,
    )
    return record


def _refresh_proposal_reports(
    output_directory: Path,
    proposals: list[dict[str, Any]],
    completed: dict[str, dict[str, Any]],
    *,
    provider: str | None,
    model: str | None,
) -> None:
    ranked_items = []
    for proposal in proposals:
        record = completed.get(proposal["promo_id"])
        if record is None:
            continue
        ranked_items.append(
            {
                "promo_id": proposal["promo_id"],
                "promo_category": proposal["promo_category"],
                "total_duration_seconds": proposal["total_duration_seconds"],
                "segment_ids": [
                    segment["segment_id"] for segment in proposal["segments"]
                ],
                "source_episode_ids": list(
                    dict.fromkeys(
                        segment["episode_id"] for segment in proposal["segments"]
                    )
                ),
                **record["evaluation"],
            }
        )
    _atomic_json(
        output_directory / "ranked_proposals.json",
        build_proposal_ranking(ranked_items),
    )
    _atomic_json(
        output_directory / "proposal_evaluation_summary.json",
        {
            "status": "complete" if len(completed) == len(proposals) else "incomplete",
            "provider": provider,
            "model": model,
            "expected_proposal_count": len(proposals),
            "validated_proposal_count": len(completed),
            "failed_proposal_count": max(0, len(proposals) - len(completed)),
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )


def run_proposal_evaluations(
    *,
    proposal_file: Path,
    parsed_directory: Path,
    candidate_directory: Path,
    output_directory: Path,
    config: LLMConfig | None,
    dry_run: bool = False,
    limit: int | None = None,
    force: bool = False,
    progress: Callable[[int, int, str], None] | None = None,
) -> dict[str, Any]:
    """Evaluate assembled proposals with independent source/playback validation."""
    proposals = load_proposals(
        proposal_file=proposal_file,
        parsed_directory=parsed_directory,
        candidate_directory=candidate_directory,
    )
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be a positive integer")
        proposals = proposals[:limit]
    category_counts: dict[str, int] = {}
    for proposal in proposals:
        category = proposal["promo_category"]
        category_counts[category] = category_counts.get(category, 0) + 1
    if dry_run:
        summary = {
            "status": "dry_run",
            "provider": config.provider if config else None,
            "model": config.model if config else None,
            "proposals_normalized": len(proposals),
            "proposals_compatible": len(proposals),
            "source_event_references_validated": sum(
                len(proposal["source_event_ids"]) for proposal in proposals
            ),
            "segments_validated": sum(len(proposal["segments"]) for proposal in proposals),
            "category_counts": category_counts,
            "playback_offsets_verified": True,
            "evaluations_are_real_model_outputs": False,
            "api_calls": 0,
            "note": "All source references, timestamps, offsets, prompt shapes, and schemas were validated offline.",
        }
        _atomic_json(output_directory / "proposal_dry_run_summary.json", summary)
        return summary
    if config is None:
        raise ValueError("Provider configuration is required outside dry-run mode")
    client = create_provider(config)
    if not hasattr(client, "evaluate_proposal"):
        raise ValueError("Configured provider does not support assembled proposal evaluation")

    validated_directory = output_directory / "validated"
    raw_directory = output_directory / "raw_responses"
    validated_directory.mkdir(parents=True, exist_ok=True)
    raw_directory.mkdir(parents=True, exist_ok=True)
    completed: dict[str, dict[str, Any]] = {}
    for proposal in proposals:
        proposal_id = proposal["promo_id"]
        path = validated_directory / f"{_proposal_key(proposal_id)}.json"
        if path.exists() and force:
            path.unlink()
            raw_path = raw_directory / path.name
            if raw_path.exists():
                raw_path.unlink()
            continue
        record = _load_completed_proposal(
            validated_directory,
            proposal,
            _fingerprint(proposal),
            provider=config.provider,
            model=config.model,
        )
        if record is not None:
            completed[proposal_id] = record

    failure_path = output_directory / "failed_proposals.json"
    failures: dict[str, dict[str, Any]] = {}
    if failure_path.is_file():
        existing_failures = _read_object(failure_path).get("failures", [])
        if not isinstance(existing_failures, list):
            raise ValueError(f"Malformed proposal failure log: {failure_path}")
        failures = {
            item["promo_id"]: item
            for item in existing_failures
            if isinstance(item, dict) and isinstance(item.get("promo_id"), str)
        }
    pending = [
        proposal for proposal in proposals if proposal["promo_id"] not in completed
    ]
    quota_exhausted = False
    processed_count = 0
    resumed_count = len(proposals) - len(pending)
    for proposal in pending:
        promo_id = proposal["promo_id"]
        fingerprint = _fingerprint(proposal)
        if progress:
            progress(resumed_count + processed_count, len(proposals), promo_id)
        try:
            raw_text, response_metadata = client.evaluate_proposal(proposal)
            _save_proposal_raw_response(
                output_directory=output_directory,
                proposal=proposal,
                fingerprint=fingerprint,
                raw_text=raw_text,
                response_metadata=response_metadata,
            )
            try:
                model_evaluation = json.loads(raw_text)
            except json.JSONDecodeError as error:
                raise EvaluationValidationError("Model response was not valid JSON") from error
            evaluation = validate_proposal_evaluation(model_evaluation, proposal)
            returned_model = response_metadata.get("returned_model")
            if returned_model is not None and (
                not isinstance(returned_model, str)
                or (
                    returned_model != config.model
                    and not returned_model.startswith(f"{config.model}-")
                )
            ):
                raise EvaluationValidationError(
                    "Provider returned a model different from the explicitly requested model"
                )
            completed[promo_id] = _save_validated_proposal(
                output_directory=output_directory,
                proposal=proposal,
                fingerprint=fingerprint,
                evaluation=evaluation,
                provider=config.provider,
                model=config.model,
            )
            failures.pop(promo_id, None)
        except LLMQuotaExceededError as error:
            failures[promo_id] = {
                "promo_id": promo_id,
                "error_type": type(error).__name__,
                "message": str(error),
            }
            quota_exhausted = True
        except (LLMClientError, EvaluationValidationError, ValueError) as error:
            failures[promo_id] = {
                "promo_id": promo_id,
                "error_type": type(error).__name__,
                "message": str(error),
            }
        _atomic_json(failure_path, {"failures": list(failures.values())})
        _refresh_proposal_reports(
            output_directory,
            proposals,
            completed,
            provider=config.provider,
            model=config.model,
        )
        processed_count += 1
        if progress:
            progress(
                resumed_count + processed_count,
                len(proposals),
                promo_id,
            )
        if quota_exhausted:
            break
    _refresh_proposal_reports(
        output_directory,
        proposals,
        completed,
        provider=config.provider,
        model=config.model,
    )
    return {
        "status": "complete"
        if not quota_exhausted and not failures and len(completed) == len(proposals)
        else "incomplete",
        "expected": len(proposals),
        "validated": len(completed),
        "failed": len(failures),
        "skipped_resumed": len(proposals) - len(pending),
        "stopped_for_quota": quota_exhausted,
        "provider": config.provider,
        "model": config.model,
    }


def reprocess_saved_response(
    *,
    input_directory: Path,
    output_directory: Path,
    candidate_id: str,
) -> dict[str, Any]:
    """Validate a saved provider response locally without making an API request."""
    candidates = load_candidates(input_directory)
    candidate_by_id = {candidate["candidate_id"]: candidate for candidate in candidates}
    candidate = candidate_by_id.get(candidate_id)
    if candidate is None:
        raise ValueError(f"Candidate not found in prepared inputs: {candidate_id}")
    fingerprint = _fingerprint(candidate)
    key = _candidate_key(candidate_id)
    validated_path = output_directory / "validated" / f"{key}.json"

    raw_record = _read_object(output_directory / "raw_responses" / f"{key}.json")
    if raw_record.get("candidate_id") != candidate_id:
        raise ValueError("Saved raw response candidate ID does not match the requested candidate")
    if raw_record.get("source_candidate_sha256") != fingerprint:
        raise ValueError("Saved raw response belongs to different prepared candidate content")
    metadata = raw_record.get("provider_metadata")
    if not isinstance(metadata, dict):
        raise ValueError("Saved raw response is missing provider metadata")
    provider = metadata.get("provider")
    model = metadata.get("requested_model")
    returned_model = metadata.get("returned_model")
    if not isinstance(provider, str) or not isinstance(model, str):
        raise ValueError("Saved raw response has invalid provider/model metadata")
    if returned_model is not None and (
        not isinstance(returned_model, str)
        or (returned_model != model and not returned_model.startswith(f"{model}-"))
    ):
        raise EvaluationValidationError(
            "Provider returned a model different from the explicitly requested model"
        )

    if validated_path.is_file():
        existing = _load_completed(
            validated_path.parent,
            candidate,
            fingerprint,
            provider=provider,
            model=model,
        )
        return {
            "status": "already_validated",
            "candidate_id": candidate_id,
            "evaluation": existing,
            "api_calls": 0,
        }

    raw_text = raw_record.get("raw_response")
    if not isinstance(raw_text, str):
        raise ValueError("Saved raw response text is missing")
    try:
        model_evaluation = json.loads(raw_text)
    except json.JSONDecodeError as error:
        raise EvaluationValidationError("Saved model response was not valid JSON") from error
    evaluation = validate_evaluation(model_evaluation, candidate)
    _save_validated_result(
        output_directory=output_directory,
        candidate=candidate,
        fingerprint=fingerprint,
        evaluation=evaluation,
        provider=provider,
        model=model,
    )

    failures = [
        item
        for item in _read_failures(output_directory)
        if item.get("candidate_id") != candidate_id
    ]
    _save_failures(output_directory, failures)
    completed: list[dict[str, Any]] = []
    for saved_candidate in candidates:
        saved_path = (
            output_directory
            / "validated"
            / f"{_candidate_key(saved_candidate['candidate_id'])}.json"
        )
        if not saved_path.is_file():
            continue
        record = _read_object(saved_path)
        if record.get("source_candidate_sha256") != _fingerprint(saved_candidate):
            raise ValueError(
                f"Existing evaluation for {saved_candidate['candidate_id']} "
                "belongs to different input"
            )
        completed.append(validate_evaluation(record.get("evaluation"), saved_candidate))
    _refresh_reports(
        output_directory,
        completed,
        expected=len(candidates),
        failed=len(failures),
        provider=provider,
        model=model,
    )
    return {
        "status": "reprocessed_saved_response",
        "candidate_id": candidate_id,
        "final_score": evaluation["final_score"],
        "model_reported_final_score": evaluation.get("model_reported_final_score"),
        "classification": evaluation["classification"],
        "validated": len(completed),
        "expected": len(candidates),
        "api_calls": 0,
    }


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def main(argv: list[str] | None = None) -> int:
    """Run evaluation or validate inputs in explicit dry-run mode."""
    root = _project_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=root / "data" / "output" / "llm_inputs",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--proposal-input",
        type=Path,
        help="Evaluate a constructed proposal shortlist instead of prepared single candidates.",
    )
    parser.add_argument(
        "--parsed-dir",
        type=Path,
        default=root / "data" / "output" / "parsed",
    )
    parser.add_argument(
        "--candidate-dir",
        type=Path,
        default=root / "data" / "output" / "candidates",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--reprocess-raw-candidate-id")
    arguments = parser.parse_args(argv)
    try:
        if arguments.reprocess_raw_candidate_id:
            if (
                arguments.dry_run
                or arguments.limit is not None
                or arguments.force
                or arguments.proposal_input is not None
            ):
                raise ValueError(
                    "--reprocess-raw-candidate-id cannot be combined with "
                    "--dry-run, --limit, --force, or --proposal-input"
                )
            output_directory = arguments.output_dir or root / "data" / "output" / "evaluations"
            result = reprocess_saved_response(
                input_directory=arguments.input_dir,
                output_directory=output_directory,
                candidate_id=arguments.reprocess_raw_candidate_id,
            )
        else:
            config = LLMConfig.from_environment()
            if arguments.proposal_input is not None:
                output_directory = (
                    arguments.output_dir
                    or root / "data" / "output" / "proposal_evaluations"
                )
                result = run_proposal_evaluations(
                    proposal_file=arguments.proposal_input,
                    parsed_directory=arguments.parsed_dir,
                    candidate_directory=arguments.candidate_dir,
                    output_directory=output_directory,
                    config=config,
                    dry_run=arguments.dry_run,
                    limit=arguments.limit,
                    force=arguments.force,
                )
            else:
                output_directory = arguments.output_dir or root / "data" / "output" / "evaluations"
                result = run_evaluations(
                    input_directory=arguments.input_dir,
                    output_directory=output_directory,
                    config=config,
                    dry_run=arguments.dry_run,
                    limit=arguments.limit,
                    force=arguments.force,
                )
    except (OSError, ValueError, LLMClientError) as error:
        print(f"Evaluation failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
