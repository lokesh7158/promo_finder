"""Prompt construction for evidence-grounded individual candidate evaluation."""

from __future__ import annotations

import json
from typing import Any

from src.evaluation.evaluation_schema import (
    OPENAI_RESPONSE_SCHEMA,
    PROPOSAL_RESPONSE_SCHEMA,
)


SYSTEM_PROMPT = """You evaluate one continuous screenplay promo candidate independently.
Use only the supplied candidate events. Do not invent dialogue, actions, relationships,
or unseen context. Treat only facts stated or depicted in these events as established;
label interpretations as interpretations and do not turn plausible inferences into facts.
Do not use screenplay events outside this candidate to justify any judgment, especially
standalone comprehensibility. Overlapping events are concurrent source events, not errors.
Assess the opening hook from events overlapping the first three seconds only; do not
credit a later moment as an opening hook. Inspect the final three seconds for ending
evidence. For a cliffhanger, identify the unresolved dramatic question and cite the
candidate event or events that directly establish that conflict. Also cite an event in
the final three seconds that shows the dramatic beat or marks the cutoff. The cutoff
event may be a transition, reaction, or atmosphere shot; do not treat it alone as proof
of the conflict. Order cliffhanger evidence with conflict-establishing evidence first
and cutoff/ending evidence afterward when both are needed. Do not force the last event
to be the sole evidence. If no suitable hook or unresolved ending exists, explicitly say
so and score accordingly. Every excerpt must directly support the claim it accompanies;
use exact, short excerpts from cited event content and cite only supplied event IDs and
exact start/end timestamps. Every evidence timestamp must fall within the candidate's
exact start and end boundaries. If an event starts before the candidate but overlaps it,
cite an exact event endpoint that falls within the candidate only when that endpoint
supports the claim. Otherwise, select another supporting event with a timestamp inside
the candidate. Never move, round, or fabricate timestamps, and never cite events outside
the candidate, even when they provide useful episode context. If evidence supports only
an interpretation, say so.
Do not compare this candidate with other candidates. Return only the required JSON.

Scores are integers 0–5:
hook_strength 25%, emotional_impact 20%, cliffhanger_strength 20%,
standalone_comprehensibility 15%, story_coherence 10%, promotional_potential 10%.
Return only the six individual dimension scores; do not calculate or return a
final score or classification. Python calculates both from the scores. A
weak/reject result is valid; do not force positivity.
Confidence calibration:
- HIGH only when the material claims are directly and clearly supported by this
  candidate's events.
- MEDIUM when a reasonable interpretation depends on incomplete context.
- LOW when an important claim is speculative or poorly supported.
Do not state uncertain character identities or relationships as facts. Describe the
ambiguity in uncertainties, lower confidence accordingly, and set human_review_required
when it could materially affect the judgment.

Evidence descriptions must stay within what their cited events actually state or depict.
For hooks, do not add expressions, reactions, gaze direction, motives, relationships, or
emotional states unless those details appear in the cited visual/action/dialogue evidence.
For cliffhangers, distinguish an explicitly established unresolved event from an
interpretation inferred from earlier events; label the latter as an interpretation.
Every significant factual claim in a hook or cliffhanger description must be supported
by at least one of its citations. Do not rely on one citation to support a separate
claim merely because it appears in the same candidate. When a cited excerpt only partly
supports a description, narrow the description to the supported facts or state the
uncertainty explicitly. If no cited evidence supports the material claim, say that the
evidence is insufficient instead of exaggerating the moment."""

PROPOSAL_SYSTEM_PROMPT = """You evaluate a complete assembled promo as one viewing experience.
The supplied segments are in PLAYBACK ORDER. Playback order is the order in which the
viewer sees them; it is not original source chronology. Every event retains its original
episode identity and source timestamp. Never compare source timestamps across episodes
or assume segments from different episodes are chronologically adjacent.

Evaluate the assembled promo on the six supplied scoring dimensions. Assess the hook
only from evidence in the first three seconds of assembled playback. Assess the
cliffhanger only from evidence in the final three seconds of assembled playback.
Emotional impact, standalone comprehensibility, story coherence, and promotional
potential must consider the complete assembled sequence. For multi-segment proposals,
explicitly assess whether transitions make narrative sense and whether a first-time
viewer can follow the assembled story. Shared character names or labels do not prove
that a specific conflict continues between segments.

Use only the supplied source events. Do not invent actions, dialogue, reactions,
expressions, relationships, motives, identities, or emotional states. Distinguish
directly shown facts from interpretations, state uncertainty where context is missing,
and set human_review_required when segment connections or material claims are uncertain.
The hook and cliffhanger descriptions must be limited to claims supported by their
citations. Cite the source event that establishes the dramatic beat, not merely an
atmosphere or transition event.

For each event, use ONLY the separately provided eligible_citation_endpoints list
when choosing a timestamp for evidence. Each entry is an eligible original event
endpoint with its exact source timestamp and corresponding playback timestamp.
Never cite a timestamp from source_endpoint_provenance, an endpoint not present in
eligible_citation_endpoints, or any clipped overlap boundary. If the eligible list
is empty, choose another supporting event.
The clipped_playback_overlap_start_timestamp and
clipped_playback_overlap_end_timestamp describe only the part of an event visible
inside a segment; they are NOT source endpoints and are NEVER valid evidence
timestamps. An event start before the segment has no eligible start citation even
when the event overlaps the segment. If neither endpoint is eligible, or no eligible
endpoint supports the specific claim, cite another supporting event. Do not invent,
shift, round, or substitute timestamps.

Every evidence item must include the exact segment_id and playback_order, original
episode_id and event_id, an eligible exact source_timestamp equal to that event's
original start or end, that endpoint's exact mapped playback_timestamp, and an exact
short excerpt copied verbatim as a contiguous substring of the original event
content. Preserve every character, including punctuation, capitalization, spacing,
and ellipses. Do not paraphrase, normalize punctuation, or reconstruct dialogue.
If a long excerpt is difficult to reproduce exactly, choose a shorter exact
substring. Hook citations must have playback timestamps
within the first three seconds. Cliffhanger citations must have playback timestamps
within the final three seconds. Never repair or guess either timestamp. Never cite an
event from another segment or episode by matching timestamps alone.
If there is no supported hook or unresolved ending, say so and leave its evidence
empty. Return only the required JSON, including the promo_id, category, and total
assembled duration. Return the six dimension scores only; Python calculates final
score and classification.

Confidence calibration: HIGH only when material claims are directly and clearly
supported; MEDIUM for reasonable interpretations with incomplete context; LOW for
speculative or poorly supported claims. Every significant factual claim in hook and
cliffhanger descriptions must be supported by its cited excerpt. The local semantic
status check is a lexical screening heuristic; do not use that as a reason to
overstate evidence."""


def _event_text(item: dict[str, Any]) -> dict[str, Any]:
    event = item.get("event", {})
    content = event.get("content", {})
    return {
        "event_id": event.get("id"),
        "scene_id": item.get("scene_id"),
        "type": event.get("type"),
        "start": event.get("start", {}).get("raw"),
        "end": event.get("end", {}).get("raw"),
        "speaker_label": event.get("speaker_label_raw"),
        "annotations": event.get("annotations_raw", []),
        "content": content.get("raw") if isinstance(content, dict) else content,
        "validation_flags": event.get("validation_flags", []),
    }


def build_candidate_prompt(candidate: dict[str, Any]) -> str:
    """Format one prepared candidate with event IDs and source text intact."""
    episode = candidate.get("episode", {})
    time_range = candidate.get("time_range", {})
    start = time_range.get("start", {}).get("raw")
    end = time_range.get("end", {}).get("raw")
    events = candidate.get("events")
    if not isinstance(events, list) or not events:
        raise ValueError(f"{candidate.get('candidate_id', '<unknown>')} has no events")
    scenes = [
        {
            "scene_id": scene.get("scene_id"),
            "scene_number": scene.get("scene_number"),
            "heading": scene.get("heading"),
        }
        for scene in candidate.get("scenes", [])
    ]
    payload = {
        "candidate_id": candidate.get("candidate_id"),
        "episode_number": episode.get("episode_number"),
        "start_timestamp": start,
        "end_timestamp": end,
        "duration_seconds": time_range.get("duration_seconds"),
        "timeline_region": candidate.get("timeline_region"),
        "duration_band": candidate.get("duration_band"),
        "selection_reason": candidate.get("selection_reason"),
        "scenes": scenes,
        "events_in_chronological_order": [_event_text(item) for item in events],
        "required_evidence_shape": {
            "event_id": "an event_id from this candidate",
            "timestamp": "that event's exact start or end timestamp",
            "excerpt": "short exact excerpt from the cited event content",
        },
        "opening_window": f"{start} through 3.00 seconds later (bounded by {end})",
        "ending_window": f"last 3.00 seconds through {end}",
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def build_proposal_prompt(proposal: dict[str, Any]) -> str:
    """Serialize a normalized assembled promo with both timeline coordinates."""
    segments = []
    for segment in proposal.get("segments", []):
        events = []
        for item in segment.get("events", []):
            event = item.get("event", {})
            content = event.get("content", {})
            start_eligible = item.get("source_start_endpoint_eligible") is True
            end_eligible = item.get("source_end_endpoint_eligible") is True
            eligible_citation_endpoints = []
            if start_eligible:
                eligible_citation_endpoints.append(
                    {
                        "endpoint": "start",
                        "source_timestamp": item.get("source_start_timestamp"),
                        "playback_timestamp": item.get(
                            "playback_timestamp_for_source_start"
                        ),
                    }
                )
            if end_eligible:
                eligible_citation_endpoints.append(
                    {
                        "endpoint": "end",
                        "source_timestamp": item.get("source_end_timestamp"),
                        "playback_timestamp": item.get(
                            "playback_timestamp_for_source_end"
                        ),
                    }
                )
            events.append(
                {
                    "event_id": item.get("event_id"),
                    "segment_id": segment.get("segment_id"),
                    "playback_order": segment.get("playback_order"),
                    "episode_id": segment.get("episode_id"),
                    "event_type": event.get("type"),
                    "speaker_label": event.get("speaker_label_raw"),
                    "annotations": event.get("annotations_raw", []),
                    "eligible_citation_endpoints": eligible_citation_endpoints,
                    "source_endpoint_provenance": {
                        "start": {
                            "source_timestamp": item.get("source_start_timestamp"),
                            "eligible": item.get("source_start_endpoint_eligible"),
                            "playback_timestamp_if_eligible": item.get(
                                "playback_timestamp_for_source_start"
                            ),
                        },
                        "end": {
                            "source_timestamp": item.get("source_end_timestamp"),
                            "eligible": item.get("source_end_endpoint_eligible"),
                            "playback_timestamp_if_eligible": item.get(
                                "playback_timestamp_for_source_end"
                            ),
                        },
                    },
                    "clipped_playback_overlap_start_timestamp": item.get(
                        "clipped_playback_overlap_start_timestamp"
                    ),
                    "clipped_playback_overlap_end_timestamp": item.get(
                        "clipped_playback_overlap_end_timestamp"
                    ),
                    "content": content.get("raw") if isinstance(content, dict) else content,
                    "validation_flags": event.get("validation_flags", []),
                }
            )
        segments.append(
            {
                "segment_id": segment.get("segment_id"),
                "playback_order": segment.get("playback_order"),
                "episode_id": segment.get("episode_id"),
                "episode_number": segment.get("episode_number"),
                "source_start_timestamp": segment.get("source_start_timestamp"),
                "source_end_timestamp": segment.get("source_end_timestamp"),
                "source_duration_seconds": segment.get("source_duration_seconds"),
                "playback_start_timestamp": segment.get("playback_start_timestamp"),
                "playback_end_timestamp": segment.get("playback_end_timestamp"),
                "source_candidate_ids": segment.get("source_candidate_ids"),
                "source_event_ids": segment.get("source_event_ids"),
                "events_in_source_order": events,
            }
        )
    duration = proposal.get("total_duration_seconds")
    payload = {
        "promo_id": proposal.get("promo_id"),
        "promo_category": proposal.get("promo_category"),
        "total_duration_seconds": duration,
        "segments_in_playback_order": segments,
        "opening_hook_window_playback": ["00:00.00", "00:03.00"],
        "ending_cliffhanger_window_playback": [
            "last 3 seconds of assembled playback",
            duration,
        ],
        "required_evidence_shape": {
            "segment_id": "exact segment_id from the cited segment",
            "playback_order": "that segment's one-based playback_order",
            "episode_id": "the cited event's original source episode_id",
            "event_id": "the exact source event ID",
            "source_timestamp": "exact event start or end timestamp, within the segment",
            "playback_timestamp": "exact mapping of source_timestamp onto assembled playback",
            "excerpt": "exact short excerpt from the cited event content",
        },
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def build_openai_messages(candidate: dict[str, Any]) -> list[dict[str, str]]:
    """Return independent system and user messages for a single candidate."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_candidate_prompt(candidate)},
    ]


def build_proposal_messages(proposal: dict[str, Any]) -> list[dict[str, str]]:
    """Return provider-neutral messages for assembled-promo evaluation."""
    return [
        {"role": "system", "content": PROPOSAL_SYSTEM_PROMPT},
        {"role": "user", "content": build_proposal_prompt(proposal)},
    ]


def response_schema() -> dict[str, Any]:
    """Return a copy of the strict provider response schema."""
    return json.loads(json.dumps(OPENAI_RESPONSE_SCHEMA))


def proposal_response_schema() -> dict[str, Any]:
    """Return a copy of the strict assembled-promo response schema."""
    return json.loads(json.dumps(PROPOSAL_RESPONSE_SCHEMA))
