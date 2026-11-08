"""Ranking and aggregate reporting for validated candidate evaluations."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from src.evaluation.evaluation_schema import SCORE_WEIGHTS


def build_ranking(evaluations: list[dict[str, Any]]) -> dict[str, Any]:
    """Create stable overall/top/per-episode rankings and score distributions."""
    ordered = sorted(
        evaluations,
        key=lambda item: (
            -item["final_score"],
            item["episode_number"],
            item["start_timestamp"],
            item["candidate_id"],
        ),
    )
    per_episode: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for item in ordered:
        per_episode[item["episode_number"]].append(item)

    classifications = Counter(item["classification"] for item in evaluations)
    averages = {
        dimension: (
            sum(item["scores"][dimension] for item in evaluations) / len(evaluations)
            if evaluations
            else None
        )
        for dimension in SCORE_WEIGHTS
    }
    return {
        "evaluated_candidate_count": len(evaluations),
        "overall_ranking": ordered,
        "top_50": ordered[:50],
        "top_10": ordered[:10],
        "top_candidates_per_episode": {
            str(episode): candidates[:3]
            for episode, candidates in sorted(per_episode.items())
        },
        "classification_distribution": {
            name: classifications.get(name, 0)
            for name in ("STRONG", "PROMISING", "WEAK", "REJECT")
        },
        "average_scores_by_dimension": averages,
        "candidate_count_per_episode": {
            str(episode): len(candidates)
            for episode, candidates in sorted(per_episode.items())
        },
        "human_review_candidates": [
            item for item in ordered if item["human_review_required"]
        ],
        "ranking_note": "Scores rank individual continuous candidates; selection is not final promo approval.",
    }


def build_proposal_ranking(evaluations: list[dict[str, Any]]) -> dict[str, Any]:
    """Rank assembled promos without attributing cross-episode proposals to one episode."""
    ordered = sorted(
        evaluations,
        key=lambda item: (
            -item["final_score"],
            item["promo_category"],
            item["promo_id"],
        ),
    )
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for evaluation in ordered:
        by_category[evaluation["promo_category"]].append(evaluation)
    classifications = Counter(item["classification"] for item in evaluations)
    averages = {
        dimension: (
            sum(item["scores"][dimension] for item in evaluations) / len(evaluations)
            if evaluations
            else None
        )
        for dimension in SCORE_WEIGHTS
    }
    return {
        "evaluated_proposal_count": len(evaluations),
        "overall_ranking": ordered,
        "top_50": ordered[:50],
        "top_10": ordered[:10],
        "top_proposals_by_category": {
            category: records[:10]
            for category, records in sorted(by_category.items())
        },
        "proposal_count_by_category": {
            category: len(records)
            for category, records in sorted(by_category.items())
        },
        "classification_distribution": {
            name: classifications.get(name, 0)
            for name in ("STRONG", "PROMISING", "WEAK", "REJECT")
        },
        "average_scores_by_dimension": averages,
        "human_review_proposals": [
            item for item in ordered if item["human_review_required"]
        ],
        "ranking_note": (
            "Scores rank assembled proposal experiences; cross-episode source timestamps "
            "are not used as a global chronology."
        ),
    }
