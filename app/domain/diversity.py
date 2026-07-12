import random
from datetime import UTC, datetime

import numpy as np

from app.domain.ports import (
    CandidateEmbedding,
    CandidatePool,
    WatchedEmbedding,
    WatchHistoryLookup,
)


class NoWatchHistoryError(RuntimeError):
    """Raised when the watch_history collection has nothing in the current window
    — there's no aversion vector to build a recommendation away from."""


def cosine_similarity(a: list[float], b: list[float]) -> float:
    va, vb = np.array(a), np.array(b)
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if denom == 0.0:
        return 0.0
    return float(np.dot(va, vb) / denom)


def cosine_distance(a: list[float], b: list[float]) -> float:
    return 1.0 - cosine_similarity(a, b)


def _recency_weight(now: datetime, viewed_at: datetime, half_life_days: float) -> float:
    days_since = max((now - viewed_at).total_seconds() / 86400, 0.0)
    return float(0.5 ** (days_since / half_life_days))


def build_aversion_vector(
    watched: list[WatchedEmbedding], half_life_days: float, now: datetime
) -> list[float] | None:
    """Recency-weighted centroid of recently watched embeddings — the most recently
    watched title dominates, older ones fade by a half-life, not a hard cutoff (the
    hard cutoff is the watch_history collection's own rolling window, enforced
    pipeline-side). None only when `watched` is empty."""
    if not watched:
        return None
    vectors = np.array([w.vector for w in watched])
    weights = np.array(
        [_recency_weight(now, w.last_viewed_at, half_life_days) for w in watched]
    )
    total_weight = float(weights.sum())
    if total_weight <= 0.0:
        weights = np.ones_like(weights)
        total_weight = float(weights.sum())
    centroid = (vectors * weights[:, None]).sum(axis=0) / total_weight
    return list(centroid.tolist())


def _distance_band(
    scored: list[tuple[CandidateEmbedding, float]],
    low_percentile: float,
    high_percentile: float,
) -> list[tuple[CandidateEmbedding, float]]:
    """Candidates whose distance from the aversion vector falls in
    [low_percentile, high_percentile) of the full candidate pool — deliberately not
    the single furthest candidate(s): the true argmax is as likely to be a
    vector-space outlier (an obscure or low-quality title sitting in a sparse
    region) as a genuinely good contrasting pick. See
    docs/diversity-recommender.md."""
    if not scored:
        return []
    ordered = sorted(scored, key=lambda pair: pair[1])
    n = len(ordered)
    low_idx = int(low_percentile * n)
    high_idx = max(low_idx + 1, int(high_percentile * n))
    return ordered[low_idx:high_idx]


def _softmax_sample(
    banded: list[tuple[CandidateEmbedding, float]],
    pool_size: int,
    temperature: float,
    rng: random.Random,
) -> list[CandidateEmbedding]:
    """Sample `pool_size` candidates from `banded` without replacement, weighted by
    a temperature-scaled softmax over distance — mildly favors the far end of the
    band over the near end, but never deterministic, so repeat requests against the
    same watch history don't always return the same movie. This is the pool MMR
    then diversifies, not the final selection."""
    remaining = list(banded)
    selected: list[CandidateEmbedding] = []
    while remaining and len(selected) < pool_size:
        distances = np.array([d for _, d in remaining])
        scaled = distances / max(temperature, 1e-6)
        scaled = scaled - scaled.max()
        weights = np.exp(scaled).tolist()
        chosen = rng.choices(remaining, weights=weights, k=1)[0]
        selected.append(chosen[0])
        remaining.remove(chosen)
    return selected


def _mmr_select(
    pool: list[CandidateEmbedding],
    aversion_distance: dict[str, float],
    k: int,
    diversity_weight: float,
) -> list[CandidateEmbedding]:
    """Greedy Maximal Marginal Relevance: at each step, pick the candidate
    maximizing `diversity_weight * distance_from_aversion - (1 - diversity_weight) *
    max_similarity_to_already_selected`. Without this, "furthest from what you
    watched" could return several near-identical picks — MMR pushes the results to
    also differ from *each other*, not just from history."""
    remaining = list(pool)
    selected: list[CandidateEmbedding] = []
    while remaining and len(selected) < k:

        def mmr_score(c: CandidateEmbedding) -> float:
            relevance = aversion_distance[c.imdb_id]
            diversity_penalty = (
                max(cosine_similarity(c.vector, s.vector) for s in selected)
                if selected
                else 0.0
            )
            return (
                diversity_weight * relevance
                - (1 - diversity_weight) * diversity_penalty
            )

        best = max(remaining, key=mmr_score)
        selected.append(best)
        remaining.remove(best)
    return selected


class DiversityRecommender:
    """Recommends unwatched movies that are semantically farthest from a
    recency-weighted embedding of recently watched movies — the deliberate
    opposite of `MovieRecommender`'s similarity search. Stateless (like
    `MovieRecommender`): `exclude` is passed in per call rather than tracked here —
    see `DiversityRecommendationService` for the session-level "don't repeat"
    state. See docs/diversity-recommender.md for the full design.

    Candidates beyond `band_high_percentile` (the true distance outliers — likely
    vector-space noise as often as genuinely great contrasting picks, see
    `_distance_band`) aren't discarded outright: each call has an independent
    `outlier_wildcard_probability` chance of adding one extra pick sampled from
    that excluded tail into the pool before MMR narrows to the final `k` — a flat
    per-call coin flip, not a softmax weight blended in with the core band. That
    keeps "how rare" a direct, distance-scale-independent probability rather than
    something that could be overwhelmed by how extreme a given outlier's raw
    distance is (a naive shared-softmax blend would let a sufficiently extreme
    outlier dominate regardless of how small its weight was — the exponential
    softmax term in distance scales with the raw gap, not with any dampening
    prior). A wildcard added to the pool still isn't guaranteed a final slot: MMR
    only picks it if it wins on relevance/diversity against the rest of the pool."""

    def __init__(
        self,
        watch_history: WatchHistoryLookup,
        candidates: CandidatePool,
        *,
        k: int = 5,
        half_life_days: float = 14.0,
        band_low_percentile: float = 0.70,
        band_high_percentile: float = 0.90,
        pool_multiplier: int = 3,
        softmax_temperature: float = 0.15,
        mmr_diversity_weight: float = 0.6,
        min_imdb_rating: float | None = 5.5,
        outlier_wildcard_probability: float = 0.15,
        rng: random.Random | None = None,
    ) -> None:
        self._watch_history = watch_history
        self._candidates = candidates
        self._k = k
        self._half_life_days = half_life_days
        self._band_low_percentile = band_low_percentile
        self._band_high_percentile = band_high_percentile
        self._pool_multiplier = pool_multiplier
        self._softmax_temperature = softmax_temperature
        self._mmr_diversity_weight = mmr_diversity_weight
        self._min_imdb_rating = min_imdb_rating
        self._outlier_wildcard_probability = outlier_wildcard_probability
        self._rng = rng or random.Random()  # noqa: S311 — recommendation variety, not security

    def recommend(self, exclude: frozenset[str] = frozenset()) -> list[str]:
        """Returns up to `k` imdb_ids. Raises `NoWatchHistoryError` if there's no
        recent watch history to build an aversion vector from — the caller decides
        how to surface that (this domain layer does no formatting/rendering)."""
        watched = self._watch_history.recent()
        if not watched:
            raise NoWatchHistoryError(
                "No recent watch history available to build a recommendation from."
            )
        aversion = build_aversion_vector(
            watched, self._half_life_days, datetime.now(UTC).replace(tzinfo=None)
        )
        if aversion is None:
            # Unreachable: build_aversion_vector only returns None for empty
            # `watched`, already ruled out above. Narrows for mypy without `assert`
            # (stripped under -O, and flagged by ruff's S101).
            raise NoWatchHistoryError("No recent watch history available.")

        candidates = [c for c in self._candidates.all() if c.imdb_id not in exclude]
        if self._min_imdb_rating is not None:
            candidates = [
                c
                for c in candidates
                if c.imdb_rating is None or c.imdb_rating >= self._min_imdb_rating
            ]
        if not candidates:
            return []

        scored = [(c, cosine_distance(aversion, c.vector)) for c in candidates]
        core = _distance_band(
            scored, self._band_low_percentile, self._band_high_percentile
        )
        tail = (
            _distance_band(scored, self._band_high_percentile, 1.0)
            if self._band_high_percentile < 1.0
            else []
        )
        # `_distance_band`'s own "at least one" clamp can make its two independent
        # calls share an item when the candidate pool is tiny (well under the
        # library sizes this ships against) -- drop any such overlap so a
        # candidate is never eligible as both a core pick and a wildcard pick.
        core_ids = {c.imdb_id for c, _ in core}
        tail = [pair for pair in tail if pair[0].imdb_id not in core_ids]
        if not core and not tail:
            core = scored

        pool_size = min(len(core), self._k * self._pool_multiplier)
        pool = (
            _softmax_sample(core, pool_size, self._softmax_temperature, self._rng)
            if core
            else []
        )

        if tail and self._rng.random() < self._outlier_wildcard_probability:
            pool.append(
                _softmax_sample(tail, 1, self._softmax_temperature, self._rng)[0]
            )

        aversion_distance = {c.imdb_id: d for c, d in core + tail}
        selected = _mmr_select(
            pool, aversion_distance, self._k, self._mmr_diversity_weight
        )
        return [c.imdb_id for c in selected]
