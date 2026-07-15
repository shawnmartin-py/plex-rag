import math
import random
from datetime import datetime, timedelta

import pytest
from hypothesis import given
from hypothesis import strategies as st

from app.domain.diversity import (
    DiversityRecommender,
    NoWatchHistoryError,
    _distance_band,
    _mmr_select,
    _softmax_sample,
    build_aversion_vector,
    cosine_distance,
    cosine_similarity,
)
from app.domain.ports import CandidateEmbedding, WatchedEmbedding

_NOW = datetime(2026, 7, 12)

# Bounded, finite floats -- unbounded magnitude just tests numpy's float64 precision
# rather than the domain invariants, and NaN/inf aren't valid embedding coordinates.
_FLOATS = st.floats(
    min_value=-1e4, max_value=1e4, allow_nan=False, allow_infinity=False
)


@st.composite
def _same_dim_vector_pair(draw: st.DrawFn) -> tuple[list[float], list[float]]:
    """Two vectors of equal (randomly chosen) dimension -- cosine similarity is
    only defined between same-length vectors."""
    dim = draw(st.integers(min_value=1, max_value=8))
    a = draw(st.lists(_FLOATS, min_size=dim, max_size=dim))
    b = draw(st.lists(_FLOATS, min_size=dim, max_size=dim))
    return a, b


@st.composite
def _distinct_candidates(
    draw: st.DrawFn, max_size: int = 15
) -> list[CandidateEmbedding]:
    n = draw(st.integers(min_value=0, max_value=max_size))
    dim = draw(st.integers(min_value=1, max_value=4))
    return [
        _candidate(f"tt{i}", draw(st.lists(_FLOATS, min_size=dim, max_size=dim)))
        for i in range(n)
    ]


@st.composite
def _same_dim_candidates(
    draw: st.DrawFn, dim: int, max_size: int = 15
) -> list[CandidateEmbedding]:
    """Candidates all sharing a fixed vector dimension -- needed wherever they're
    scored against a fixed-dimension aversion vector, since cosine similarity is
    only defined between same-length vectors."""
    n = draw(st.integers(min_value=0, max_value=max_size))
    return [
        _candidate(f"tt{i}", draw(st.lists(_FLOATS, min_size=dim, max_size=dim)))
        for i in range(n)
    ]


@st.composite
def _watched_list(draw: st.DrawFn, max_size: int = 10) -> list[WatchedEmbedding]:
    n = draw(st.integers(min_value=0, max_value=max_size))
    dim = draw(st.integers(min_value=1, max_value=4))
    return [
        _watched(
            f"tt{i}",
            draw(st.lists(_FLOATS, min_size=dim, max_size=dim)),
            days_ago=draw(st.floats(min_value=0.0, max_value=365.0, allow_nan=False)),
        )
        for i in range(n)
    ]


def _watched(tmdb_id: str, vector: list[float], days_ago: float) -> WatchedEmbedding:
    return WatchedEmbedding(
        tmdb_id=tmdb_id, vector=vector, last_viewed_at=_NOW - timedelta(days=days_ago)
    )


def _candidate(
    tmdb_id: str, vector: list[float], imdb_rating: float | None = 7.0
) -> CandidateEmbedding:
    return CandidateEmbedding(tmdb_id=tmdb_id, vector=vector, imdb_rating=imdb_rating)


def _angled_candidate(tmdb_id: str, degrees: float) -> CandidateEmbedding:
    """A 2D unit vector at `degrees` from the watched vector [1, 0] -- gives a
    candidate a precise, predictable cosine distance (1 - cos(degrees)) from an
    aversion vector built from a single watch at [1, 0], without hand-computing
    distances directly."""
    theta = math.radians(degrees)
    return CandidateEmbedding(
        tmdb_id=tmdb_id, vector=[math.cos(theta), math.sin(theta)], imdb_rating=7.0
    )


# --- cosine_similarity / cosine_distance ---


def test_cosine_similarity_identical_vectors_is_one() -> None:
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal_vectors_is_zero() -> None:
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_similarity_opposite_vectors_is_negative_one() -> None:
    assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)


def test_cosine_similarity_zero_vector_is_zero_not_nan() -> None:
    assert cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0


def test_cosine_distance_identical_vectors_is_zero() -> None:
    assert cosine_distance([1.0, 0.0], [1.0, 0.0]) == pytest.approx(0.0)


@given(_same_dim_vector_pair())
def test_cosine_similarity_always_within_unit_range(
    pair: tuple[list[float], list[float]],
) -> None:
    a, b = pair
    # Cauchy-Schwarz guarantees [-1, 1] mathematically; the epsilon covers float64
    # rounding, not a looser invariant.
    assert -1.0 - 1e-6 <= cosine_similarity(a, b) <= 1.0 + 1e-6


# --- build_aversion_vector ---


def test_build_aversion_vector_returns_none_for_empty_list() -> None:
    assert build_aversion_vector([], half_life_days=14.0, now=_NOW) is None


@given(
    watched=_watched_list(),
    half_life_days=st.floats(min_value=0.1, max_value=365.0, allow_nan=False),
)
def test_build_aversion_vector_none_iff_watched_empty(
    watched: list[WatchedEmbedding], half_life_days: float
) -> None:
    result = build_aversion_vector(watched, half_life_days=half_life_days, now=_NOW)
    assert (result is None) == (len(watched) == 0)


def test_build_aversion_vector_single_item_returns_its_vector() -> None:
    result = build_aversion_vector(
        [_watched("tt1", [1.0, 2.0], days_ago=0)], half_life_days=14.0, now=_NOW
    )
    assert result is not None
    assert result == pytest.approx([1.0, 2.0])


def test_build_aversion_vector_weights_recent_watch_more_heavily() -> None:
    # A watch from today and one from 60 days ago (well past the half-life) --
    # the centroid should sit much closer to the recent vector.
    result = build_aversion_vector(
        [
            _watched("recent", [10.0, 0.0], days_ago=0),
            _watched("old", [0.0, 10.0], days_ago=60),
        ],
        half_life_days=14.0,
        now=_NOW,
    )
    assert result is not None
    assert result[0] > result[1]


# --- _distance_band ---


def test_distance_band_selects_percentile_slice() -> None:
    scored = [(_candidate(f"tt{i}", [float(i)]), float(i)) for i in range(10)]
    banded = _distance_band(scored, low_percentile=0.7, high_percentile=0.9)
    assert [c.tmdb_id for c, _ in banded] == ["tt7", "tt8"]


def test_distance_band_empty_input_returns_empty() -> None:
    assert _distance_band([], 0.7, 0.9) == []


def test_distance_band_always_returns_at_least_one_for_nonempty_input() -> None:
    scored = [(_candidate("tt1", [1.0]), 0.5)]
    banded = _distance_band(scored, low_percentile=0.99, high_percentile=1.0)
    assert len(banded) == 1


@given(
    n=st.integers(min_value=1, max_value=30),
    low_percentile=st.floats(min_value=0.0, max_value=0.99, allow_nan=False),
    data=st.data(),
)
def test_distance_band_nonempty_for_any_nonempty_input(
    n: int, low_percentile: float, data: st.DataObject
) -> None:
    high_percentile = data.draw(
        st.floats(min_value=low_percentile, max_value=1.0, allow_nan=False)
    )
    scored = [(_candidate(f"tt{i}", [float(i)]), float(i)) for i in range(n)]
    banded = _distance_band(scored, low_percentile, high_percentile)
    assert len(banded) >= 1


# --- _softmax_sample ---


def test_softmax_sample_returns_requested_pool_size() -> None:
    banded = [(_candidate(f"tt{i}", [float(i)]), float(i)) for i in range(10)]
    rng = random.Random(42)
    pool = _softmax_sample(banded, pool_size=4, temperature=0.5, rng=rng)
    assert len(pool) == 4


def test_softmax_sample_never_repeats_a_candidate() -> None:
    banded = [(_candidate(f"tt{i}", [float(i)]), float(i)) for i in range(5)]
    rng = random.Random(1)
    pool = _softmax_sample(banded, pool_size=5, temperature=0.5, rng=rng)
    assert len({c.tmdb_id for c in pool}) == 5


def test_softmax_sample_caps_at_available_candidates() -> None:
    banded = [(_candidate("tt1", [1.0]), 1.0)]
    rng = random.Random(1)
    pool = _softmax_sample(banded, pool_size=5, temperature=0.5, rng=rng)
    assert len(pool) == 1


# --- _mmr_select ---


def test_mmr_select_respects_k() -> None:
    pool = [_candidate(f"tt{i}", [float(i), 0.0]) for i in range(5)]
    distances = {c.tmdb_id: 1.0 for c in pool}
    selected = _mmr_select(pool, distances, k=3, diversity_weight=0.5)
    assert len(selected) == 3


def test_mmr_select_prefers_diverse_candidates_over_near_duplicates() -> None:
    # Two near-identical vectors (a, a2) and one distinct (b), all equally "relevant"
    # -- MMR should pick a (or a2) then b, not both near-duplicates, once diversity
    # is weighted heavily.
    a = _candidate("a", [1.0, 0.0])
    a2 = _candidate("a2", [0.99, 0.01])
    b = _candidate("b", [0.0, 1.0])
    distances = {"a": 1.0, "a2": 1.0, "b": 1.0}
    selected = _mmr_select([a, a2, b], distances, k=2, diversity_weight=0.1)
    ids = {c.tmdb_id for c in selected}
    assert "b" in ids
    assert not {"a", "a2"} <= ids


@given(
    candidates=_distinct_candidates(),
    k=st.integers(min_value=0, max_value=20),
    diversity_weight=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    data=st.data(),
)
def test_mmr_select_never_exceeds_k_or_duplicates_a_candidate(
    candidates: list[CandidateEmbedding],
    k: int,
    diversity_weight: float,
    data: st.DataObject,
) -> None:
    aversion_distance = {c.tmdb_id: data.draw(_FLOATS) for c in candidates}
    selected = _mmr_select(candidates, aversion_distance, k, diversity_weight)
    assert len(selected) <= min(k, len(candidates))
    ids = [c.tmdb_id for c in selected]
    assert len(ids) == len(set(ids))


# --- DiversityRecommender ---


class _FakeWatchHistory:
    def __init__(self, items: list[WatchedEmbedding]) -> None:
        self._items = items

    def recent(self) -> list[WatchedEmbedding]:
        return self._items


class _FakeCandidatePool:
    def __init__(self, items: list[CandidateEmbedding]) -> None:
        self._items = items

    def all(self) -> list[CandidateEmbedding]:
        return self._items


def test_recommend_raises_when_no_watch_history() -> None:
    recommender = DiversityRecommender(
        _FakeWatchHistory([]), _FakeCandidatePool([_candidate("tt1", [1.0])])
    )
    with pytest.raises(NoWatchHistoryError):
        recommender.recommend()


def test_recommend_excludes_given_tmdb_ids() -> None:
    watched = [_watched("watched1", [1.0, 0.0], days_ago=1)]
    candidates = [
        _candidate("tt1", [0.0, 1.0]),
        _candidate("tt2", [0.0, -1.0]),
    ]
    recommender = DiversityRecommender(
        _FakeWatchHistory(watched),
        _FakeCandidatePool(candidates),
        k=2,
        band_low_percentile=0.0,
        band_high_percentile=1.0,
        rng=random.Random(0),
    )
    result = recommender.recommend(exclude=frozenset({"tt1"}))
    assert "tt1" not in result


def test_recommend_excludes_candidates_below_quality_floor() -> None:
    watched = [_watched("watched1", [1.0, 0.0], days_ago=1)]
    candidates = [
        _candidate("low", [0.0, 1.0], imdb_rating=2.0),
        _candidate("high", [0.0, -1.0], imdb_rating=8.0),
    ]
    recommender = DiversityRecommender(
        _FakeWatchHistory(watched),
        _FakeCandidatePool(candidates),
        k=2,
        min_imdb_rating=5.5,
        band_low_percentile=0.0,
        band_high_percentile=1.0,
        rng=random.Random(0),
    )
    result = recommender.recommend()
    assert "low" not in result
    assert "high" in result


def test_recommend_returns_empty_list_when_all_candidates_filtered_out() -> None:
    watched = [_watched("watched1", [1.0, 0.0], days_ago=1)]
    candidates = [_candidate("only", [0.0, 1.0], imdb_rating=1.0)]
    recommender = DiversityRecommender(
        _FakeWatchHistory(watched),
        _FakeCandidatePool(candidates),
        min_imdb_rating=5.5,
    )
    assert recommender.recommend() == []


def test_recommend_returns_at_most_k_results() -> None:
    watched = [_watched("watched1", [1.0, 0.0], days_ago=1)]
    candidates = [_candidate(f"tt{i}", [0.0, float(i)]) for i in range(20)]
    recommender = DiversityRecommender(
        _FakeWatchHistory(watched),
        _FakeCandidatePool(candidates),
        k=3,
        rng=random.Random(0),
    )
    result = recommender.recommend()
    assert len(result) == 3


# --- outlier wildcard (band_high_percentile+ tail, previously discarded outright) ---

# 90 candidates clustered close (0-89 degrees from the watched vector, cosine
# distance 0 to ~0.98) rank as the "core" 70th-90th percentile band; 10 candidates
# near-opposite (170-179 degrees, distance ~1.98-2.0) are unambiguously the tail
# beyond band_high_percentile, however the percentile math falls for this n.
_CORE_CANDIDATES = [_angled_candidate(f"core{i}", float(i)) for i in range(90)]
_TAIL_CANDIDATES = [_angled_candidate(f"outlier{i}", 170.0 + i) for i in range(10)]
_ALL_CANDIDATES = _CORE_CANDIDATES + _TAIL_CANDIDATES


def _recommend_with_seed(outlier_wildcard_probability: float, seed: int) -> list[str]:
    recommender = DiversityRecommender(
        _FakeWatchHistory([_watched("watched1", [1.0, 0.0], days_ago=1)]),
        _FakeCandidatePool(_ALL_CANDIDATES),
        k=5,
        outlier_wildcard_probability=outlier_wildcard_probability,
        rng=random.Random(seed),
    )
    return recommender.recommend()


def test_recommend_never_surfaces_outlier_tail_when_probability_is_zero() -> None:
    for seed in range(20):
        result = _recommend_with_seed(outlier_wildcard_probability=0.0, seed=seed)
        assert not any(tmdb_id.startswith("outlier") for tmdb_id in result)


def test_recommend_can_surface_outlier_tail_when_probability_is_one() -> None:
    # probability=1.0 guarantees a wildcard is *added to the pool* on every call,
    # but MMR still has to pick it into the final k -- across enough seeds it
    # should win at least once, since it's also the single most "relevant" (most
    # distant) candidate available.
    surfaced = any(
        any(
            tmdb_id.startswith("outlier")
            for tmdb_id in _recommend_with_seed(
                outlier_wildcard_probability=1.0, seed=seed
            )
        )
        for seed in range(20)
    )
    assert surfaced


def test_recommend_surfaces_outlier_tail_less_often_at_low_probability() -> None:
    def _hit_rate(probability: float) -> int:
        return sum(
            1
            for seed in range(60)
            if any(
                tmdb_id.startswith("outlier")
                for tmdb_id in _recommend_with_seed(probability, seed)
            )
        )

    rare = _hit_rate(0.15)  # the shipped default
    common = _hit_rate(0.9)
    assert rare < common


# --- core/tail dedup on tiny candidate pools (recommend()'s own overlap guard) ---


@given(
    candidates=_same_dim_candidates(dim=2, max_size=8),
    seed=st.integers(min_value=0, max_value=10_000),
)
def test_recommend_never_returns_duplicate_ids_for_tiny_pools(
    candidates: list[CandidateEmbedding], seed: int
) -> None:
    # Tiny pools are exactly where _distance_band's own "at least one" clamp can
    # make the core and tail bands share a candidate (see the dedup comment in
    # DiversityRecommender.recommend) -- without that dedup, the shared candidate
    # could enter the MMR pool twice (once via the core softmax sample, once as
    # the wildcard) and come out selected twice. outlier_wildcard_probability=1.0
    # maximizes the chance of hitting that path on every draw.
    watched = [_watched("watched1", [1.0, 0.0], days_ago=1)]
    recommender = DiversityRecommender(
        _FakeWatchHistory(watched),
        _FakeCandidatePool(candidates),
        k=len(candidates) or 1,
        outlier_wildcard_probability=1.0,
        rng=random.Random(seed),
    )
    result = recommender.recommend()
    assert len(result) == len(set(result))
