from unittest.mock import MagicMock, patch

from app.synopsis import _fetch_wikipedia, _titles_match

AVENGEMENT_PLOT = (
    "Cain Burgess, a prisoner being escorted to a hospital to learn of his mother's death, "
    "escapes after seeing her corpse."
)

AVENGEMENT_EXTRACT = f"== Plot ==\n{AVENGEMENT_PLOT}\n== Cast ==\nScott Adkins"

ENDGAME_EXTRACT = "== Plot ==\nIn 2018, 23 days after Thanos...\n== Cast ==\nRobert Downey Jr."


def _search_response(*titles: str) -> MagicMock:
    resp = MagicMock()
    resp.json.return_value = {"query": {"search": [{"title": t} for t in titles]}}
    return resp


def _extract_response(extract: str, page_id: str = "123") -> MagicMock:
    resp = MagicMock()
    resp.json.return_value = {"query": {"pages": {page_id: {"extract": extract}}}}
    return resp


# --- _titles_match ---


def test_titles_match_exact():
    assert _titles_match("Avengement", "Avengement") is True


def test_titles_match_wiki_has_film_disambiguation():
    assert _titles_match("Avengement", "Avengement (film)") is True


def test_titles_match_wiki_has_year_disambiguation():
    assert _titles_match("Avengement", "Avengement (2019 film)") is True


def test_titles_match_case_insensitive():
    assert _titles_match("avengement", "AVENGEMENT") is True


def test_titles_match_rejects_similar_but_different_film():
    assert _titles_match("Avengement", "Avengers: Endgame") is False


def test_titles_match_rejects_unrelated_film():
    assert _titles_match("Avengement", "The Dark Knight") is False


def test_titles_match_movie_title_contained_in_wiki_title():
    assert _titles_match("The Dark Knight", "The Dark Knight Rises") is True


def test_titles_match_wiki_title_contained_in_movie_title():
    assert _titles_match("The Dark Knight Rises", "The Dark Knight") is True


def test_titles_match_ignores_punctuation():
    assert _titles_match("Se7en", "Se7en") is True


# --- _fetch_wikipedia: title matching ---


@patch("app.synopsis.requests.get")
def test_fetch_wikipedia_skips_wrong_first_result_and_uses_correct_second(mock_get):
    # First search result is Avengers: Endgame (wrong), second is Avengement (correct)
    mock_get.side_effect = [
        _search_response("Avengers: Endgame", "Avengement"),
        _extract_response(AVENGEMENT_EXTRACT),
    ]
    result = _fetch_wikipedia("Avengement", 2019)
    assert result is not None
    assert "Cain Burgess" in result


@patch("app.synopsis.requests.get")
def test_fetch_wikipedia_uses_matching_first_result_directly(mock_get):
    mock_get.side_effect = [
        _search_response("Avengement", "Avengers: Endgame"),
        _extract_response(AVENGEMENT_EXTRACT),
    ]
    result = _fetch_wikipedia("Avengement", 2019)
    assert result is not None
    assert "Cain Burgess" in result


@patch("app.synopsis.requests.get")
def test_fetch_wikipedia_returns_none_when_no_result_matches(mock_get):
    mock_get.side_effect = [
        _search_response("Avengers: Endgame", "Avengers: Infinity War", "Avengers: Age of Ultron"),
    ]
    result = _fetch_wikipedia("Avengement", 2019)
    assert result is None


@patch("app.synopsis.requests.get")
def test_fetch_wikipedia_does_not_fetch_extract_when_no_title_matches(mock_get):
    mock_get.side_effect = [
        _search_response("Avengers: Endgame", "Avengers: Infinity War"),
    ]
    _fetch_wikipedia("Avengement", 2019)
    assert mock_get.call_count == 1  # only the search call, no extract call


@patch("app.synopsis.requests.get")
def test_fetch_wikipedia_fetches_extract_for_matched_title(mock_get):
    mock_get.side_effect = [
        _search_response("Avengers: Endgame", "Avengement"),
        _extract_response(AVENGEMENT_EXTRACT),
    ]
    _fetch_wikipedia("Avengement", 2019)
    extract_call_params = mock_get.call_args_list[1][1]["params"]
    assert extract_call_params["titles"] == "Avengement"


# --- _fetch_wikipedia: plot extraction ---


@patch("app.synopsis.requests.get")
def test_fetch_wikipedia_extracts_plot_section(mock_get):
    mock_get.side_effect = [
        _search_response("Avengement"),
        _extract_response(AVENGEMENT_EXTRACT),
    ]
    result = _fetch_wikipedia("Avengement", 2019)
    assert result == AVENGEMENT_PLOT.strip()


@patch("app.synopsis.requests.get")
def test_fetch_wikipedia_returns_none_when_no_plot_section(mock_get):
    extract = "Avengement is a 2019 British action film.\n== Cast ==\nScott Adkins"
    mock_get.side_effect = [
        _search_response("Avengement"),
        _extract_response(extract),
    ]
    result = _fetch_wikipedia("Avengement", 2019)
    assert result is None


@patch("app.synopsis.requests.get")
def test_fetch_wikipedia_returns_none_when_search_empty(mock_get):
    resp = MagicMock()
    resp.json.return_value = {"query": {"search": []}}
    mock_get.return_value = resp
    result = _fetch_wikipedia("Avengement", 2019)
    assert result is None
