from app.formatting.runtime import format_runtime


def test_format_runtime_none_when_unresolved() -> None:
    assert format_runtime(None) is None


def test_format_runtime_hours_and_minutes() -> None:
    assert format_runtime(104) == "1h 44m"


def test_format_runtime_under_an_hour() -> None:
    assert format_runtime(45) == "45m"


def test_format_runtime_exact_hour() -> None:
    assert format_runtime(120) == "2h 0m"


def test_format_runtime_zero() -> None:
    assert format_runtime(0) == "0m"
