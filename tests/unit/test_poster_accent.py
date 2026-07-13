import io

import pytest
from PIL import Image

from app.adapters.poster_accent import PosterAccents, boosted_band_colors


def _jpeg(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def _solid(color: tuple[int, int, int]) -> bytes:
    return _jpeg(Image.new("RGB", (40, 60), color))


def test_band_colors_brighten_dark_posters() -> None:
    # A dark desaturated poster must still yield clearly visible accents —
    # the whole point of the boost (mirrors the old CSS brightness(2.6)).
    colors = boosted_band_colors(_solid((60, 50, 40)))
    for accent in colors:
        r, g, b = (int(accent[i : i + 2], 16) for i in (1, 3, 5))
        assert max(r, g, b) > 120
        assert r > b  # warm input stays warm


def test_band_colors_follow_the_artwork_left_to_right() -> None:
    # The old sampled strip varied along the line with the poster's middle
    # band; the extraction must preserve that, not collapse to one color.
    img = Image.new("RGB", (40, 60))
    for x in range(40):
        for y in range(60):
            img.putpixel((x, y), (200, 30, 30) if x < 20 else (30, 30, 200))
    colors = boosted_band_colors(_jpeg(img))
    first_r, first_b = int(colors[0][1:3], 16), int(colors[0][5:7], 16)
    last_r, last_b = int(colors[-1][1:3], 16), int(colors[-1][5:7], 16)
    assert first_r > first_b  # left samples red
    assert last_b > last_r  # right samples blue


def test_band_colors_clamp_bright_input() -> None:
    assert set(boosted_band_colors(_solid((240, 240, 240)))) == {"#FFFFFF"}


def test_band_colors_reject_non_image_bytes() -> None:
    with pytest.raises(OSError):
        boosted_band_colors(b"not an image")


def test_poster_accents_caches_failures_without_refetching(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    accents = PosterAccents()
    calls = {"n": 0}

    def failing_extract(url: str) -> tuple[str, ...] | None:
        calls["n"] += 1
        return None

    monkeypatch.setattr(accents, "_extract", failing_extract)
    assert accents.accent_for("http://example.invalid/p.jpg") is None
    assert accents.accent_for("http://example.invalid/p.jpg") is None
    assert calls["n"] == 1
