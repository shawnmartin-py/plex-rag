import colorsys
import io
import logging
import urllib.request
from typing import cast
from urllib.parse import urlsplit

from PIL import Image, UnidentifiedImageError

logger = logging.getLogger(__name__)

_FETCH_TIMEOUT_S = 3.0

# Mirrors the boost the old CSS key light applied to its sampled poster strip
# (filter: saturate(1.8) brightness(2.6)) so the extracted accent matches the
# design the user approved. Values > 1 clamp at full saturation/lightness.
_SATURATION_BOOST = 1.8
_BRIGHTNESS_BOOST = 2.6

# How many colors to sample along the line. The old CSS strip displayed the
# poster's blurred middle band, so its color varied left-to-right with the
# artwork; five gradient stops reproduce that drift without visible banding
# (adjacent stops interpolate linearly, standing in for the blur).
_BAND_SAMPLES = 5

# The vertical slice of the poster the old strip effectively showed
# (background-size: cover on a wide 2px strip = the artwork's middle band).
_BAND_TOP, _BAND_BOTTOM = 0.4, 0.6


def _boost(r8: int, g8: int, b8: int) -> str:
    h, s, v = colorsys.rgb_to_hsv(r8 / 255, g8 / 255, b8 / 255)
    r, g, b = colorsys.hsv_to_rgb(
        h, min(s * _SATURATION_BOOST, 1.0), min(v * _BRIGHTNESS_BOOST, 1.0)
    )
    return f"#{round(r * 255):02X}{round(g * 255):02X}{round(b * 255):02X}"


def boosted_band_colors(image_bytes: bytes) -> tuple[str, ...]:
    """Boosted average colors along the image's horizontal middle band.

    Left-to-right, ``_BAND_SAMPLES`` entries of ``#RRGGBB``. Raises
    ``UnidentifiedImageError``/``OSError`` on undecodable bytes — the caller
    decides how a bad poster degrades.
    """
    with Image.open(io.BytesIO(image_bytes)) as img:
        rgb = img.convert("RGB")
        band = rgb.crop(
            (0, int(rgb.height * _BAND_TOP), rgb.width, int(rgb.height * _BAND_BOTTOM))
        )
        samples = band.resize((_BAND_SAMPLES, 1))
        pixels = [samples.getpixel((x, 0)) for x in range(_BAND_SAMPLES)]
    colors = []
    for pixel in pixels:
        # getpixel is typed as float | tuple | None across modes; "RGB"
        # always yields a 3-tuple.
        colors.append(_boost(*cast(tuple[int, int, int], pixel)))
    return tuple(colors)


class PosterAccents:
    """Poster-derived accent colors for the web UI's card key lights.

    The accent used to be produced in the browser by sampling a blurred CSS
    copy of the poster, but Safari painted that construct unreliably (a 2px
    masked+filtered strip sometimes never repainted after image decode).
    Extracting the colors server-side and shipping plain hexes to CSS is
    deterministic in every browser.

    One poster fetch per distinct URL, then cached for the process lifetime —
    including negative results, so an unreachable poster is not re-fetched on
    every render. Blocking: call via ``run.io_bound`` from the UI event loop.
    """

    def __init__(self) -> None:
        self._cache: dict[str, tuple[str, ...] | None] = {}

    def accent_for(self, thumb_url: str) -> tuple[str, ...] | None:
        if thumb_url in self._cache:
            return self._cache[thumb_url]
        accent = self._extract(thumb_url)
        self._cache[thumb_url] = accent
        return accent

    def _extract(self, thumb_url: str) -> tuple[str, ...] | None:
        # Plex-hosted poster URLs are always http(s); reject anything else
        # before it reaches urlopen so a malformed/malicious payload can't
        # make the server open a file:// or other local/unexpected scheme.
        if urlsplit(thumb_url).scheme not in ("http", "https"):
            logger.warning(
                "Refusing to fetch poster with non-http scheme: %s", thumb_url
            )
            return None
        try:
            with urllib.request.urlopen(thumb_url, timeout=_FETCH_TIMEOUT_S) as resp:  # noqa: S310
                data = resp.read()
            return boosted_band_colors(data)
        except OSError, ValueError, UnidentifiedImageError:
            # No accent is a graceful state (cards render without a key
            # light) — a broken poster URL must never break the chat turn.
            logger.warning("Could not extract poster accent from %s", thumb_url)
            return None
