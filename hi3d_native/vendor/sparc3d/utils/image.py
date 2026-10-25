"""Deterministic in-memory image preparation and background removal."""

from __future__ import annotations

from typing import Any

from PIL import Image, ImageOps

SUBJECT_PADDING = 1.2
"""Framing the conditioner was trained on: the subject fills ``1 / 1.2`` of the square."""


def prepare_image(image: Image.Image) -> Image.Image:
    """Frame an image the way the released conditioner expects to see it.

    Crops to the smallest square containing the opaque subject, composites the
    transparency over white, then pads to a square :data:`SUBJECT_PADDING` times
    larger so the subject sits centered with a consistent margin. Fully opaque
    images -- including any image without an alpha channel -- keep their whole
    extent, since then the subject bounding box is the image itself.

    The framing must match what conditioning saw at training time: the model reads
    apparent object scale off the margin, so an image that is cropped tighter or
    looser than this produces a differently proportioned mesh.

    Args:
        image: Source PIL image, with or without an alpha channel.

    Returns:
        New square RGB image on a white background.

    Raises:
        TypeError: ``image`` is not a PIL image.
        ValueError: ``image`` has a zero-sized dimension.
    """
    if not isinstance(image, Image.Image):
        raise TypeError(f"image must be a PIL image, got {type(image).__name__}")

    # Rotating up front means the alpha bounding box is measured in display
    # orientation rather than in the sensor's.
    oriented = ImageOps.exif_transpose(image)
    if oriented.width <= 0 or oriented.height <= 0:
        raise ValueError(f"image dimensions must be positive, got {oriented.size}")

    rgba = oriented.convert("RGBA")
    subject = _crop_subject_square(rgba)

    white = Image.new("RGB", subject.size, (255, 255, 255))
    white.paste(subject, (0, 0), subject)

    side = int(max(white.size) * SUBJECT_PADDING)
    padded = Image.new("RGB", (side, side), (255, 255, 255))
    offset = ((side - white.width) // 2, (side - white.height) // 2)
    padded.paste(white, offset)
    return padded


def _crop_subject_square(rgba: Image.Image) -> Image.Image:
    """Crop ``rgba`` to the smallest square centered on its opaque region.

    The crop box may extend past the source edges when the subject is close to one;
    PIL fills the overhang with transparent pixels, which the caller turns white.
    """
    bbox = rgba.getchannel("A").getbbox()
    if bbox is None:
        return rgba

    left, upper, right, lower = bbox
    width, height = right - left, lower - upper
    side = max(width, height)
    left -= (side - width) // 2
    upper -= (side - height) // 2
    return rgba.crop((left, upper, left + side, upper + side))


def remove_background(image: Image.Image, session: Any = None) -> Image.Image:
    """Remove an image background lazily with the optional demo dependency.

    Args:
        image: Source PIL image.
        session: Optional reusable ``rembg`` inference session.

    Returns:
        In-memory PIL image returned by ``rembg``.

    Raises:
        TypeError: ``image`` is not a PIL image or ``rembg`` returns another type.
        ImportError: The ``[demo]`` dependency group is not installed.
    """
    if not isinstance(image, Image.Image):
        raise TypeError(f"image must be a PIL image, got {type(image).__name__}")

    try:
        import rembg
    except ImportError as exc:
        raise ImportError(
            "Background removal requires the optional [demo] dependencies; "
            "install them with `pip install 'sparc3d[demo]'`."
        ) from exc

    result = rembg.remove(image, session=session)
    if not isinstance(result, Image.Image):
        raise TypeError(
            "rembg returned an unsupported result; expected an in-memory PIL image, "
            f"got {type(result).__name__}"
        )
    return result
