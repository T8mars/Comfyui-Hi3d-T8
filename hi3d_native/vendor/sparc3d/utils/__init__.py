"""Shared image, batch/offset, precision, and serialization helpers."""

# Deliberately empty: importing one helper here used to pull in every sibling,
# and sparc3d.utils.image drags rembg and PIL behind it. Import from the
# defining module instead, e.g. ``from sparc3d.utils.image import prepare_image``.
