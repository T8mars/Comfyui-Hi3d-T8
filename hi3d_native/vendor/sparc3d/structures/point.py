"""Sparse point-cloud and mesh containers passed between PTv3 modules."""

# Adapted from Pointcept / Point Transformer V3 under the MIT License.
# Modified for SPARC3D: a local ``AttrDict`` replaces ``addict``, the octree
# branch is dropped, and the layouts describe the mesh autoencoder's fields.

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import Tensor

from sparc3d.ops import sparse_backend
from sparc3d.utils.misc import batch2offset, deep_clone, offset2batch
from sparc3d.utils.serialization import encode


class AttrDict(dict):
    """Dict whose keys are also readable and writable as attributes."""

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key) from None

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def __delattr__(self, key: str) -> None:
        try:
            del self[key]
        except KeyError:
            raise AttributeError(key) from None


class Point(AttrDict):
    """
    Sparse voxel point cloud with a packed feature tensor and a declared layout.

    A ``Point`` is a dict of tensors. Per-attribute inputs (``sdf``, ``deform``) are
    concatenated into a single ``feat`` tensor during :meth:`initialization`, and
    ``layouts`` records the channel range each attribute occupies so it can be
    recovered later. Passing an attribute as ``None`` declares
    its channels without supplying values, which is how a decoder output is described.

    Derived representations are built on demand and tracked by flags:
    :meth:`serialization` adds space-filling-curve orders, :meth:`sparsify` adds the
    backend sparse tensor the sparse convolutions consume.
    """

    ALL_LAYOUTS: Dict[str, Any] = {
        "sdf": {
            "shape": (8, 1),
            "size": 8,
            "denorm": lambda x: x,
            "act": lambda x: torch.tanh(x),
        },
        "deform": {
            "shape": (8, 3),
            "size": 8 * 3,
            "denorm": lambda x: x * 0.5,
            "act": lambda x: torch.tanh(x),
        },
    }
    """Per-corner attributes in packing order, with their denormalization and activation."""

    def __init__(self, data_dict: Optional[dict] = None) -> None:
        """
        Args:
            data_dict: Initial contents. Nested dicts are wrapped so their entries are
                reachable as attributes too.
        """
        super().__init__()
        if data_dict is not None:
            for key, value in data_dict.items():
                self[key] = AttrDict(value) if isinstance(value, dict) else value

    ## Public interface

    def initialization(self) -> None:
        """Build ``layouts``, ``feat``, ``matrix``, and the batch/offset pair.

        Idempotent with respect to already-present keys: an existing ``feat`` is kept
        as-is, so a decoder output can be re-wrapped without re-normalizing.
        """
        if isinstance(self.resolution, torch.Tensor):
            self.resolution = int(self.resolution[0])

        self["layouts"] = self._build_layouts()
        self["feat"] = self._build_feat()
        self["matrix"] = self._build_matrix()

        if "batch" not in self and "offset" in self:
            self["batch"] = offset2batch(self["offset"])
        elif "offset" not in self and "batch" in self:
            self["offset"] = batch2offset(self["batch"])

        self.is_serialized = False
        self.is_sparsified = False

    def serialization(
        self,
        order: str | List[str] = "z",
        depth: Optional[int] = None,
    ) -> None:
        """Add space-filling-curve orders over ``grid_coord``.

        Args:
            order: Curve name or list of names; one order is produced per name.
            depth: Bits per axis; inferred from ``grid_coord`` when ``None``.

        Raises:
            KeyError: ``batch`` or ``grid_coord`` is missing.
            ValueError: ``depth`` exceeds 16 bits, or the batched code would not fit
                in an int64.
        """
        if self.is_serialized:
            return

        if "batch" not in self:
            raise KeyError("Missing 'batch' key in point cloud data.")

        self._require_grid_coord("serialization")

        if depth is None:
            depth = int(self["grid_coord"].max().item()).bit_length()

        if depth * 3 + len(self["offset"]).bit_length() > 63:
            raise ValueError("Serialization depth too high for int64 representation.")
        if depth > 16:
            raise ValueError("Serialization depth exceeds maximum allowed value of 16.")

        if isinstance(order, str):
            order = [order]
        code_list = [
            encode(self["grid_coord"], self["batch"], depth, order=order_)
            for order_ in order
        ]
        code_tensor = torch.stack(code_list)
        order_tensor = torch.argsort(code_tensor)
        inverse = torch.zeros_like(order_tensor).scatter_(
            dim=1,
            index=order_tensor,
            src=torch.arange(
                0, code_tensor.shape[1], device=order_tensor.device
            ).repeat(code_tensor.shape[0], 1),
        )

        self["serialized_order"] = order_tensor
        self["serialized_inverse"] = inverse

        self.is_serialized = True

    def sparsify(self) -> None:
        """Attach the sparse tensor the sparse convolutions operate on.

        Raises:
            ImportError: ``spconv`` is not installed.
            KeyError: ``feat``, ``batch``, or ``grid_coord`` is missing.
        """
        if self.is_sparsified:
            return

        if not {"feat", "batch"}.issubset(self.keys()):
            raise KeyError("Missing 'feat' or 'batch' key required for sparsification.")

        self._require_grid_coord("sparsification")

        # Declaring the grid extent turns "grid_coord outside resolution" into an
        # error instead of a silently wrong hash on the spconv backend.
        resolution = self.get("resolution")
        self["sparse_conv_feat"] = sparse_backend.sparse_tensor(
            self["feat"],
            torch.cat(
                [self["batch"].unsqueeze(-1).int(), self["grid_coord"].int()], dim=1
            ).contiguous(),
            spatial_shape=None if resolution is None else (int(resolution),) * 3,
        )
        self.is_sparsified = True

    def update_feat(self, feat: Tensor) -> "Point":
        """Replace ``feat``, keeping any sparse tensor view in sync."""
        self["feat"] = feat
        if self.is_sparsified:
            self["sparse_conv_feat"] = sparse_backend.replace_feats(
                self["sparse_conv_feat"], feat
            )
        return self

    def float(self) -> "Point":
        """Cast ``feat`` to float32."""
        return self.update_feat(self["feat"].float())

    def to(self, dtype: torch.dtype) -> "Point":
        """Cast ``feat`` to ``dtype`` if it is not already."""
        if self.dtype != dtype:
            self.update_feat(self["feat"].to(dtype))
        return self

    def get_layout(self) -> Dict[str, Any]:
        """Return the attribute-to-channel-range layout of ``feat``."""
        return self["layouts"]

    def get_vert_feat(self, feat: Tensor) -> Tuple[Optional[Tensor], Dict[str, Any]]:
        """Split ``feat`` into denormalized per-corner vertex attributes.

        Args:
            feat: Packed features ``[N, C]``.

        Returns:
            ``(vert_feat, layouts)`` where ``vert_feat`` is ``[N, 8, C']`` with the
            present attributes concatenated on the last axis, and ``layouts`` maps
            each of ``sdf`` and ``deform`` to its range in ``vert_feat`` or to
            ``None`` when absent.
        """
        return self._get_attrs(feat, ["sdf", "deform"])

    def activate(self) -> None:
        """Apply each attribute's activation to its channels of ``feat``, in place."""
        layout = self.get_layout()
        activated_feat = []
        for key, entry in layout.items():
            feat = self.get_attr_from_feat(self["feat"], layout, key)
            activated_feat.append(entry["act"](feat))
        self.update_feat(torch.cat(activated_feat, dim=-1))

    @staticmethod
    def get_attr_from_feat(
        feat: Tensor, layouts: Dict[str, Any], key: str
    ) -> Optional[Tensor]:
        """Slice one attribute out of ``feat``; ``None`` when ``key`` is not laid out."""
        if key not in layouts or layouts[key] is None:
            return None
        return feat[..., layouts[key]["range"][0]:layouts[key]["range"][1]]

    @staticmethod
    def grid_to_coord(grid: Tensor, matrix: Tensor) -> Tensor:
        """Map grid coordinates into the normalized coordinate frame.

        Args:
            grid: Grid coordinates ``[N, 3]``.
            matrix: Homogeneous grid-to-coord transform ``[4, 4]``.

        Returns:
            Transformed coordinates ``[N, 3]``.

        Raises:
            ValueError: ``grid`` is not 3-column or ``matrix`` is not ``[4, 4]``.
        """
        if grid.shape[-1] != 3 or matrix.shape != (4, 4):
            raise ValueError(
                "Invalid shape: 'grid' must have 3 columns and 'matrix' must be 4x4."
            )
        grid = grid.float()
        ones = grid.new_ones((grid.shape[0], 1))
        stack = torch.cat([grid, ones], dim=1)
        with torch.amp.autocast(device_type="cuda", enabled=False):
            transformed = torch.matmul(matrix, stack.T).T
        return transformed[:, :3]

    def clone(self, **kwargs: Any) -> "Point":
        """Return an initialized copy, with ``kwargs`` replacing same-named entries.

        Tensors are shared rather than copied (see :func:`~sparc3d.utils.misc.deep_clone`)
        and the sparse-convolution view is dropped so the clone can re-derive it.
        """
        point = Point()
        for key, value in self.items():
            if key in kwargs:
                continue
            if key == "sparse_conv_feat":
                continue
            point[key] = deep_clone(value)
        point.update(kwargs)
        point.initialization()
        return point

    ## Internal helpers

    def _require_grid_coord(self, operation: str) -> None:
        """Raise unless ``grid_coord`` is present.

        Raises:
            KeyError: ``grid_coord`` is absent.
        """
        if "grid_coord" not in self:
            raise KeyError(f"Missing 'grid_coord' key for {operation}.")

    def _build_layouts(self) -> Dict[str, Any]:
        """Assign each present attribute a contiguous channel range in ``feat``."""
        if "layouts" in self:
            return self["layouts"]
        layouts: Dict[str, Any] = {}
        start = 0
        for key, entry in self.ALL_LAYOUTS.items():
            if key in self:
                # Copy so the per-instance range never mutates the class-level table.
                layouts[key] = {**entry, "range": (start, start + entry["size"])}
                start += entry["size"]
        return layouts

    def _build_feat(self) -> Tensor:
        """Concatenate the laid-out attributes into a single tensor.

        Raises:
            KeyError: A laid-out attribute is missing, or no attribute was supplied.
            ValueError: An attribute cannot be reshaped to its layout size.
        """
        if "feat" in self:
            return self["feat"]

        feats: List[Tensor] = []
        layouts = self.get("layouts", self._build_layouts())
        if not layouts:
            raise KeyError(
                "a Point needs either 'feat' or at least one of "
                f"{sorted(self.ALL_LAYOUTS)} to build features from"
            )
        for key, entry in layouts.items():
            if key not in self:
                raise KeyError(f"Missing attribute '{key}' for feature construction.")
            try:
                attr_tensor = self[key].reshape(-1, entry["size"])
            except RuntimeError as error:
                raise ValueError(
                    f"Error reshaping attribute '{key}': {error}"
                ) from error
            feats.append(attr_tensor)
            # The raw attribute is now redundant with its slice of feat.
            del self[key]
        return torch.cat(feats, dim=-1)

    def _build_matrix(self) -> Tensor:
        """Build the ``[4, 4]`` transform placing ``grid_coord`` in the unit ball.

        Raises:
            KeyError: ``grid_coord`` is missing.
        """
        if "matrix" in self:
            return self["matrix"]
        if "grid_coord" not in self:
            raise KeyError("Missing 'grid_coord' key for building transformation matrix.")
        grid_coord = self["grid_coord"].float()
        center = grid_coord.mean(dim=0)
        centered_grid = grid_coord - center
        scale = torch.norm(centered_grid, dim=1).max() + 1e-8
        matrix = torch.eye(4, device=centered_grid.device)
        matrix[:3, :3] /= scale
        matrix[:3, 3] = -center / scale
        return matrix

    def _get_attr(
        self, feat: Tensor, key: str
    ) -> Tuple[Optional[Tensor], Optional[Dict[str, Any]]]:
        """Slice, reshape to ``[N, *shape]``, and denormalize one attribute.

        Raises:
            ValueError: The attribute's channels do not fit its declared shape.
        """
        layouts = self.layouts
        if key not in layouts:
            return None, None
        attr = feat[:, layouts[key]["range"][0]:layouts[key]["range"][1]]
        try:
            attr = attr.reshape(-1, *layouts[key]["shape"])
        except RuntimeError as error:
            raise ValueError(f"Error reshaping attribute '{key}': {error}") from error
        attr = layouts[key]["denorm"](attr)
        return attr, deepcopy(layouts[key])

    def _get_attrs(
        self, feat: Tensor, keys: List[str]
    ) -> Tuple[Optional[Tensor], Dict[str, Any]]:
        """Concatenate several denormalized attributes and re-range their layouts."""
        v_feat: List[Tensor] = []
        layouts: Dict[str, Any] = {}
        start = 0
        for key in keys:
            attr, layout = self._get_attr(feat, key)
            if attr is None:
                layouts[key] = None
                continue
            v_feat.append(attr)
            layout["range"] = (start, start + attr.shape[-1])
            layouts[key] = layout
            start += attr.shape[-1]
        cat_feat = torch.cat(v_feat, dim=-1) if v_feat else None
        return cat_feat, layouts

    ## Properties

    @property
    def dtype(self) -> torch.dtype:
        """Dtype of ``feat``."""
        return self["feat"].dtype


class Mesh(AttrDict):
    """Triangle mesh held as torch tensors."""

    def __init__(self, vertices: Tensor, faces: Tensor) -> None:
        """
        Args:
            vertices: Vertex positions ``[N, 3]``.
            faces: Triangle indices ``[F, 3]``.
        """
        super().__init__()
        self["vertices"] = vertices
        self["faces"] = faces
        self["success"] = vertices.shape[0] != 0 and faces.shape[0] != 0
