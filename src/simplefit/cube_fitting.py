"""Cube-level wrappers around the CARTA-like one-dimensional spectrum fitter."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from astropy.table import Table
from astropy.wcs.utils import pixel_to_skycoord
from joblib import Parallel, delayed

from .spectrum_fitting import detect_initial_components, fit_spectrum, fit_spectrum_from_components


@dataclass
class FitCubeResult:
    """Container returned by :func:`fit_cube`."""

    component_table: Table
    n_components: np.ndarray
    success: np.ndarray
    noise: np.ndarray
    message: np.ndarray
    model_cube: np.ndarray | None = None
    residual_cube: np.ndarray | None = None
    ssa_labels: np.ndarray | None = None
    ssa_table: Table | None = None

    def to_table(self) -> Table:
        """Return the sparse component table."""

        return self.component_table

    def write_table(self, path: str | Path, format: str = "csv", overwrite: bool = True) -> None:
        """Write the sparse component table using Astropy Table writers."""

        self.component_table.write(path, format=format, overwrite=overwrite)

    def component_map(self, parameter: str, component: int = 1) -> np.ndarray:
        """Return a 2D map for one component number and one table parameter.

        Missing pixels or pixels without the requested component are filled with
        NaN. Components are one-indexed to match the table.
        """

        if component < 1:
            raise ValueError("component must be one-indexed and >= 1.")
        if parameter not in self.component_table.colnames:
            raise ValueError(f"Unknown parameter {parameter!r}. Available columns: {self.component_table.colnames}")

        output = np.full(self.n_components.shape, np.nan, dtype=float)
        if len(self.component_table) == 0:
            return output

        rows = self.component_table[self.component_table["component"] == component]
        for row in rows:
            output[int(row["y"]), int(row["x"])] = row[parameter]
        return output


def fit_cube(
    cube: Any,
    *,
    n_jobs: int = 1,
    spectral_axis: np.ndarray | None = None,
    spectral_unit: str | None = None,
    mask: np.ndarray | None = None,
    max_components: int | None = None,
    fit_baseline: bool = True,
    store_models: bool = False,
    progress: bool = True,
    ssa_size: int | tuple[int, int] | None = None,
    ssa_min_pixels: int = 1,
) -> FitCubeResult:
    """Fit every selected spatial pixel in a SpectralCube with ``fit_spectrum``.

    Parameters
    ----------
    cube
        A ``spectral_cube.SpectralCube``-like object with shape
        ``(spectral, y, x)`` and a ``spectral_axis`` attribute.
    n_jobs
        Number of parallel workers. Use ``1`` for serial execution.
    spectral_axis
        Optional spectral axis values. If omitted, ``cube.spectral_axis`` is
        used.
    spectral_unit
        Optional unit to convert the cube spectral axis into before fitting.
    mask
        Optional 2D boolean spatial mask. True pixels are fit.
    max_components
        Optional cap passed to ``fit_spectrum``.
    fit_baseline
        Whether each spectrum fit should include the CARTA-like baseline
        heuristic.
    store_models
        If true, include full model and residual cubes in the result.
    progress
        If true, show a progress bar while fitting selected pixels. Uses
        ``tqdm`` when available and falls back to a lightweight stderr bar.
    ssa_size
        Optional mask-aware spatial averaging box size. If set, auto-detection
        is run once on each averaged SSA spectrum, and member pixels are fit
        from those SSA components.
    ssa_min_pixels
        Minimum number of masked pixels required to keep an SSA box.
    """

    data = _cube_data_values(cube)
    if data.ndim != 3:
        raise ValueError(f"Expected cube data with shape (spectral, y, x), got {data.shape}.")

    axis = _spectral_axis_values(cube, spectral_axis=spectral_axis, spectral_unit=spectral_unit)
    if axis.size != data.shape[0]:
        raise ValueError(f"Spectral axis length {axis.size} does not match cube spectral size {data.shape[0]}.")

    spatial_mask = _validate_spatial_mask(mask, data.shape[1:])
    world_lookup = _world_coordinates(cube, data.shape[1:])

    if ssa_size is None:
        positions = [(y, x) for y in range(data.shape[1]) for x in range(data.shape[2]) if spatial_mask[y, x]]
        pixel_results = _fit_positions_independent(
            data,
            axis,
            positions,
            n_jobs=n_jobs,
            max_components=max_components,
            fit_baseline=fit_baseline,
            progress=progress,
        )
        ssa_labels = None
        ssa_table = None
    else:
        ssa_defs = _build_mask_aware_ssas(spatial_mask, ssa_size=ssa_size, ssa_min_pixels=ssa_min_pixels)
        ssa_results = _fit_ssa_spectra(
            data,
            axis,
            ssa_defs,
            n_jobs=n_jobs,
            max_components=max_components,
            fit_baseline=fit_baseline,
            progress=progress,
        )
        pixel_results = _fit_positions_from_ssas(
            data,
            axis,
            ssa_results,
            n_jobs=n_jobs,
            fit_baseline=fit_baseline,
            progress=progress,
        )
        ssa_labels = _ssa_label_map(data.shape[1:], ssa_results)
        ssa_table = _ssa_results_table(ssa_results)

    n_components = np.zeros(data.shape[1:], dtype=int)
    success = np.zeros(data.shape[1:], dtype=bool)
    noise = np.full(data.shape[1:], np.nan, dtype=float)
    message = np.full(data.shape[1:], "", dtype=object)
    model_cube = np.full(data.shape, np.nan, dtype=float) if store_models else None
    residual_cube = np.full(data.shape, np.nan, dtype=float) if store_models else None
    rows: list[dict[str, Any]] = []

    for pixel_result in pixel_results:
        y = pixel_result["y"]
        x = pixel_result["x"]
        fit = pixel_result["fit"]
        error_message = pixel_result["error"]

        if fit is None:
            message[y, x] = error_message
            continue

        n_components[y, x] = len(fit.components)
        success[y, x] = fit.success
        noise[y, x] = fit.noise
        message[y, x] = fit.message

        if store_models:
            model_cube[:, y, x] = _restore_spectral_order(fit.x_axis, fit.model, axis)
            residual_cube[:, y, x] = _restore_spectral_order(fit.x_axis, fit.residual, axis)

        world_x, world_y = world_lookup[y, x]
        for component_number, component in enumerate(fit.components, start=1):
            rows.append(
                {
                    "ssa_id": pixel_result["ssa_id"],
                    "y": y,
                    "x": x,
                    "component": component_number,
                    "amplitude": component.amplitude,
                    "center": component.center,
                    "fwhm": component.fwhm,
                    "integral": component.integral,
                    "amplitude_error": component.amplitude_error,
                    "center_error": component.center_error,
                    "fwhm_error": component.fwhm_error,
                    "noise": fit.noise,
                    "success": fit.success,
                    "message": fit.message,
                    "world_x": world_x,
                    "world_y": world_y,
                }
            )

    component_table = Table(rows=rows, names=_component_table_columns()) if rows else _empty_component_table()
    return FitCubeResult(
        component_table=component_table,
        n_components=n_components,
        success=success,
        noise=noise,
        message=message,
        model_cube=model_cube,
        residual_cube=residual_cube,
        ssa_labels=ssa_labels,
        ssa_table=ssa_table,
    )


def _cube_data_values(cube: Any) -> np.ndarray:
    if hasattr(cube, "unmasked_data"):
        data = cube.unmasked_data[:]
    else:
        data = cube[:]
    if hasattr(data, "value"):
        data = data.value
    return np.asarray(data, dtype=float)


def _fit_positions_independent(
    data: np.ndarray,
    spectral_axis: np.ndarray,
    positions: list[tuple[int, int]],
    *,
    n_jobs: int,
    max_components: int | None,
    fit_baseline: bool,
    progress: bool,
) -> list[dict[str, Any]]:
    worker = delayed(_fit_one_pixel)
    if n_jobs == 1:
        result_iter = (
            _fit_one_pixel(
                y,
                x,
                data[:, y, x],
                spectral_axis,
                max_components=max_components,
                fit_baseline=fit_baseline,
            )
            for y, x in positions
        )
    else:
        result_iter = Parallel(n_jobs=n_jobs, prefer="threads", return_as="generator")(
            worker(
                y,
                x,
                data[:, y, x],
                spectral_axis,
                max_components=max_components,
                fit_baseline=fit_baseline,
            )
            for y, x in positions
        )
    return list(_progress_iter(result_iter, total=len(positions), enabled=progress, desc="Fitting spectra"))


def _fit_positions_from_ssas(
    data: np.ndarray,
    spectral_axis: np.ndarray,
    ssa_results: list[dict[str, Any]],
    *,
    n_jobs: int,
    fit_baseline: bool,
    progress: bool,
) -> list[dict[str, Any]]:
    tasks = [
        (ssa_result, y, x)
        for ssa_result in ssa_results
        if ssa_result["fit"] is not None and len(ssa_result["fit"].components) > 0
        for y, x in ssa_result["pixels"]
    ]
    skipped = [
        {
            "y": y,
            "x": x,
            "ssa_id": ssa_result["ssa_id"],
            "fit": None,
            "error": ssa_result["message"],
        }
        for ssa_result in ssa_results
        if ssa_result["fit"] is None or len(ssa_result["fit"].components) == 0
        for y, x in ssa_result["pixels"]
    ]

    worker = delayed(_fit_one_pixel_from_ssa)
    if n_jobs == 1:
        result_iter = (
            _fit_one_pixel_from_ssa(
                ssa_result,
                y,
                x,
                data[:, y, x],
                spectral_axis,
                fit_baseline=fit_baseline,
            )
            for ssa_result, y, x in tasks
        )
    else:
        result_iter = Parallel(n_jobs=n_jobs, prefer="threads", return_as="generator")(
            worker(
                ssa_result,
                y,
                x,
                data[:, y, x],
                spectral_axis,
                fit_baseline=fit_baseline,
            )
            for ssa_result, y, x in tasks
        )
    fitted = list(_progress_iter(result_iter, total=len(tasks), enabled=progress, desc="Fitting SSA pixels"))
    return fitted + skipped


def _build_mask_aware_ssas(
    spatial_mask: np.ndarray,
    *,
    ssa_size: int | tuple[int, int],
    ssa_min_pixels: int,
) -> list[dict[str, Any]]:
    y_size, x_size = _normalize_ssa_size(ssa_size)
    if ssa_min_pixels < 1:
        raise ValueError("ssa_min_pixels must be >= 1.")

    ys, xs = np.where(spatial_mask)
    if ys.size == 0:
        return []

    y_min, y_max = int(ys.min()), int(ys.max()) + 1
    x_min, x_max = int(xs.min()), int(xs.max()) + 1
    ssa_defs: list[dict[str, Any]] = []
    ssa_id = 1

    for y0 in range(y_min, y_max, y_size):
        y1 = min(y0 + y_size, spatial_mask.shape[0])
        for x0 in range(x_min, x_max, x_size):
            x1 = min(x0 + x_size, spatial_mask.shape[1])
            local_y, local_x = np.where(spatial_mask[y0:y1, x0:x1])
            if local_y.size < ssa_min_pixels:
                continue
            pixels = [(int(y0 + y), int(x0 + x)) for y, x in zip(local_y, local_x)]
            ssa_defs.append(
                {
                    "ssa_id": ssa_id,
                    "y_min": y0,
                    "y_max": y1,
                    "x_min": x0,
                    "x_max": x1,
                    "pixels": pixels,
                }
            )
            ssa_id += 1
    return ssa_defs


def _normalize_ssa_size(ssa_size: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(ssa_size, tuple):
        if len(ssa_size) != 2:
            raise ValueError("ssa_size tuple must be (y_size, x_size).")
        y_size, x_size = ssa_size
    else:
        y_size = x_size = ssa_size
    y_size = int(y_size)
    x_size = int(x_size)
    if y_size < 1 or x_size < 1:
        raise ValueError("ssa_size values must be >= 1.")
    return y_size, x_size


def _fit_ssa_spectra(
    data: np.ndarray,
    spectral_axis: np.ndarray,
    ssa_defs: list[dict[str, Any]],
    *,
    n_jobs: int,
    max_components: int | None,
    fit_baseline: bool,
    progress: bool,
) -> list[dict[str, Any]]:
    worker = delayed(_fit_one_ssa)
    if n_jobs == 1:
        result_iter = (
            _fit_one_ssa(
                data,
                spectral_axis,
                ssa_def,
                max_components=max_components,
                fit_baseline=fit_baseline,
            )
            for ssa_def in ssa_defs
        )
    else:
        result_iter = Parallel(n_jobs=n_jobs, prefer="threads", return_as="generator")(
            worker(
                data,
                spectral_axis,
                ssa_def,
                max_components=max_components,
                fit_baseline=fit_baseline,
            )
            for ssa_def in ssa_defs
        )
    return list(_progress_iter(result_iter, total=len(ssa_defs), enabled=progress, desc="Fitting SSAs"))


def _fit_one_ssa(
    data: np.ndarray,
    spectral_axis: np.ndarray,
    ssa_def: dict[str, Any],
    *,
    max_components: int | None,
    fit_baseline: bool,
) -> dict[str, Any]:
    pixels = ssa_def["pixels"]
    spectra = np.stack([data[:, y, x] for y, x in pixels], axis=1)
    finite = np.isfinite(spectra)
    counts = finite.sum(axis=1)
    sums = np.where(finite, spectra, 0.0).sum(axis=1)
    mean_spectrum = np.full(spectra.shape[0], np.nan, dtype=float)
    np.divide(sums, counts, out=mean_spectrum, where=counts > 0)
    result = dict(ssa_def)
    result["mean_spectrum"] = mean_spectrum
    result["initial_guess"] = None
    result["fit"] = None
    result["message"] = ""

    if np.count_nonzero(np.isfinite(mean_spectrum)) < 5:
        result["message"] = "SSA mean spectrum has fewer than five finite spectral channels."
        return result

    try:
        initial_guess = detect_initial_components(
            mean_spectrum,
            spectral_axis,
            fit_baseline=fit_baseline,
            max_components=max_components,
        )
        fit = fit_spectrum_from_components(
            mean_spectrum,
            spectral_axis,
            initial_guess.components,
            fit_baseline=fit_baseline,
            baseline_hint=initial_guess,
        )
        result["initial_guess"] = initial_guess
        result["fit"] = fit
        result["message"] = fit.message
    except Exception as exc:
        result["message"] = str(exc)
    return result


def _fit_one_pixel_from_ssa(
    ssa_result: dict[str, Any],
    y: int,
    x: int,
    spectrum: np.ndarray,
    spectral_axis: np.ndarray,
    *,
    fit_baseline: bool,
) -> dict[str, Any]:
    if np.count_nonzero(np.isfinite(spectrum)) < 5:
        return {
            "y": y,
            "x": x,
            "ssa_id": ssa_result["ssa_id"],
            "fit": None,
            "error": "Fewer than five finite spectral channels.",
        }

    ssa_fit = ssa_result["fit"]
    baseline_hint = ssa_result["initial_guess"]
    try:
        fit = fit_spectrum_from_components(
            spectrum,
            spectral_axis,
            ssa_fit.components,
            fit_baseline=fit_baseline,
            baseline_hint=baseline_hint,
        )
        return {"y": y, "x": x, "ssa_id": ssa_result["ssa_id"], "fit": fit, "error": ""}
    except Exception as exc:
        return {"y": y, "x": x, "ssa_id": ssa_result["ssa_id"], "fit": None, "error": str(exc)}


def _ssa_label_map(spatial_shape: tuple[int, int], ssa_results: list[dict[str, Any]]) -> np.ndarray:
    labels = np.zeros(spatial_shape, dtype=int)
    for ssa_result in ssa_results:
        for y, x in ssa_result["pixels"]:
            labels[y, x] = ssa_result["ssa_id"]
    return labels


def _ssa_results_table(ssa_results: list[dict[str, Any]]) -> Table:
    if not ssa_results:
        return _empty_ssa_table()

    rows = []
    for ssa_result in ssa_results:
        fit = ssa_result["fit"]
        rows.append(
            {
                "ssa_id": ssa_result["ssa_id"],
                "y_min": ssa_result["y_min"],
                "y_max": ssa_result["y_max"],
                "x_min": ssa_result["x_min"],
                "x_max": ssa_result["x_max"],
                "n_pixels": len(ssa_result["pixels"]),
                "n_components": len(fit.components) if fit is not None else 0,
                "success": bool(fit.success) if fit is not None else False,
                "noise": float(fit.noise) if fit is not None else np.nan,
                "message": ssa_result["message"],
            }
        )
    return Table(rows=rows, names=_ssa_table_columns())


def _ssa_table_columns() -> list[str]:
    return ["ssa_id", "y_min", "y_max", "x_min", "x_max", "n_pixels", "n_components", "success", "noise", "message"]


def _empty_ssa_table() -> Table:
    return Table(
        names=_ssa_table_columns(),
        dtype=[int, int, int, int, int, int, int, bool, float, object],
    )


def _progress_iter(iterable: Any, *, total: int, enabled: bool, desc: str) -> Iterator[Any]:
    if not enabled:
        yield from iterable
        return

    try:
        from tqdm.auto import tqdm
    except Exception:
        yield from _stderr_progress_iter(iterable, total=total, desc=desc)
        return

    yield from tqdm(iterable, total=total, desc=desc, unit="pix")


def _stderr_progress_iter(iterable: Any, *, total: int, desc: str) -> Iterator[Any]:
    if total <= 0:
        yield from iterable
        return

    width = 30
    update_every = max(1, total // 100)

    def write_progress(count: int) -> None:
        fraction = count / total
        filled = min(width, int(round(width * fraction)))
        bar = "#" * filled + "-" * (width - filled)
        sys.stderr.write(f"\r{desc}: |{bar}| {count}/{total} pix")
        sys.stderr.flush()

    write_progress(0)
    for count, item in enumerate(iterable, start=1):
        yield item
        if count == total or count % update_every == 0:
            write_progress(count)
    sys.stderr.write("\n")
    sys.stderr.flush()


def _spectral_axis_values(cube: Any, spectral_axis: np.ndarray | None, spectral_unit: str | None) -> np.ndarray:
    if spectral_axis is not None:
        if hasattr(spectral_axis, "to_value"):
            values = spectral_axis.to_value(spectral_unit) if spectral_unit is not None else spectral_axis.value
        else:
            values = spectral_axis
        return np.asarray(values, dtype=float)

    axis = cube.spectral_axis
    if spectral_unit is not None and hasattr(axis, "to"):
        axis = axis.to(spectral_unit)
    values = axis.value if hasattr(axis, "value") else axis
    return np.asarray(values, dtype=float)


def _validate_spatial_mask(mask: np.ndarray | None, spatial_shape: tuple[int, int]) -> np.ndarray:
    if mask is None:
        return np.ones(spatial_shape, dtype=bool)
    spatial_mask = np.asarray(mask, dtype=bool)
    if spatial_mask.shape != spatial_shape:
        raise ValueError(f"mask must have shape {spatial_shape}, got {spatial_mask.shape}.")
    return spatial_mask


def _fit_one_pixel(
    y: int,
    x: int,
    spectrum: np.ndarray,
    spectral_axis: np.ndarray,
    *,
    max_components: int | None,
    fit_baseline: bool,
) -> dict[str, Any]:
    if np.count_nonzero(np.isfinite(spectrum)) < 5:
        return {"y": y, "x": x, "ssa_id": 0, "fit": None, "error": "Fewer than five finite spectral channels."}

    try:
        fit = fit_spectrum(
            spectrum,
            spectral_axis,
            fit_baseline=fit_baseline,
            max_components=max_components,
        )
        return {"y": y, "x": x, "ssa_id": 0, "fit": fit, "error": ""}
    except Exception as exc:
        return {"y": y, "x": x, "ssa_id": 0, "fit": None, "error": str(exc)}


def _restore_spectral_order(fit_axis: np.ndarray, values: np.ndarray, original_axis: np.ndarray) -> np.ndarray:
    if fit_axis.shape == original_axis.shape and np.allclose(fit_axis, original_axis, equal_nan=True):
        return values

    restored = np.full(original_axis.shape, np.nan, dtype=float)
    finite = np.isfinite(original_axis)
    original_finite = original_axis[finite]
    if fit_axis.shape == original_finite.shape and np.allclose(fit_axis, original_finite, equal_nan=True):
        restored[finite] = values
        return restored

    finite_indices = np.where(finite)[0]
    order = finite_indices[np.argsort(original_finite)]
    restored[order[: values.size]] = values
    return restored


def _world_coordinates(cube: Any, spatial_shape: tuple[int, int]) -> np.ndarray:
    world = np.full((*spatial_shape, 2), np.nan, dtype=float)
    try:
        y_grid, x_grid = np.indices(spatial_shape)
        coords = pixel_to_skycoord(x_grid, y_grid, cube.wcs.celestial, origin=0)
        world[..., 0] = coords.ra.deg
        world[..., 1] = coords.dec.deg
    except Exception:
        pass
    return world


def _component_table_columns() -> list[str]:
    return [
        "ssa_id",
        "y",
        "x",
        "component",
        "amplitude",
        "center",
        "fwhm",
        "integral",
        "amplitude_error",
        "center_error",
        "fwhm_error",
        "noise",
        "success",
        "message",
        "world_x",
        "world_y",
    ]


def _empty_component_table() -> Table:
    names = _component_table_columns()
    dtypes = [
        int,
        int,
        int,
        int,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        bool,
        object,
        float,
        float,
    ]
    return Table(names=names, dtype=dtypes)
