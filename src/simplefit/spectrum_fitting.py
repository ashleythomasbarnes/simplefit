"""CARTA-like Gaussian fitting heuristics for one-dimensional spectra.

This module ports the useful parts of CARTA's TypeScript fitting heuristic into
plain NumPy/SciPy code.  The public entry point is :func:`fit_spectrum`, which
accepts intensity values and their spectral axis, detects likely Gaussian
components, and refines them with a multi-Gaussian least-squares fit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.optimize import curve_fit


FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))
GAUSSIAN_INTEGRAL_FACTOR = np.sqrt(np.pi / (4.0 * np.log(2.0)))


@dataclass
class GaussianComponent:
    """One fitted Gaussian component."""

    amplitude: float
    center: float
    fwhm: float
    amplitude_error: float = np.nan
    center_error: float = np.nan
    fwhm_error: float = np.nan

    @property
    def integral(self) -> float:
        return self.amplitude * self.fwhm * GAUSSIAN_INTEGRAL_FACTOR


@dataclass
class FitResult:
    """Output from :func:`fit_spectrum`."""

    x_axis: np.ndarray
    x_data: np.ndarray
    components: list[GaussianComponent]
    model: np.ndarray
    residual: np.ndarray
    baseline: np.ndarray
    noise: float
    success: bool
    message: str


@dataclass
class SpectrumInitialGuess:
    """Initial components and baseline selected by the CARTA-like detector."""

    components: list[GaussianComponent]
    baseline_order: int
    y_intercept: float
    slope: float
    noise: float


@dataclass(frozen=True)
class _LineSegment:
    from_index: int
    to_index: int
    from_index_original: int
    to_index_original: int


def _gaussian(x_axis: np.ndarray, amplitude: float, center: float, fwhm: float) -> np.ndarray:
    fwhm = max(float(abs(fwhm)), np.finfo(float).eps)
    return amplitude * np.exp(-4.0 * np.log(2.0) * ((x_axis - center) / fwhm) ** 2)


def _multi_gaussian(x_axis: np.ndarray, parameters: Iterable[float]) -> np.ndarray:
    parameters = list(parameters)
    model = np.zeros_like(x_axis, dtype=float)
    for i in range(0, len(parameters), 3):
        model += _gaussian(x_axis, parameters[i], parameters[i + 1], parameters[i + 2])
    return model


def _hanning_smooth(values: np.ndarray) -> np.ndarray:
    if values.size < 3:
        return values.astype(float, copy=True)

    smoothed = values.astype(float, copy=True)
    smoothed[1:-1] = 0.25 * values[:-2] + 0.5 * values[1:-1] + 0.25 * values[2:]
    return smoothed


def _bin_mean(values: np.ndarray, bin_width: int) -> np.ndarray:
    bin_width = max(int(bin_width), 1)
    if bin_width == 1:
        return values.astype(float, copy=True)

    chunks = [values[i : i + bin_width] for i in range(0, values.size, bin_width)]
    return np.array([np.nanmean(chunk) for chunk in chunks if chunk.size], dtype=float)


def _profile_preprocessing(values: np.ndarray) -> np.ndarray:
    processed = _bin_mean(values, values.size // 128 + 1)
    processed = _hanning_smooth(processed)
    return _hanning_smooth(processed)


def _index_by_value(values: np.ndarray, target_value: float) -> int | None:
    for i in range(values.size - 1):
        if values[i] <= target_value < values[i + 1]:
            return i
        if values[i] >= target_value > values[i + 1]:
            return i
        if target_value == values[i + 1]:
            return i + 1
    return None


def _histogram_gaussian_fit(values: np.ndarray, bins: int) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0, 1.0

    bins = max(int(bins), 2)
    hist, edges = np.histogram(values, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    if centers.size < 2:
        return float(np.nanmean(values)), float(np.nanstd(values) or 1.0)

    step = abs(float(centers[1] - centers[0]))
    padded_x = np.r_[centers[0] - step, centers, centers[-1] + step]
    padded_y = np.r_[0.0, hist.astype(float), 0.0]
    peak_index = int(np.nanargmax(padded_y))

    if peak_index in (1, padded_y.size - 2):
        return float(padded_x[peak_index]), max(step, np.finfo(float).eps)

    def histogram_model(x_axis: np.ndarray, amplitude: float, center: float, sigma: float) -> np.ndarray:
        sigma = np.maximum(abs(sigma), np.finfo(float).eps)
        return amplitude * np.exp(-0.5 * ((x_axis - center) / sigma) ** 2)

    initial = [float(padded_y[peak_index]), float(padded_x[peak_index]), max(step, np.nanstd(values) / 2.0)]

    try:
        params, _ = curve_fit(
            histogram_model,
            padded_x,
            padded_y,
            p0=initial,
            bounds=([0.0, np.nanmin(values), np.finfo(float).eps], [np.inf, np.nanmax(values), np.inf]),
            maxfev=10000,
        )
        center = float(params[1])
        stddev = float(abs(params[2]))
    except Exception:
        center = float(padded_x[peak_index])
        stddev = float(np.nanstd(values))

    if not np.isfinite(stddev) or stddev <= 0:
        stddev = max(step, np.finfo(float).eps)
    return center, stddev


def _line_free_endpoint_estimates(x_axis: np.ndarray, y_data: np.ndarray) -> tuple[tuple[float, float], tuple[float, float]]:
    flipped_sum = y_data + y_data[::-1]
    mean, stddev = _histogram_gaussian_fit(flipped_sum, int(np.sqrt(flipped_sum.size)))
    floor = mean - 2.0 * stddev
    ceiling = mean + 2.0 * stddev

    x_means: list[float] = []
    y_means: list[float] = []
    start: int | None = None

    for i, value in enumerate(flipped_sum):
        in_range = floor < value < ceiling
        if in_range and start is None and i <= flipped_sum.size - 2:
            start = i
        elif not in_range and start is not None:
            x_means.append(float(np.nanmean(x_axis[start:i])))
            y_means.append(float(np.nanmean(y_data[start:i])))
            start = None
        elif in_range and start is not None and i == flipped_sum.size - 1:
            x_means.append(float(np.nanmean(x_axis[start:i])))
            y_means.append(float(np.nanmean(y_data[start:i])))

    if len(x_means) <= 1:
        middle = x_axis.size // 2
        x_means = [float(np.nanmean(x_axis[:middle])), float(np.nanmean(x_axis[middle:]))]
        y_means = [float(np.nanmean(y_data[:middle])), float(np.nanmean(y_data[middle:]))]

    return (x_means[0], y_means[0]), (x_means[-1], y_means[-1])


def _estimate_baseline(
    x_axis: np.ndarray,
    y_data: np.ndarray,
    residual_noise: float,
    fit_baseline: bool,
) -> tuple[int, float, float]:
    if not fit_baseline:
        return -1, 0.0, 0.0

    start_point, end_point = _line_free_endpoint_estimates(x_axis, y_data)
    x0, y0 = start_point
    x1, y1 = end_point
    y_mean = 0.5 * (y0 + y1)

    if x1 == x0:
        initial_slope = 0.0
    else:
        initial_slope = (y1 - y0) / (x1 - x0)
    initial_intercept = y0 - initial_slope * x0

    valid_width = 3.0 * residual_noise
    if -valid_width < y0 < valid_width and -valid_width < y1 < valid_width:
        return -1, 0.0, 0.0
    if y_mean - valid_width < y0 < y_mean + valid_width and y_mean - valid_width < y1 < y_mean + valid_width:
        return 0, float(y_mean), 0.0
    return 1, float(initial_intercept), float(initial_slope)


def _detect_signal_segments(
    x_axis: np.ndarray,
    x_smoothed: np.ndarray,
    y_smoothed: np.ndarray,
    noise_center: float,
    noise_stddev: float,
) -> list[_LineSegment]:
    segments: list[_LineSegment] = []
    is_signal_started = False
    start = 0
    floor = noise_center - 2.0 * noise_stddev
    ceiling = noise_center + 2.0 * noise_stddev

    for i, value in enumerate(y_smoothed):
        is_signal = value > ceiling or value < floor
        if is_signal and not is_signal_started:
            start = i
            is_signal_started = True
        elif not is_signal and is_signal_started:
            end = i - 1
            is_signal_started = False
            _append_valid_segment(segments, x_axis, x_smoothed, y_smoothed, start, end, noise_center, noise_stddev)
        elif is_signal and is_signal_started and i == y_smoothed.size - 1:
            _append_valid_segment(segments, x_axis, x_smoothed, y_smoothed, start, i, noise_center, noise_stddev)

    return segments


def _append_valid_segment(
    segments: list[_LineSegment],
    x_axis: np.ndarray,
    x_smoothed: np.ndarray,
    y_smoothed: np.ndarray,
    start: int,
    end: int,
    noise_center: float,
    noise_stddev: float,
) -> None:
    start_original = _index_by_value(x_axis, x_smoothed[start])
    end_original = _index_by_value(x_axis, x_smoothed[end])
    if start_original is None or end_original is None:
        return

    if end_original < start_original:
        start_original, end_original = end_original, start_original

    segment_mean = float(np.nanmean(y_smoothed[start : end + 1]))
    enough_channels = end_original - start_original + 1 >= 4
    enough_signal = abs(segment_mean - noise_center) > 3.0 * noise_stddev
    if enough_channels and enough_signal:
        segments.append(_LineSegment(start, end, start_original, end_original))


def _find_divider_indices(
    segment: _LineSegment,
    x_axis: np.ndarray,
    x_smoothed: np.ndarray,
    y_smoothed: np.ndarray,
    mean_sn: float,
) -> list[int]:
    local_minima: list[int] = []
    local_maxima: list[int] = []

    for j in range(segment.from_index, max(segment.from_index, segment.to_index - 4)):
        window = y_smoothed[j : j + 5]
        if window.size < 5:
            continue
        sorted_indices = list(np.argsort(window))
        middle_original = _index_by_value(x_axis, x_smoothed[j + 2])
        if middle_original is None:
            continue
        if {sorted_indices[3], sorted_indices[4]} == {0, 4}:
            local_minima.append(middle_original)
        if {sorted_indices[0], sorted_indices[1]} == {0, 4}:
            local_maxima.append(middle_original)

    candidates = sorted(set(local_minima + local_maxima))
    dividers = [segment.from_index_original]

    if len(candidates) == 1:
        middle = candidates[0]
        if mean_sn > 0 and middle in local_minima:
            dividers.append(middle)
        elif mean_sn < 0 and middle in local_maxima:
            dividers.append(middle)
    elif len(candidates) == 2:
        left, right = candidates
        dividers.extend(_divider_pair(left, right, local_minima, local_maxima, mean_sn))
    elif len(candidates) >= 3:
        dividers.extend(_divider_sequence(candidates, local_minima, local_maxima, mean_sn))

    dividers.append(segment.to_index_original)
    return sorted(set(idx for idx in dividers if segment.from_index_original <= idx <= segment.to_index_original))


def _divider_pair(
    left: int,
    right: int,
    minima: list[int],
    maxima: list[int],
    mean_sn: float,
) -> list[int]:
    if mean_sn > 0:
        if left in minima and right in minima:
            return [left, right]
        if left in minima and right in maxima:
            return [left]
        if left in maxima and right in minima:
            return [right]
        if left in maxima and right in maxima:
            return [(left + right) // 2]
    else:
        if left in minima and right in minima:
            return [(left + right) // 2]
        if left in minima and right in maxima:
            return [right]
        if left in maxima and right in minima:
            return [left]
        if left in maxima and right in maxima:
            return [left, right]
    return []


def _divider_sequence(
    candidates: list[int],
    minima: list[int],
    maxima: list[int],
    mean_sn: float,
) -> list[int]:
    dividers: list[int] = []
    if mean_sn > 0:
        for k in range(len(candidates) - 2):
            left, middle, right = candidates[k : k + 3]
            if left in minima and (middle in minima or (middle in maxima and k == 0)):
                dividers.append(left)
            elif left in maxima:
                if middle in minima and right in maxima:
                    dividers.append(middle)
                elif middle in maxima:
                    dividers.append((left + middle) // 2)

        last1, last2, last3 = candidates[-1], candidates[-2], candidates[-3]
        if last1 in minima:
            dividers.append(last1)
        if last2 in maxima and last1 in maxima:
            dividers.append((last2 + last1) // 2)
        if last3 in minima and last2 in minima and last1 in maxima:
            dividers.append(last2)
    else:
        for k in range(len(candidates) - 2):
            left, middle, right = candidates[k : k + 3]
            if left in maxima and (middle in maxima or (middle in minima and k == 0)):
                dividers.append(left)
            elif left in minima:
                if middle in maxima and right in minima:
                    dividers.append(middle)
                elif middle in minima:
                    dividers.append((left + middle) // 2)

        last1, last2, last3 = candidates[-1], candidates[-2], candidates[-3]
        if last1 in maxima:
            dividers.append(last1)
        if last2 in minima and last1 in minima:
            dividers.append((last2 + last1) // 2)
        if last3 in maxima and last2 in maxima and last1 in minima:
            dividers.append(last2)

    return dividers


def _split_segments(
    segments: list[_LineSegment],
    x_axis: np.ndarray,
    x_smoothed: np.ndarray,
    y_smoothed: np.ndarray,
    noise_center: float,
    noise_stddev: float,
) -> list[_LineSegment]:
    final_segments: list[_LineSegment] = []

    for segment in segments:
        segment_values = y_smoothed[segment.from_index : segment.to_index + 1]
        mean_sn = (float(np.nanmean(segment_values)) - noise_center) / noise_stddev
        channel_count = segment.to_index - segment.from_index + 1

        if abs(mean_sn) < 4.0 or channel_count < 12:
            final_segments.append(segment)
            continue

        divider_indices = _find_divider_indices(segment, x_axis, x_smoothed, y_smoothed, mean_sn)
        if len(divider_indices) < 2:
            final_segments.append(segment)
            continue

        for left, right in zip(divider_indices[:-1], divider_indices[1:]):
            if right <= left:
                continue
            smoothed_left = _index_by_value(x_smoothed, x_axis[left])
            smoothed_right = _index_by_value(x_smoothed, x_axis[right])
            if smoothed_left is None or smoothed_right is None:
                continue
            if smoothed_right < smoothed_left:
                smoothed_left, smoothed_right = smoothed_right, smoothed_left
            final_segments.append(_LineSegment(smoothed_left, smoothed_right, left, right))

    return final_segments


def _initial_components_from_segments(
    segments: list[_LineSegment],
    x_axis: np.ndarray,
    x_smoothed: np.ndarray,
    y_smoothed: np.ndarray,
    noise_center: float,
) -> list[GaussianComponent]:
    components: list[GaussianComponent] = []
    channel_width = _typical_channel_width(x_axis)

    for segment in segments:
        fwhm = abs(x_axis[segment.to_index_original] - x_axis[segment.from_index_original]) / 2.0
        fwhm = max(float(fwhm), channel_width)
        local = y_smoothed[segment.from_index : segment.to_index + 1]
        if local.size == 0:
            continue
        if float(np.nanmean(local)) > noise_center:
            local_index = int(np.nanargmax(local))
            amplitude = float(local[local_index])
        else:
            local_index = int(np.nanargmin(local))
            amplitude = float(local[local_index])
        center_index = min(segment.from_index + local_index, x_smoothed.size - 1)
        components.append(GaussianComponent(amplitude=amplitude, center=float(x_smoothed[center_index]), fwhm=fwhm))

    return components


def _typical_channel_width(x_axis: np.ndarray) -> float:
    diffs = np.diff(x_axis)
    diffs = np.abs(diffs[np.isfinite(diffs) & (diffs != 0)])
    if diffs.size:
        return float(np.nanmedian(diffs))
    return max(float(np.nanmax(x_axis) - np.nanmin(x_axis)), 1.0)


def _validate_inputs(x_data: np.ndarray, x_axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(x_data, dtype=float)
    x = np.asarray(x_axis, dtype=float)
    if y.ndim != 1 or x.ndim != 1:
        raise ValueError("x_data and x_axis must be one-dimensional arrays.")
    if y.size != x.size:
        raise ValueError("x_data and x_axis must have the same length.")
    if y.size < 5:
        raise ValueError("At least five samples are required for fitting.")

    finite = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(finite) < 5:
        raise ValueError("At least five finite samples are required for fitting.")
    x = x[finite]
    y = y[finite]

    diffs = np.diff(x)
    if not (np.all(diffs > 0) or np.all(diffs < 0)):
        order = np.argsort(x)
        x = x[order]
        y = y[order]

    return y, x


def _build_initial_components(
    y_data: np.ndarray,
    x_axis: np.ndarray,
    fit_baseline: bool,
) -> tuple[list[GaussianComponent], int, float, float, float]:
    start_point, end_point = _line_free_endpoint_estimates(x_axis, y_data)
    x0, y0 = start_point
    x1, y1 = end_point
    initial_slope = 0.0 if x1 == x0 else (y1 - y0) / (x1 - x0)
    initial_intercept = y0 - initial_slope * x0
    y_adjusted = y_data - (initial_slope * x_axis + initial_intercept)

    x_smoothed = _profile_preprocessing(x_axis)
    y_smoothed = _profile_preprocessing(y_adjusted)
    bins = max(int(np.sqrt(y_data.size)), 8)
    noise_center, noise_stddev = _histogram_gaussian_fit(y_smoothed, bins)

    baseline_order, y_intercept, slope = _estimate_baseline(x_axis, y_data, noise_stddev, fit_baseline)
    if baseline_order == -1:
        y_adjusted = y_data.copy()
    else:
        y_adjusted = y_data - (slope * x_axis + y_intercept)
    y_smoothed = _profile_preprocessing(y_adjusted)
    noise_center, noise_stddev = _histogram_gaussian_fit(y_smoothed, bins)

    segments = _detect_signal_segments(x_axis, x_smoothed, y_smoothed, noise_center, noise_stddev)
    split_segments = _split_segments(segments, x_axis, x_smoothed, y_smoothed, noise_center, noise_stddev)
    components = _initial_components_from_segments(split_segments, x_axis, x_smoothed, y_smoothed, noise_center)
    return components, baseline_order, y_intercept, slope, noise_stddev


def detect_initial_components(
    x_data: np.ndarray,
    x_axis: np.ndarray,
    *,
    fit_baseline: bool = True,
    max_components: int | None = None,
) -> SpectrumInitialGuess:
    """Detect CARTA-like initial Gaussian components for one spectrum.

    Parameters
    ----------
    x_data
        Intensity values.
    x_axis
        Spectral coordinate values, such as velocity, frequency, or wavelength.
    fit_baseline
        If true, estimate and fit a constant or linear baseline when the
        heuristic indicates that one is present.
    max_components
        Optional cap on the number of detected components to refine.
    """

    y, x = _validate_inputs(np.asarray(x_data), np.asarray(x_axis))
    initial_components, baseline_order, y_intercept, slope, noise = _build_initial_components(y, x, fit_baseline)

    if max_components is not None and len(initial_components) > max_components:
        initial_components = sorted(initial_components, key=lambda comp: abs(comp.amplitude), reverse=True)[:max_components]
        initial_components = sorted(initial_components, key=lambda comp: comp.center)

    return SpectrumInitialGuess(
        components=initial_components,
        baseline_order=baseline_order,
        y_intercept=y_intercept,
        slope=slope,
        noise=noise,
    )


def fit_spectrum_from_components(
    x_data: np.ndarray,
    x_axis: np.ndarray,
    components: list[GaussianComponent],
    *,
    fit_baseline: bool = True,
    baseline_hint: SpectrumInitialGuess | None = None,
) -> FitResult:
    """Fit a spectrum from supplied Gaussian component guesses."""

    y, x = _validate_inputs(np.asarray(x_data), np.asarray(x_axis))
    initial_components = [
        GaussianComponent(amplitude=component.amplitude, center=component.center, fwhm=component.fwhm)
        for component in components
    ]

    if baseline_hint is None:
        _, baseline_order, y_intercept, slope, noise = _build_initial_components(y, x, fit_baseline)
    else:
        baseline_order = baseline_hint.baseline_order
        y_intercept = baseline_hint.y_intercept
        slope = baseline_hint.slope
        noise = baseline_hint.noise

    if not initial_components:
        baseline = np.zeros_like(y)
        if fit_baseline and baseline_order >= 0:
            baseline = slope * x + y_intercept
        model = baseline.copy()
        return FitResult(
            x_axis=x,
            x_data=y,
            components=[],
            model=model,
            residual=y - model,
            baseline=baseline,
            noise=noise,
            success=False,
            message="No Gaussian components were detected.",
        )

    x_ref = float(np.nanmean(x))
    channel_width = _typical_channel_width(x)
    axis_min = float(np.nanmin(x))
    axis_max = float(np.nanmax(x))
    axis_span = max(axis_max - axis_min, channel_width)

    baseline_p0: list[float] = []
    lower: list[float] = []
    upper: list[float] = []
    if fit_baseline and baseline_order == 0:
        baseline_p0 = [y_intercept]
        lower = [-np.inf]
        upper = [np.inf]
    elif fit_baseline and baseline_order == 1:
        baseline_p0 = [slope * x_ref + y_intercept, slope]
        lower = [-np.inf, -np.inf]
        upper = [np.inf, np.inf]

    gaussian_p0: list[float] = []
    for component in initial_components:
        gaussian_p0.extend([component.amplitude, component.center, max(component.fwhm, channel_width)])
        if component.amplitude >= 0:
            lower.extend([0.0, axis_min, channel_width * 0.25])
            upper.extend([np.inf, axis_max, axis_span * 2.0])
        else:
            lower.extend([-np.inf, axis_min, channel_width * 0.25])
            upper.extend([0.0, axis_max, axis_span * 2.0])

    p0 = np.array(baseline_p0 + gaussian_p0, dtype=float)

    def model_function(x_values: np.ndarray, *parameters: float) -> np.ndarray:
        offset = 0
        baseline = np.zeros_like(x_values, dtype=float)
        if fit_baseline and baseline_order == 0:
            baseline = np.full_like(x_values, parameters[0], dtype=float)
            offset = 1
        elif fit_baseline and baseline_order == 1:
            baseline = parameters[0] + parameters[1] * (x_values - x_ref)
            offset = 2
        return baseline + _multi_gaussian(x_values, parameters[offset:])

    try:
        params, covariance = curve_fit(
            model_function,
            x,
            y,
            p0=p0,
            bounds=(np.array(lower, dtype=float), np.array(upper, dtype=float)),
            maxfev=50000,
        )
        errors = np.sqrt(np.diag(covariance)) if covariance.size else np.full_like(params, np.nan)
        success = True
        message = "Fit converged."
    except Exception as exc:
        params = p0
        errors = np.full_like(params, np.nan)
        success = False
        message = f"Fit did not converge; returning heuristic initial guesses. ({exc})"

    offset = 0
    if fit_baseline and baseline_order == 0:
        baseline = np.full_like(x, params[0], dtype=float)
        offset = 1
    elif fit_baseline and baseline_order == 1:
        baseline = params[0] + params[1] * (x - x_ref)
        offset = 2
    else:
        baseline = np.zeros_like(x)

    fitted_components: list[GaussianComponent] = []
    for i in range(offset, len(params), 3):
        fitted_components.append(
            GaussianComponent(
                amplitude=float(params[i]),
                center=float(params[i + 1]),
                fwhm=float(abs(params[i + 2])),
                amplitude_error=float(errors[i]) if i < errors.size else np.nan,
                center_error=float(errors[i + 1]) if i + 1 < errors.size else np.nan,
                fwhm_error=float(errors[i + 2]) if i + 2 < errors.size else np.nan,
            )
        )

    gaussian_model = _multi_gaussian(x, params[offset:])
    model = baseline + gaussian_model
    residual = y - model

    return FitResult(
        x_axis=x,
        x_data=y,
        components=fitted_components,
        model=model,
        residual=residual,
        baseline=baseline,
        noise=noise,
        success=success,
        message=message,
    )


def fit_spectrum(
    x_data: np.ndarray,
    x_axis: np.ndarray,
    *,
    fit_baseline: bool = True,
    max_components: int | None = None,
) -> FitResult:
    """Fit Gaussian components to a one-dimensional spectrum.

    Parameters
    ----------
    x_data
        Intensity values.
    x_axis
        Spectral coordinate values, such as velocity, frequency, or wavelength.
    fit_baseline
        If true, estimate and fit a constant or linear baseline when the
        heuristic indicates that one is present.
    max_components
        Optional cap on the number of detected components to refine.
    """

    initial_guess = detect_initial_components(
        x_data,
        x_axis,
        fit_baseline=fit_baseline,
        max_components=max_components,
    )
    return fit_spectrum_from_components(
        x_data,
        x_axis,
        initial_guess.components,
        fit_baseline=fit_baseline,
        baseline_hint=initial_guess,
    )


def plot_fit(result: FitResult, ax=None, residual_ax=None):
    """Plot fitted spectrum, components, total model, and residuals."""

    import matplotlib.pyplot as plt

    created_figure = ax is None or residual_ax is None
    if created_figure:
        figure, (ax, residual_ax) = plt.subplots(
            2,
            1,
            figsize=(8, 5),
            sharex=True,
            gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05},
        )
    else:
        figure = ax.figure

    ax.plot(result.x_axis, result.x_data, color="#2f5f9f", lw=1.3, drawstyle="steps-mid", label="Data")
    ax.plot(result.x_axis, result.model, color="#8a4f17", lw=2.0, label="Model")
    if np.any(result.baseline):
        ax.plot(result.x_axis, result.baseline, color="0.45", lw=1.0, ls="--", label="Baseline")

    for index, component in enumerate(result.components, start=1):
        component_model = result.baseline + _gaussian(result.x_axis, component.amplitude, component.center, component.fwhm)
        ax.plot(result.x_axis, component_model, color="#b08a61", lw=1.2, alpha=0.8)
        y_label = result.baseline[np.argmin(np.abs(result.x_axis - component.center))] + 0.55 * component.amplitude
        ax.text(component.center, y_label, str(index), color="0.35", ha="center", va="center", fontsize=9)

    residual_ax.axhline(0.0, color="0.35", lw=1.0)
    residual_ax.plot(result.x_axis, result.residual, color="#8a4f17", marker=".", ms=3, lw=0.8)
    residual_ax.set_ylabel("Residual")
    residual_ax.set_xlabel("Spectral axis")
    ax.set_ylabel("Intensity")
    ax.legend(loc="best", frameon=False)
    ax.grid(alpha=0.25)
    residual_ax.grid(alpha=0.25)
    return figure, (ax, residual_ax)
