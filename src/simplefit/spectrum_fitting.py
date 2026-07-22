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


@dataclass(frozen=True)
class FitConfig:
    """Tuning parameters for automatic component detection and fitting.

    All sigma thresholds are dimensionless multiples of the histogram-derived
    noise standard deviation. Channel counts are explicitly identified as
    referring to the original or preprocessed spectrum.

    Attributes
    ----------
    preprocess_target_channels
        Positive target used by the preprocessing bin-mean stage. The bin
        width is ``n_original_channels // preprocess_target_channels + 1``;
        therefore smaller values average more original channels together and
        produce stronger smoothing. Default: 128.
    smoothing_passes
        Number of Hanning ``[0.25, 0.5, 0.25]`` passes applied after binning to
        both the spectral axis and intensity profile. This is a non-negative,
        dimensionless integer; zero disables Hanning smoothing. Larger values
        suppress channel-scale structure more strongly. Default: 2.
    histogram_bins
        Optional integer of at least two that directly sets the number of bins
        in all histogram noise fits. ``None`` selects the automatic square-root
        rule; setting a larger value increases histogram resolution but reduces
        the count per bin. Default: ``None``.
    histogram_min_bins
        Integer lower bound of at least two for automatic binning of the
        preprocessed intensity histogram used for line detection. It does not
        alter the endpoint histogram used for baseline discovery. Larger values
        give that detection histogram finer resolution. Default: 8.
    histogram_initial_sigma_scale
        Positive, dimensionless multiplier applied to the sample standard
        deviation when initializing the sigma of every histogram Gaussian.
        It changes the optimizer's starting point, not the returned noise
        directly; larger values start with a broader Gaussian. Default: 0.5.
    histogram_maxfev
        Positive maximum number of model evaluations allowed for each
        histogram Gaussian fit. Larger values may help difficult noise
        histograms converge at additional cost. Default: 10,000.
    line_free_sigma
        Positive sigma half-width around the mirrored-profile histogram center
        used to select line-free runs for baseline endpoint estimates. It is
        evaluated on the original intensity samples after mirroring; larger
        values accept more samples as line-free. Default: 2.0.
    baseline_sigma
        Positive sigma multiplier used to classify the estimated baseline as
        absent, constant, or linear. The width is based on the preprocessed
        intensity noise, while endpoint values come from original channels.
        Larger values favor simpler baselines. Default: 3.0.
    signal_sigma
        Positive per-channel sigma threshold for starting and ending candidate
        signal runs in the preprocessed intensity profile. Larger values detect
        only stronger excursions. Default: 2.0.
    min_signal_channels
        Positive minimum number of connected *original* spectral channels
        represented by a candidate signal run. Larger values reject narrower
        features. Default: 4.
    segment_mean_sigma
        Positive sigma threshold that the mean intensity of a candidate run in
        the preprocessed profile must exceed before it becomes a line segment.
        This is distinct from ``signal_sigma``, which marks individual
        channels. Larger values reject low-average-S/N runs. Default: 3.0.
    split_mean_sigma
        Positive absolute mean-S/N threshold above which an accepted segment
        may be searched for multiple components. It operates on preprocessed
        channels; larger values split only stronger segments. Default: 4.0.
    split_min_channels
        Positive minimum number of connected *preprocessed* channels required
        before the multi-component splitting search runs. Larger values leave
        more segments unsplit. Default: 12.
    split_window_channels
        Odd integer of at least three giving the local preprocessed-channel
        window used to find extrema that divide blended components. Larger
        windows look for broader turning points. Default: 5.
    initial_fwhm_fraction
        Positive, dimensionless fraction of an accepted segment's width on the
        original spectral axis used for its initial Gaussian FWHM. Larger
        values start the final fit with broader components. Default: 0.5.
    min_fwhm_channels
        Positive number of typical original-channel widths used as the lower
        FWHM bound in the final nonlinear fit. Larger values prevent narrower
        fitted components. Default: 0.25.
    max_fwhm_axis_spans
        Positive number of complete spectral-axis spans used as the upper FWHM
        bound in the final nonlinear fit. Larger values permit broader fitted
        components. Default: 2.0.
    fit_maxfev
        Positive maximum number of model evaluations in the final simultaneous
        baseline-plus-Gaussian fit. Larger values may improve convergence at
        additional cost. Default: 50,000.
    """

    preprocess_target_channels: int = 128
    smoothing_passes: int = 2
    histogram_bins: int | None = None
    histogram_min_bins: int = 8
    histogram_initial_sigma_scale: float = 0.5
    histogram_maxfev: int = 10_000
    line_free_sigma: float = 2.0
    baseline_sigma: float = 3.0
    signal_sigma: float = 2.0
    min_signal_channels: int = 4
    segment_mean_sigma: float = 3.0
    split_mean_sigma: float = 4.0
    split_min_channels: int = 12
    split_window_channels: int = 5
    initial_fwhm_fraction: float = 0.5
    min_fwhm_channels: float = 0.25
    max_fwhm_axis_spans: float = 2.0
    fit_maxfev: int = 50_000

    def __post_init__(self) -> None:
        positive_integers = (
            "preprocess_target_channels",
            "histogram_maxfev",
            "min_signal_channels",
            "split_min_channels",
            "fit_maxfev",
        )
        for name in positive_integers:
            value = getattr(self, name)
            if not isinstance(value, (int, np.integer)) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")

        if (
            not isinstance(self.histogram_min_bins, (int, np.integer))
            or isinstance(self.histogram_min_bins, bool)
            or self.histogram_min_bins < 2
        ):
            raise ValueError("histogram_min_bins must be an integer greater than or equal to 2.")

        if (
            not isinstance(self.smoothing_passes, (int, np.integer))
            or isinstance(self.smoothing_passes, bool)
            or self.smoothing_passes < 0
        ):
            raise ValueError("smoothing_passes must be a non-negative integer.")
        if self.histogram_bins is not None and (
            not isinstance(self.histogram_bins, (int, np.integer))
            or isinstance(self.histogram_bins, bool)
            or self.histogram_bins < 2
        ):
            raise ValueError("histogram_bins must be None or an integer greater than or equal to 2.")
        if (
            not isinstance(self.split_window_channels, (int, np.integer))
            or isinstance(self.split_window_channels, bool)
            or self.split_window_channels < 3
            or self.split_window_channels % 2 == 0
        ):
            raise ValueError("split_window_channels must be an odd integer greater than or equal to 3.")

        positive_numbers = (
            "histogram_initial_sigma_scale",
            "line_free_sigma",
            "baseline_sigma",
            "signal_sigma",
            "segment_mean_sigma",
            "split_mean_sigma",
            "initial_fwhm_fraction",
            "min_fwhm_channels",
            "max_fwhm_axis_spans",
        )
        for name in positive_numbers:
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float, np.integer, np.floating))
                or not np.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be a positive finite number.")


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


def _profile_preprocessing(values: np.ndarray, config: FitConfig) -> np.ndarray:
    processed = _bin_mean(values, values.size // config.preprocess_target_channels + 1)
    for _ in range(config.smoothing_passes):
        processed = _hanning_smooth(processed)
    return processed


def _index_by_value(values: np.ndarray, target_value: float) -> int | None:
    for i in range(values.size - 1):
        if values[i] <= target_value < values[i + 1]:
            return i
        if values[i] >= target_value > values[i + 1]:
            return i
        if target_value == values[i + 1]:
            return i + 1
    return None


def _histogram_gaussian_fit(values: np.ndarray, bins: int, config: FitConfig) -> tuple[float, float]:
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

    initial = [
        float(padded_y[peak_index]),
        float(padded_x[peak_index]),
        max(step, np.nanstd(values) * config.histogram_initial_sigma_scale),
    ]

    try:
        params, _ = curve_fit(
            histogram_model,
            padded_x,
            padded_y,
            p0=initial,
            bounds=([0.0, np.nanmin(values), np.finfo(float).eps], [np.inf, np.nanmax(values), np.inf]),
            maxfev=config.histogram_maxfev,
        )
        center = float(params[1])
        stddev = float(abs(params[2]))
    except Exception:
        center = float(padded_x[peak_index])
        stddev = float(np.nanstd(values))

    if not np.isfinite(stddev) or stddev <= 0:
        stddev = max(step, np.finfo(float).eps)
    return center, stddev


def _line_free_endpoint_estimates(
    x_axis: np.ndarray,
    y_data: np.ndarray,
    config: FitConfig,
) -> tuple[tuple[float, float], tuple[float, float]]:
    flipped_sum = y_data + y_data[::-1]
    bins = config.histogram_bins if config.histogram_bins is not None else int(np.sqrt(flipped_sum.size))
    mean, stddev = _histogram_gaussian_fit(flipped_sum, bins, config)
    floor = mean - config.line_free_sigma * stddev
    ceiling = mean + config.line_free_sigma * stddev

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
    config: FitConfig,
) -> tuple[int, float, float]:
    if not fit_baseline:
        return -1, 0.0, 0.0

    start_point, end_point = _line_free_endpoint_estimates(x_axis, y_data, config)
    x0, y0 = start_point
    x1, y1 = end_point
    y_mean = 0.5 * (y0 + y1)

    if x1 == x0:
        initial_slope = 0.0
    else:
        initial_slope = (y1 - y0) / (x1 - x0)
    initial_intercept = y0 - initial_slope * x0

    valid_width = config.baseline_sigma * residual_noise
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
    config: FitConfig,
) -> list[_LineSegment]:
    segments: list[_LineSegment] = []
    is_signal_started = False
    start = 0
    floor = noise_center - config.signal_sigma * noise_stddev
    ceiling = noise_center + config.signal_sigma * noise_stddev

    for i, value in enumerate(y_smoothed):
        is_signal = value > ceiling or value < floor
        if is_signal and not is_signal_started:
            start = i
            is_signal_started = True
        elif not is_signal and is_signal_started:
            end = i - 1
            is_signal_started = False
            _append_valid_segment(
                segments, x_axis, x_smoothed, y_smoothed, start, end, noise_center, noise_stddev, config
            )
        elif is_signal and is_signal_started and i == y_smoothed.size - 1:
            _append_valid_segment(
                segments, x_axis, x_smoothed, y_smoothed, start, i, noise_center, noise_stddev, config
            )

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
    config: FitConfig,
) -> None:
    start_original = _index_by_value(x_axis, x_smoothed[start])
    end_original = _index_by_value(x_axis, x_smoothed[end])
    if start_original is None or end_original is None:
        return

    if end_original < start_original:
        start_original, end_original = end_original, start_original

    segment_mean = float(np.nanmean(y_smoothed[start : end + 1]))
    enough_channels = end_original - start_original + 1 >= config.min_signal_channels
    enough_signal = abs(segment_mean - noise_center) > config.segment_mean_sigma * noise_stddev
    if enough_channels and enough_signal:
        segments.append(_LineSegment(start, end, start_original, end_original))


def _find_divider_indices(
    segment: _LineSegment,
    x_axis: np.ndarray,
    x_smoothed: np.ndarray,
    y_smoothed: np.ndarray,
    mean_sn: float,
    config: FitConfig,
) -> list[int]:
    local_minima: list[int] = []
    local_maxima: list[int] = []

    window_size = config.split_window_channels
    middle_offset = window_size // 2
    endpoint_indices = {0, window_size - 1}
    for j in range(segment.from_index, max(segment.from_index, segment.to_index - (window_size - 1))):
        window = y_smoothed[j : j + window_size]
        if window.size < window_size:
            continue
        sorted_indices = list(np.argsort(window))
        middle_original = _index_by_value(x_axis, x_smoothed[j + middle_offset])
        if middle_original is None:
            continue
        if set(sorted_indices[-2:]) == endpoint_indices:
            local_minima.append(middle_original)
        if set(sorted_indices[:2]) == endpoint_indices:
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
    config: FitConfig,
) -> list[_LineSegment]:
    final_segments: list[_LineSegment] = []

    for segment in segments:
        segment_values = y_smoothed[segment.from_index : segment.to_index + 1]
        mean_sn = (float(np.nanmean(segment_values)) - noise_center) / noise_stddev
        channel_count = segment.to_index - segment.from_index + 1

        if abs(mean_sn) < config.split_mean_sigma or channel_count < config.split_min_channels:
            final_segments.append(segment)
            continue

        divider_indices = _find_divider_indices(segment, x_axis, x_smoothed, y_smoothed, mean_sn, config)
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
    config: FitConfig,
) -> list[GaussianComponent]:
    components: list[GaussianComponent] = []
    channel_width = _typical_channel_width(x_axis)

    for segment in segments:
        fwhm = (
            abs(x_axis[segment.to_index_original] - x_axis[segment.from_index_original])
            * config.initial_fwhm_fraction
        )
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
    config: FitConfig,
) -> tuple[list[GaussianComponent], int, float, float, float]:
    start_point, end_point = _line_free_endpoint_estimates(x_axis, y_data, config)
    x0, y0 = start_point
    x1, y1 = end_point
    initial_slope = 0.0 if x1 == x0 else (y1 - y0) / (x1 - x0)
    initial_intercept = y0 - initial_slope * x0
    y_adjusted = y_data - (initial_slope * x_axis + initial_intercept)

    x_smoothed = _profile_preprocessing(x_axis, config)
    y_smoothed = _profile_preprocessing(y_adjusted, config)
    bins = (
        config.histogram_bins
        if config.histogram_bins is not None
        else max(int(np.sqrt(y_data.size)), config.histogram_min_bins)
    )
    noise_center, noise_stddev = _histogram_gaussian_fit(y_smoothed, bins, config)

    baseline_order, y_intercept, slope = _estimate_baseline(
        x_axis, y_data, noise_stddev, fit_baseline, config
    )
    if baseline_order == -1:
        y_adjusted = y_data.copy()
    else:
        y_adjusted = y_data - (slope * x_axis + y_intercept)
    y_smoothed = _profile_preprocessing(y_adjusted, config)
    noise_center, noise_stddev = _histogram_gaussian_fit(y_smoothed, bins, config)

    segments = _detect_signal_segments(x_axis, x_smoothed, y_smoothed, noise_center, noise_stddev, config)
    split_segments = _split_segments(
        segments, x_axis, x_smoothed, y_smoothed, noise_center, noise_stddev, config
    )
    components = _initial_components_from_segments(
        split_segments, x_axis, x_smoothed, y_smoothed, noise_center, config
    )
    return components, baseline_order, y_intercept, slope, noise_stddev


def detect_initial_components(
    x_data: np.ndarray,
    x_axis: np.ndarray,
    *,
    fit_baseline: bool = True,
    max_components: int | None = None,
    config: FitConfig | None = None,
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
    config
        Optional :class:`FitConfig` controlling preprocessing, histogram noise
        estimation, baseline classification, signal-segment detection,
        multi-component splitting, and initial component widths. The final-fit
        bounds and ``fit_maxfev`` fields are not used by this detection-only
        function. If omitted, all documented defaults are used.
    """

    config = FitConfig() if config is None else config
    if not isinstance(config, FitConfig):
        raise TypeError("config must be a FitConfig instance or None.")
    y, x = _validate_inputs(np.asarray(x_data), np.asarray(x_axis))
    initial_components, baseline_order, y_intercept, slope, noise = _build_initial_components(
        y, x, fit_baseline, config
    )

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
    config: FitConfig | None = None,
) -> FitResult:
    """Fit a spectrum from supplied Gaussian component guesses.

    Parameters
    ----------
    x_data, x_axis
        One-dimensional intensity and spectral-coordinate arrays.
    components
        Gaussian starting values to refine.
    fit_baseline
        Whether a baseline selected by the heuristic may be fitted.
    baseline_hint
        Optional previously detected baseline and noise information. If absent,
        component detection settings in ``config`` are used to derive it.
    config
        Optional :class:`FitConfig`. Its FWHM-bound and final-optimizer fields
        always control refinement. Detection, histogram, and baseline fields
        are additionally used when ``baseline_hint`` is absent. Defaults are
        used when omitted.
    """

    config = FitConfig() if config is None else config
    if not isinstance(config, FitConfig):
        raise TypeError("config must be a FitConfig instance or None.")
    y, x = _validate_inputs(np.asarray(x_data), np.asarray(x_axis))
    initial_components = [
        GaussianComponent(amplitude=component.amplitude, center=component.center, fwhm=component.fwhm)
        for component in components
    ]

    if baseline_hint is None:
        _, baseline_order, y_intercept, slope, noise = _build_initial_components(y, x, fit_baseline, config)
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
    minimum_fwhm = channel_width * config.min_fwhm_channels
    maximum_fwhm = axis_span * config.max_fwhm_axis_spans
    if minimum_fwhm > maximum_fwhm:
        raise ValueError(
            "FitConfig produces an empty FWHM interval: min_fwhm_channels times the typical channel width "
            "must not exceed max_fwhm_axis_spans times the spectral-axis span."
        )

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
        initial_fwhm = float(np.clip(max(component.fwhm, channel_width), minimum_fwhm, maximum_fwhm))
        gaussian_p0.extend([component.amplitude, component.center, initial_fwhm])
        if component.amplitude >= 0:
            lower.extend([0.0, axis_min, minimum_fwhm])
            upper.extend([np.inf, axis_max, maximum_fwhm])
        else:
            lower.extend([-np.inf, axis_min, minimum_fwhm])
            upper.extend([0.0, axis_max, maximum_fwhm])

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
            maxfev=config.fit_maxfev,
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
    config: FitConfig | None = None,
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
    config
        Optional :class:`FitConfig` used for every detection and refinement
        stage: preprocessing, histogram noise estimation, baseline selection,
        signal detection, segment splitting, initial widths, FWHM bounds, and
        optimizer limits. If omitted, the defaults reproduce the original
        fitting behavior.
    """

    config = FitConfig() if config is None else config
    if not isinstance(config, FitConfig):
        raise TypeError("config must be a FitConfig instance or None.")
    initial_guess = detect_initial_components(
        x_data,
        x_axis,
        fit_baseline=fit_baseline,
        max_components=max_components,
        config=config,
    )
    return fit_spectrum_from_components(
        x_data,
        x_axis,
        initial_guess.components,
        fit_baseline=fit_baseline,
        baseline_hint=initial_guess,
        config=config,
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
