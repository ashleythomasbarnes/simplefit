from dataclasses import fields, replace
from pathlib import Path

import numpy as np
import pytest

from simplefit import (
    FitConfig,
    GaussianComponent,
    SpectrumInitialGuess,
    detect_initial_components,
    fit_cube,
    fit_spectrum,
    fit_spectrum_from_components,
)
from simplefit import spectrum_fitting


def _line_spectrum() -> tuple[np.ndarray, np.ndarray]:
    axis = np.arange(64, dtype=float)
    data = np.zeros(64, dtype=float)
    data[25:32] = 10.0
    return data, axis


class _ArrayCube:
    def __init__(self, data: np.ndarray, spectral_axis: np.ndarray):
        self._data = data
        self.spectral_axis = spectral_axis

    def __getitem__(self, item):
        return self._data[item]


def test_default_config_preserves_implicit_behavior():
    data, axis = _line_spectrum()

    implicit_guess = detect_initial_components(data, axis, fit_baseline=False)
    explicit_guess = detect_initial_components(data, axis, fit_baseline=False, config=FitConfig())
    assert implicit_guess == explicit_guess

    implicit_fit = fit_spectrum(data, axis, fit_baseline=False)
    explicit_fit = fit_spectrum(data, axis, fit_baseline=False, config=FitConfig())
    assert implicit_fit.success == explicit_fit.success
    assert np.allclose(implicit_fit.model, explicit_fit.model)
    assert np.allclose(implicit_fit.residual, explicit_fit.residual)


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("preprocess_target_channels", 0),
        ("smoothing_passes", -1),
        ("histogram_bins", 1),
        ("histogram_min_bins", 1),
        ("histogram_initial_sigma_scale", 0),
        ("histogram_maxfev", 0),
        ("line_free_sigma", 0),
        ("baseline_sigma", 0),
        ("signal_sigma", 0),
        ("min_signal_channels", 0),
        ("segment_mean_sigma", 0),
        ("split_mean_sigma", 0),
        ("split_min_channels", 0),
        ("split_window_channels", 4),
        ("initial_fwhm_fraction", 0),
        ("min_fwhm_channels", 0),
        ("max_fwhm_axis_spans", 0),
        ("fit_maxfev", 0),
    ],
)
def test_config_validation_names_invalid_field(field_name, invalid_value):
    with pytest.raises(ValueError, match=field_name):
        replace(FitConfig(), **{field_name: invalid_value})


def test_common_detection_controls_change_candidate_selection():
    data, axis = _line_spectrum()

    assert len(detect_initial_components(data, axis, fit_baseline=False).components) == 1
    assert (
        len(
            detect_initial_components(
                data,
                axis,
                fit_baseline=False,
                config=FitConfig(signal_sigma=20.0),
            ).components
        )
        == 0
    )
    assert (
        len(
            detect_initial_components(
                data,
                axis,
                fit_baseline=False,
                config=FitConfig(min_signal_channels=20),
            ).components
        )
        == 0
    )


def test_preprocessing_and_initial_width_controls_are_applied():
    data, axis = _line_spectrum()
    unbinned = spectrum_fitting._profile_preprocessing(data, FitConfig(preprocess_target_channels=128))
    binned = spectrum_fitting._profile_preprocessing(data, FitConfig(preprocess_target_channels=16))
    unsmoothed = spectrum_fitting._profile_preprocessing(data, FitConfig(smoothing_passes=0))
    assert unbinned.size == 64
    # 64 // 16 + 1 gives a five-channel bin width, including a final partial bin.
    assert binned.size == 13
    assert not np.array_equal(unsmoothed, unbinned)

    narrow = detect_initial_components(data, axis, fit_baseline=False, config=FitConfig(initial_fwhm_fraction=0.5))
    broad = detect_initial_components(data, axis, fit_baseline=False, config=FitConfig(initial_fwhm_fraction=1.0))
    assert broad.components[0].fwhm == pytest.approx(2 * narrow.components[0].fwhm)


def test_histogram_starting_scale_and_iteration_limit_are_forwarded(monkeypatch):
    calls = []

    def fake_curve_fit(function, x, y, *, p0, bounds, maxfev):
        calls.append((np.asarray(p0), maxfev))
        return np.asarray(p0), np.eye(len(p0))

    monkeypatch.setattr(spectrum_fitting, "curve_fit", fake_curve_fit)
    values = np.linspace(-2.0, 2.0, 101) ** 3
    config = FitConfig(histogram_initial_sigma_scale=0.75, histogram_maxfev=321)
    spectrum_fitting._histogram_gaussian_fit(values, 12, config)

    assert calls[0][1] == 321
    histogram_step = (values.max() - values.min()) / 12
    assert calls[0][0][2] == pytest.approx(max(histogram_step, np.std(values) * 0.75))


def test_final_fit_bounds_and_iteration_limit_are_forwarded(monkeypatch):
    recorded = {}

    def fake_curve_fit(function, x, y, *, p0, bounds, maxfev):
        recorded["p0"] = np.asarray(p0)
        recorded["bounds"] = tuple(np.asarray(value) for value in bounds)
        recorded["maxfev"] = maxfev
        return np.asarray(p0), np.eye(len(p0))

    monkeypatch.setattr(spectrum_fitting, "curve_fit", fake_curve_fit)
    data, axis = _line_spectrum()
    hint = SpectrumInitialGuess([], -1, 0.0, 0.0, 1.0)
    config = FitConfig(min_fwhm_channels=2.0, max_fwhm_axis_spans=0.1, fit_maxfev=4321)
    fit_spectrum_from_components(
        data,
        axis,
        [GaussianComponent(10.0, 28.0, 3.0)],
        fit_baseline=False,
        baseline_hint=hint,
        config=config,
    )

    assert recorded["maxfev"] == 4321
    assert recorded["bounds"][0][-1] == pytest.approx(2.0)
    assert recorded["bounds"][1][-1] == pytest.approx(6.3)


@pytest.mark.parametrize(("n_jobs", "ssa_size"), [(1, None), (2, None), (1, 1), (2, 1)])
def test_cube_paths_forward_config(n_jobs, ssa_size):
    data, axis = _line_spectrum()
    cube = _ArrayCube(np.stack([data, data], axis=1).reshape(64, 1, 2), axis)

    result = fit_cube(
        cube,
        n_jobs=n_jobs,
        ssa_size=ssa_size,
        chunk_size=1,
        fit_baseline=False,
        progress=False,
        config=FitConfig(signal_sigma=20.0),
    )
    assert np.array_equal(result.n_components, np.zeros((1, 2), dtype=int))


def test_every_config_field_is_documented_in_api_and_readme():
    readme = (Path(__file__).parents[1] / "README.md").read_text()
    class_doc = FitConfig.__doc__ or ""

    for config_field in fields(FitConfig):
        assert config_field.name in class_doc
        assert f"`{config_field.name}`" in readme
