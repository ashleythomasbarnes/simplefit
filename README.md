# simplefit

`simplefit` provides lightweight Gaussian fitting helpers for one-dimensional
spectra and spectral cubes. The current fitter is Gaussian-only and uses a
CARTA-style heuristic to detect initial components before refining them with
SciPy.

## Install

From this repository:

```bash
pip install -e .
```

If you are working offline or pip cannot fetch build dependencies, use the local
build tools already installed in your environment:

```bash
pip install -e . --no-build-isolation
```

The optional progress bar dependency can be installed with:

```bash
pip install -e ".[progress]"
```

The PHANGS plotting/post-processing examples also use Plotly and pandas:

```bash
pip install plotly pandas
```

## Single-Spectrum Fitting

```python
import numpy as np
from simplefit import fit_spectrum, plot_fit

fit = fit_spectrum(x_data=spectrum, x_axis=velocity, max_components=None)

print(fit.success, fit.message)
for component in fit.components:
    print(component.center, component.amplitude, component.fwhm, component.integral)

fig, axes = plot_fit(fit)
```

`fit_spectrum` returns a `FitResult` with:

- `components`: fitted Gaussian components.
- `model`: total model sampled on the input spectral axis.
- `residual`: `x_data - model`.
- `baseline`: fitted baseline array.
- `noise`: histogram-derived noise estimate.
- `success` and `message`: fit status.

Set `fit_baseline=False` to skip the baseline heuristic, or `max_components` to
cap the number of detected Gaussian components passed into the final fit.

### Configuring the fitting heuristic

The original fitting behavior remains the default; no configuration is needed
for an ordinary call. To change the most commonly adjusted line-detection
controls, create a `FitConfig`:

```python
from simplefit import FitConfig, fit_spectrum

config = FitConfig(
    smoothing_passes=1,
    signal_sigma=3.0,
    min_signal_channels=5,
)
fit = fit_spectrum(spectrum, velocity, config=config)
```

`FitConfig` is immutable, so one instance can safely be reused across spectra:

```python
fits = [fit_spectrum(spectrum, velocity, config=config) for spectrum in spectra]
```

The parameters control distinct stages of the heuristic. In particular,
`signal_sigma` tests each preprocessed channel to form a connected candidate
run, whereas `segment_mean_sigma` subsequently tests the mean of the entire
run. Likewise, `min_signal_channels` measures the candidate on the **original**
spectral grid, while `split_min_channels` measures it on the binned and smoothed
grid.

| Parameter | Default | Allowed values | Exact effect |
|---|---:|---|---|
| `preprocess_target_channels` | `128` | integer `>= 1` | Sets preprocessing bin width to `n_original_channels // value + 1`. A smaller value averages more original channels per bin; a larger value preserves more channel-scale detail. |
| `smoothing_passes` | `2` | integer `>= 0` | Applies this many Hanning `[0.25, 0.5, 0.25]` passes to both the binned spectral axis and binned intensity profile. Zero disables Hanning smoothing; larger values smooth more strongly. |
| `histogram_bins` | `None` | `None` or integer `>= 2` | Overrides the bin count in every histogram Gaussian fit. `None` uses automatic square-root binning. More bins increase resolution but reduce samples per bin. |
| `histogram_min_bins` | `8` | integer `>= 2` | Lower bound on automatic bins for the preprocessed intensity histogram used to estimate detection noise. It does not affect the mirrored endpoint histogram. Larger values make the detection histogram finer. |
| `histogram_initial_sigma_scale` | `0.5` | finite number `> 0` | Multiplies the sample standard deviation to initialize histogram-Gaussian sigma. It affects the optimizer starting point, not the noise result directly; larger values start broader. |
| `histogram_maxfev` | `10000` | integer `>= 1` | Maximum model evaluations for each histogram Gaussian fit. Larger values may improve convergence but take longer. |
| `line_free_sigma` | `2.0` | finite number `> 0` | Selects mirrored-profile samples within this many histogram-derived sigmas as line-free baseline runs. It acts on original intensity samples; larger values accept more samples. |
| `baseline_sigma` | `3.0` | finite number `> 0` | Sets the tolerance used to classify the baseline as absent, constant, or linear. Endpoint values come from original channels and the noise comes from the preprocessed profile. Larger values favor simpler baselines. |
| `signal_sigma` | `2.0` | finite number `> 0` | Marks an individual preprocessed channel as signal when it lies more than this many noise sigmas from the histogram center. Larger values require stronger channel excursions. |
| `min_signal_channels` | `4` | integer `>= 1` | Requires a connected candidate run to represent at least this many **original** channels. Larger values reject narrower features. |
| `segment_mean_sigma` | `3.0` | finite number `> 0` | Requires the mean of an entire candidate run in the preprocessed profile to differ from the noise center by this many sigmas. Larger values reject lower-average-S/N runs. |
| `split_mean_sigma` | `4.0` | finite number `> 0` | Searches an accepted segment for multiple components only when its absolute mean S/N reaches this value. It operates on the preprocessed profile; larger values split only stronger segments. |
| `split_min_channels` | `12` | integer `>= 1` | Searches for multiple components only when a segment contains at least this many **preprocessed** channels. Larger values leave more segments unsplit. |
| `split_window_channels` | `5` | odd integer `>= 3` | Width, in preprocessed channels, of the local-extrema window used to divide blended features. Larger windows look for broader turning points. |
| `initial_fwhm_fraction` | `0.5` | finite number `> 0` | Initializes component FWHM to this fraction of its accepted segment width on the original spectral axis, subject to a one-channel floor. Larger values start broader. |
| `min_fwhm_channels` | `0.25` | finite number `> 0` | Lower final-fit FWHM bound in units of the typical original-channel width. Larger values prevent narrower fitted components. |
| `max_fwhm_axis_spans` | `2.0` | finite number `> 0` | Upper final-fit FWHM bound in units of the complete spectral-axis span. Larger values permit broader fitted components. |
| `fit_maxfev` | `50000` | integer `>= 1` | Maximum model evaluations for the final simultaneous baseline-plus-Gaussian fit. Larger values may improve convergence but take longer. |

The selected lower FWHM bound must not exceed the selected upper bound for the
spectral axis being fitted. Invalid configuration values raise field-specific
`ValueError` exceptions.

## Cube Fitting

```python
from spectral_cube import SpectralCube
from simplefit import fit_cube

cube = SpectralCube.read("cube.fits")
cube_fit = fit_cube(cube, n_jobs=4, chunk_size=256, progress=True)
```

The same configuration can be used for every spectrum and SSA in a cube,
including multiprocessing workers:

```python
from simplefit import FitConfig, fit_cube

config = FitConfig(signal_sigma=3.0, min_signal_channels=5)
cube_fit = fit_cube(cube, n_jobs=4, config=config)
```

`fit_cube` returns a `FitCubeResult` with:

- `component_table`: sparse table with one row per fitted component.
- `n_components`: 2D map of fitted component counts.
- `success`: 2D boolean map.
- `noise`: 2D map of per-spectrum noise estimates.
- `message`: 2D map of fit messages.
- `model_cube` and `residual_cube`: optional full cubes when `store_models=True`.

The sparse component table is useful because each pixel can have a different
number of Gaussian components.

```python
cube_fit.component_table[:10]
cube_fit.component_map("center", component=1)
cube_fit.write_table("/private/tmp/cube_fit_components.csv")
```

The most useful cube-fitting options are:

- `mask`: optional 2D boolean spatial mask. Only `True` pixels are fit.
- `n_jobs`: number of worker processes. Use `1` for serial execution.
- `chunk_size`: number of spectra, or SSAs, bundled into one worker task. Larger
  chunks reduce scheduling overhead; smaller chunks update progress more often.
  If omitted, `simplefit` chooses a conservative automatic size from the number
  of selected pixels and `n_jobs`.
- `progress`: show fitting progress. `tqdm` is used when installed; otherwise a
  lightweight stderr progress indicator is used.
- `max_components`: cap the number of Gaussian components detected per spectrum
  or SSA.
- `fit_baseline`: enable or disable the CARTA-like baseline heuristic.
- `config`: reuse a `FitConfig` for identical detection and optimizer controls
  across every spectrum, SSA mean, and multiprocessing worker.
- `store_models`: include full model and residual cubes in the result. Leave
  this off for large cubes unless you need those arrays.

`n_jobs` and `chunk_size` are independent tuning knobs. `n_jobs` controls how
many worker processes run concurrently, while `chunk_size` controls how many
spectra each submitted task contains. With `n_jobs > 1`, progress advances when a
chunk finishes, so a large chunk size can make the progress bar update less
frequently even though work is running.

## SSA-Guided Cube Fitting

For faster cube fitting, use mask-aware spectral averaging areas (SSAs). In this
mode, `simplefit` runs the expensive component auto-detection once on each
averaged SSA spectrum, then fits each member pixel from that SSA guess.

```python
import numpy as np

mask = np.zeros(cube.shape[1:], dtype=bool)
mask[10:20, 10:20] = True

cube_fit = fit_cube(cube, mask=mask, ssa_size=3, n_jobs=2)
```

With `ssa_size=3`, local 3x3 candidate boxes are built over the selected mask.
Only `mask=True` pixels inside each box are averaged and fit. `ssa_size` can also
be a `(y_size, x_size)` tuple. Use `ssa_min_pixels` to require a minimum number
of masked pixels in each SSA box.

SSA-guided results include:

- `ssa_labels`: 2D map identifying the SSA assigned to each fitted pixel.
- `ssa_table`: one row per SSA with bounds, pixel count, fit success, component
  count, noise, and message.
- `ssa_id` in `component_table`, linking each pixel component back to its SSA.

Chunking is used in both SSA stages when `n_jobs > 1`: fitting the averaged SSA
spectra, then fitting member pixels from each successful SSA guess.

## Examples

The repository includes small notebooks and PHANGS-oriented scripts:

- `examples/fitting_example.ipynb`: one-dimensional spectrum fitting.
- `examples/fitting_example_cube.ipynb`: compact cube fitting with SSA labels and
  component maps.
- `examples/fitting_example_cube_bigger.ipynb`: larger cube example using
  `n_jobs`, `chunk_size`, table export, and Plotly component visualization.
- `examples/fitting_phangs_hi.py`: batch PHANGS MeerKAT HI cube fitting. The
  script discovers `*_meerkat_hi21cm*.fits` cubes, pairs each with a matching
  `*_broad_mom0.fits` mask, fits each galaxy with the top-level `N_JOBS`,
  `CHUNK_SIZE`, and `SSA_SIZE` settings, then writes per-galaxy CSV and Plotly
  HTML outputs.
- `examples/fitting_post_phangs_hi.py`: post-processes `*_fits.csv` tables,
  keeps successful positive components with finite centers/FWHMs and amplitude
  signal-to-noise above the script threshold, then writes pruned CSV and Plotly
  HTML outputs.

For the PHANGS scripts, edit `INPUT_DIR` and `OUTPUT_DIR` near the top of the
file before running. The fitting script prepends the local `src/` directory to
`sys.path` so a checkout run from `examples/` uses the repository code rather
than an older installed package.

## Notes

- The package is currently intended for local installation and notebook use.
- The fitter currently supports Gaussian components only.
- The original CARTA TypeScript heuristic source is preserved in
  `references/carta_fitting_heuristics.ts`.
