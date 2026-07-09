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

## Cube Fitting

```python
from spectral_cube import SpectralCube
from simplefit import fit_cube

cube = SpectralCube.read("cube.fits")
cube_fit = fit_cube(cube, n_jobs=4, chunk_size=256, progress=True)
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
