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

## Single-Spectrum Fitting

```python
import numpy as np
from simplefit import fit_spectrum, plot_fit

fit = fit_spectrum(x_data=spectrum, x_axis=velocity)

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

## Cube Fitting

```python
from spectral_cube import SpectralCube
from simplefit import fit_cube

cube = SpectralCube.read("cube.fits")
cube_fit = fit_cube(cube, n_jobs=2)
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
Only `mask=True` pixels inside each box are averaged and fit.

SSA-guided results include:

- `ssa_labels`: 2D map identifying the SSA assigned to each fitted pixel.
- `ssa_table`: one row per SSA with bounds, pixel count, fit success, component
  count, noise, and message.
- `ssa_id` in `component_table`, linking each pixel component back to its SSA.

## Notes

- The package is currently intended for local installation and notebook use.
- The fitter currently supports Gaussian components only.
- The original CARTA TypeScript heuristic source is preserved in
  `references/carta_fitting_heuristics.ts`.
