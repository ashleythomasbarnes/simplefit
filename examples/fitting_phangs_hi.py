import sys
from pathlib import Path

import a.pyplot as plt
import numpy as np
from astropy import constants as const
from astropy import units as u
from astropy.io import fits
from spectral_cube import SpectralCube

from simplefit import fit_cube, fit_spectrum, plot_fit

import numpy as np
import plotly.express as px

# FILES
INPUT_FILE = "./example_data/ngc1512_meerkat_hi21cm_pbcorr_zoom_15arcsec_k.fits"
INPUT_MASK_FILE = "./example_data/ngc1512_meerkat_hi21cm_pbcorr_zoom_15arcsec_k_broad_mom0.fits"
OUTPUT_TABLE = Path("./example_outputs/cube_fit_components_bigger.csv")
OUTPUT_FIGURE = Path("./example_outputs/cube_fit_example_bigger.png")

#LOADING
cube = SpectralCube.read(INPUT_FILE)
cube.allow_huge_operations = True
cube = cube.with_spectral_unit(u.km / u.s, velocity_convention="radio")
mask = fits.getdata(INPUT_MASK_FILE).astype(bool)

# FITTING
cube_fit = fit_cube(cube, n_jobs=15, progress=True, mask=mask, ssa_size=10)
cube_fit.write_table(OUTPUT_TABLE)

# PLOTTING
cube_fit.component_table["log(amplitude)"] = np.log10(cube_fit.component_table["amplitude"])
plot_components = cube_fit.component_table
plot_components = plot_components[
    plot_components["success"]
    & (plot_components["amplitude"] > 0.0)
    & np.isfinite(plot_components["center"])
    & np.isfinite(plot_components["fwhm"]) 
    & (plot_components["fwhm"] < 100.0)
]

plot_df = plot_components.to_pandas()

fig = px.scatter_3d(
    plot_df,
    x="x",
    y="y",
    z="center",
    color="log(amplitude)",      # point colour = intensity
    size="fwhm",            # point size = FWHM
    color_continuous_scale="magma",
    range_color=[0, 2],
    size_max=12,
    title="Fitted Gaussian Components in Position-Position-Velocity Space",
)

fig.update_traces(
    marker={
        "opacity": 0.2,
        "line": {"width": 0},
    }
)

fig.update_layout(
    scene={
        "xaxis_title": "x pixel",
        "yaxis_title": "y pixel",
        "zaxis_title": "center velocity",
        "yaxis": {"autorange": "reversed"},
    },
    margin={"l": 0, "r": 0, "b": 0, "t": 45},
)

fig.show()
fig.write_html(OUTPUT_FIGURE, auto_open=True)