from __future__ import annotations

from pathlib import Path

import numpy as np
import plotly.express as px
from astropy.table import Table


# OUTPUT_DIR = Path("/lustre/opsw/work/abarnes/phangs/HI_WORK/fitting/")

OUTPUT_DIR = Path("/Users/abarnes/Dropbox/Data/Extragalactic/misc/hi_fitting/")

FWHM_MAX = 100.0
AMPLITUDE_SN_MIN = 3.0
COLOR_PERCENTILES = (1.0, 99.0)


def display_path(path: Path) -> str:
    """Return a compact path for logging without assuming a repo-relative path."""

    path = path.resolve()
    for root in (Path.cwd().resolve(), OUTPUT_DIR.resolve()):
        if path == root:
            continue
        try:
            return str(path.relative_to(root))
        except ValueError:
            continue
    return str(path)


def iter_fit_tables() -> list[Path]:
    """Find per-galaxy fit tables, excluding already pruned outputs."""

    return sorted(
        path
        for path in OUTPUT_DIR.glob("*_fits.csv")
        if not path.name.endswith("_fits_pruned.csv")
    )


def error_column_name(table: Table) -> str:
    """Return the amplitude uncertainty column name used by this table."""

    for name in ("amplitude_error", "amplitude_err"):
        if name in table.colnames:
            return name
    raise ValueError(
        "No amplitude uncertainty column found. Expected 'amplitude_error' or 'amplitude_err'."
    )


def success_mask(table: Table) -> np.ndarray:
    """Return a boolean success mask for bool or string-valued success columns."""

    success = np.asarray(table["success"])
    if success.dtype == bool:
        return success

    return np.char.lower(success.astype(str)) == "true"


def prune_components(table: Table) -> Table:
    amp_err_name = error_column_name(table)
    amplitude = np.asarray(table["amplitude"], dtype=float)
    amplitude_error = np.asarray(table[amp_err_name], dtype=float)
    center = np.asarray(table["center"], dtype=float)
    fwhm = np.asarray(table["fwhm"], dtype=float)

    keep = (
        success_mask(table)
        & (amplitude > 0.0)
        & np.isfinite(center)
        & np.isfinite(fwhm)
        & (fwhm < FWHM_MAX)
        & np.isfinite(amplitude_error)
        & (amplitude_error > 0.0)
        & (amplitude / amplitude_error > AMPLITUDE_SN_MIN)
    )
    return table[keep]


def color_range(values: np.ndarray) -> list[float] | None:
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return None

    vmin, vmax = np.nanpercentile(finite_values, COLOR_PERCENTILES)
    if np.isclose(vmin, vmax):
        padding = max(abs(vmin) * 0.01, 0.01)
        vmin -= padding
        vmax += padding
    return [float(vmin), float(vmax)]


def make_component_plot(component_table: Table, output_figure: Path) -> None:
    plot_components = component_table.copy()
    plot_components["log(amplitude)"] = np.log10(plot_components["amplitude"])
    plot_df = plot_components.to_pandas()
    range_color = color_range(np.asarray(plot_components["log(amplitude)"], dtype=float))

    fig = px.scatter_3d(
        plot_df,
        x="x",
        y="y",
        z="center",
        color="log(amplitude)",
        size="fwhm",
        color_continuous_scale="magma",
        range_color=range_color,
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

    fig.write_html(output_figure, auto_open=False)


def post_process_one_table(input_table: Path) -> None:
    galaxy = input_table.name.removesuffix("_fits.csv")
    output_table = input_table.with_name(f"{galaxy}_fits_pruned.csv")
    output_figure = input_table.with_name(f"{galaxy}_fits_pruned.html")

    print(f"Post-processing {galaxy}:")
    print(f"  input table: {display_path(input_table)}")
    print(f"  output table: {display_path(output_table)}")
    print(f"  output figure: {display_path(output_figure)}")

    table = Table.read(input_table, format="csv")
    pruned_table = prune_components(table)

    print(f"  keeping {len(pruned_table)} of {len(table)} components")
    pruned_table.write(output_table, format="csv", overwrite=True)

    if len(pruned_table) == 0:
        print("  skipping figure: no components remain after pruning")
        return

    make_component_plot(pruned_table, output_figure)


def main() -> None:
    tables = iter_fit_tables()
    if not tables:
        raise FileNotFoundError(f"No *_fits.csv tables found in {display_path(OUTPUT_DIR)}.")

    for table in tables:
        post_process_one_table(table)


if __name__ == "__main__":
    main()
