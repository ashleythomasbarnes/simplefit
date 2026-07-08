from __future__ import annotations

from os.path import commonprefix
from pathlib import Path
import sys


REPO_SRC = Path(__file__).resolve().parents[1] / "src"
if REPO_SRC.exists():
    sys.path.insert(0, str(REPO_SRC))

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from astropy.io import fits
from spectral_cube import SpectralCube

from simplefit import fit_cube


INPUT_DIR = Path("/lustre/opsw/work/abarnes/phangs/HI_WORK/Archive/HI/MeerKAT/v0p1/")
OUTPUT_DIR = Path("/lustre/opsw/work/abarnes/phangs/HI_WORK/fitting/")

HI_MARKER = "_meerkat_hi21cm"
MASK_SUFFIX = "_broad_mom0"
N_JOBS = 15
SSA_SIZE = 0


def display_path(path: Path) -> str:
    """Return a compact path for logging without assuming a repo-relative path."""

    path = path.resolve()
    for root in (Path.cwd().resolve(), INPUT_DIR.resolve(), OUTPUT_DIR.resolve()):
        if path == root:
            continue
        try:
            return str(path.relative_to(root))
        except ValueError:
            continue
    return str(path)


def galaxy_name(path: Path) -> str:
    """Return the galaxy prefix before the MeerKAT HI filename marker."""

    if HI_MARKER not in path.stem:
        raise ValueError(f"Could not parse galaxy name from {path.name!r}.")
    return path.stem.split(HI_MARKER, maxsplit=1)[0]


def find_mask_file(cube_file: Path, galaxy: str) -> Path | None:
    """Find the matching broad moment-0 mask for one HI cube."""

    exact_mask = cube_file.with_name(f"{cube_file.stem}{MASK_SUFFIX}.fits")
    if exact_mask.exists():
        return exact_mask

    mask_files = sorted(INPUT_DIR.glob(f"{galaxy}{HI_MARKER}*{MASK_SUFFIX}.fits"))
    if not mask_files:
        return None
    if len(mask_files) == 1:
        return mask_files[0]

    return max(mask_files, key=lambda path: len(commonprefix([cube_file.stem, path.stem])))


def iter_input_pairs() -> list[tuple[str, Path, Path]]:
    """Pair each MeerKAT HI cube with its broad moment-0 mask."""

    cube_files = sorted(
        path
        for path in INPUT_DIR.glob(f"*{HI_MARKER}*.fits")
        if MASK_SUFFIX not in path.stem
    )
    pairs = []

    for cube_file in cube_files:
        galaxy = galaxy_name(cube_file)
        mask_file = find_mask_file(cube_file, galaxy)
        if mask_file is None:
            print(f"Skipping {cube_file.name}: no matching {MASK_SUFFIX} mask found.")
            continue
        pairs.append((galaxy, cube_file, mask_file))

    return pairs


def load_cube_and_mask(cube_file: Path, mask_file: Path) -> tuple[SpectralCube, np.ndarray]:
    cube = SpectralCube.read(cube_file)
    cube.allow_huge_operations = True
    cube = cube.with_spectral_unit(u.km / u.s, velocity_convention="radio")
    mask_data = fits.getdata(mask_file)
    mask = np.isfinite(mask_data) & (mask_data != 0)
    return cube, mask


def make_component_plot(cube_fit, output_figure: Path) -> None:
    cube_fit.component_table["log(amplitude)"] = np.log10(
        cube_fit.component_table["amplitude"]
    )
    plot_components = cube_fit.component_table
    plot_components = plot_components[
        plot_components["success"]
        & (plot_components["amplitude"] > 0.0)
        & np.isfinite(plot_components["center"])
        & np.isfinite(plot_components["fwhm"])
        & (plot_components["fwhm"] < 100.0)
    ]
    plot_df = plot_components.to_pandas()

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(projection="3d")
    points = ax.scatter(
        plot_df["x"],
        plot_df["y"],
        plot_df["center"],
        c=plot_df["log(amplitude)"],
        s=np.clip(plot_df["fwhm"], 1.0, 100.0),
        cmap="magma",
        vmin=0,
        vmax=2,
        alpha=0.2,
        linewidths=0,
    )

    ax.set_title("Fitted Gaussian Components in Position-Position-Velocity Space")
    ax.set_xlabel("x pixel")
    ax.set_ylabel("y pixel")
    ax.set_zlabel("center velocity")
    ax.invert_yaxis()
    fig.colorbar(points, ax=ax, label="log(amplitude)", pad=0.12)
    fig.tight_layout()
    fig.savefig(output_figure, dpi=200)
    plt.close(fig)


def fit_one_galaxy(galaxy: str, cube_file: Path, mask_file: Path) -> None:
    output_table = OUTPUT_DIR / f"{galaxy}_fits.csv"
    output_figure = OUTPUT_DIR / f"{galaxy}_fits.png"

    print(f"Fitting {galaxy}:")
    print(f"  cube: {display_path(cube_file)}")
    print(f"  mask: {display_path(mask_file)}")
    print(f"  table: {display_path(output_table)}")
    print(f"  figure: {display_path(output_figure)}")

    cube, mask = load_cube_and_mask(cube_file, mask_file)
    ssa_size = SSA_SIZE if SSA_SIZE else None
    cube_fit = fit_cube(cube, n_jobs=N_JOBS, progress=True, mask=mask, ssa_size=ssa_size)
    cube_fit.write_table(output_table)
    make_component_plot(cube_fit, output_figure)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pairs = iter_input_pairs()
    if not pairs:
        raise FileNotFoundError(
            f"No MeerKAT HI cube/mask pairs found in {display_path(INPUT_DIR)}."
        )

    for galaxy, cube_file, mask_file in pairs:
        fit_one_galaxy(galaxy, cube_file, mask_file)


if __name__ == "__main__":
    main()
