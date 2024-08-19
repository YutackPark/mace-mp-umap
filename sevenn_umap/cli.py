import pathlib
import typing as t
import typer
from typing_extensions import Annotated
app = typer.Typer()
import warnings
from enum import Enum

import torch

from mace_mp_umap.analysis import find_closest_training_points
from mace_mp_umap.dim_reduction import (
    apply_dimensionality_reduction,
    fit_dimensionality_reduction,
)
from mace_mp_umap.plotting import plot_dimensionality_reduction

from sevenn.sevennet_calculator import SevenNetCalculator
from .chemiscope_handling import write_chemiscope_input
from .data_manipulations import get_cleaned_dataframe
from .utils import patch_calc_for_descriptor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class FilterType(str, Enum):
    exclusive = "exclusive"
    inclusive = "inclusive"
    combinations = "combinations"
    none = "none"


@app.command()
def produce_mace_chemiscope_input(
    model_cp: str = typer.Argument(
        default="7net-0",
        help="Path to sevennet checkpoint or model kwards"
    ),
    data_path: str = typer.Argument(
        default=None,
        help="Path to XYZ file containing your system",
    ),
    mp_data_path: str = typer.Argument(default=None, help="Path to MP data"),
    filtering: FilterType = typer.Option(
        default=FilterType.none,
        case_sensitive=False,
        help="Whether to filter out structures that contain elements not in the subset or to include them.",
    ),
    element_subset: Annotated[
        t.List[str],
        typer.Option(
            "--add-element", "-e", help="List of elements to include in the subset."
        ),
    ] = [],
    create_plots: bool = typer.Option(
        default=False, help="Whether to create static UMAP and PCA plots."
    ),
):
    if DEVICE != "cuda":
        warnings.warn("CUDA not available, using CPU. Might be slow.")

    if filtering == FilterType.none:
        raise ValueError(
            "You must specify filtering type (either `--filtering exclusive` or `--filtering inclusive`).\n"
            "Combinations mode means that structures are kept if they're composed only of elements supplied via `-e` flags but don't need to contail all of the supplied elements.\n"
            "Exclusive mode means those and only those structures are kept that contail all elements supplied via `-e` flags. This is a subset of `combinations`\n"
            "Inclusive mode means that other elements are allowed in addition to those supplied via `-e` flags.\n"
            "Most applications should use `--filtering inclusive`. However, for elemental compounds or molecular compounds like water `exclusive` or `combinations` modes are more appropriate."
        )

    # Load model
    calc = SevenNetCalculator(
        model=model_cp,
        device=DEVICE,
    )
    calc = patch_calc_for_descriptor(calc)

    cutoff = calc.cutoff
    print(
        f"Using the SevenNet cutoff ({cutoff} Angstrom) for neighbour analysis for all elements."
    )

    # Load MP data
    print("For MP")
    train_atoms, training_data_df = get_cleaned_dataframe(
        mp_data_path, calc, element_subset, cutoff, filtering_type=filtering
    )
    # Load test data
    print("For Test")
    test_atoms, test_data_df = get_cleaned_dataframe(
        data_path, calc, element_subset, cutoff, filtering_type="none"
    )
    if len(test_data_df) == 0 or len(training_data_df) == 0:
        raise ValueError(
            f"No structures found in {data_path} or {mp_data_path}. Check your filtering settings."
        )

    element_subset_str = "".join(element_subset)
    system_name = f"{pathlib.Path(data_path).stem}_{filtering}_{element_subset_str}"

    print(f"Will use {system_name} for naming output files.")

    tag = "readout"
    sli = slice(None)
    umap_reducer, pca_reducer = fit_dimensionality_reduction(
        training_data_df, tag, sli,
    )
    if create_plots:
        apply_dimensionality_reduction(
            test_data_df, tag, sli, umap_reducer, pca_reducer
        )
        figure = plot_dimensionality_reduction(
            training_data_df, test_data_df, 1,
        )
        figure.savefig(f"{system_name}_dimensionality_reduction.pdf")

    results_df = find_closest_training_points(training_data_df, test_data_df)
    results_df.to_csv(f"{system_name}_closest_training_points.csv", index=False)

    # To fit original code, see get_reduced_embeddings
    reducers = [None, (umap_reducer, pca_reducer)]
    # Produce chemiscope input file
    write_chemiscope_input(train_atoms, test_atoms, reducers, system_name)


if __name__ == "__main__":
    app()
