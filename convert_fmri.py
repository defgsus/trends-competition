import os

# deactivate multithreading for numpy and co, so we can efficiently use multiprocessing.Pool from conversion
# see https://stackoverflow.com/questions/15414027/multiprocessing-pool-makes-numpy-matrix-multiplication-slower
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import pandas as pd
import numpy as np

from src import *
from src import FMRIConverterCSV


def convert_fmri():
    CONVERSIONS = (
        "1time",
        #"halftime",
        # "1time-diff",
        #"sum",
    )
    REDUCTIONS = (
        "bucket-6th",
        #"center-slice",
    )
    for feature_type in (
            "train",
            "test",
    ):
        for convert_mode in CONVERSIONS:
            for reduction_mode in REDUCTIONS:
                output_filename = f"fmri-{feature_type}-{convert_mode}-{reduction_mode}"
                if os.path.exists(os.path.join(DATA_PROC_DIR, f"{output_filename}.csv")):
                    printe(f"skipping {output_filename}")
                    continue

                converter = FMRIConverterCSV(
                    source_path=os.path.join(DATA_ORG_DIR, f"fMRI_{feature_type}"),
                    output_filename=output_filename,
                    convert_mode=convert_mode,
                    reduction_mode=reduction_mode,
                    #processing_mode="conv",
                )

                printe(f"converting {len(converter.source_filenames)} files to {output_filename}")

                converter.run_conversion(num_processes=None)

                del converter


def convert_fmri_activation():
    PROCESSES = (
        "cut-slice-z",
        "act-series",
    )
    for feature_type in (
            "train",
            #"test",
    ):
        for process_mode in PROCESSES:
            output_filename = f"fmri-{feature_type}-{process_mode}"
            if os.path.exists(os.path.join(DATA_PROC_DIR, f"{output_filename}.csv")):
                printe(f"skipping {output_filename}")
                continue

            converter = FMRIConverterActivationCSV(
                source_path=os.path.join(DATA_ORG_DIR, f"fMRI_{feature_type}"),
                output_filename=output_filename,
                processing_mode=process_mode,
            )

            printe(f"converting {len(converter.source_filenames)} files to {output_filename}")

            converter.run_conversion(num_processes=None)

            del converter


if __name__ == "__main__":
    convert_fmri()
    #convert_fmri_activation()
