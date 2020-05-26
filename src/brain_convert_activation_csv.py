import os
import sys
from multiprocessing import Pool
import csv

import tqdm
import h5py
import numpy as np
import pandas as pd
from scipy.ndimage import zoom
from scipy.ndimage.filters import convolve
import nibabel
from nilearn.plotting import find_cut_slices
from nilearn.image import index_img, get_data
from nilearn.input_data import NiftiSpheresMasker

from . import printe, DATA_ORG_DIR, DATA_PROC_DIR


class FMRIConverterActivationCSV:

    def __init__(self, source_path=None, destination_path=None, output_filename=None,
                 processing_mode="cut-slice-z", reduction_mode=None):
        """
        Test for measuring activity data via nilearn tools
        :param source_path:
        :param destination_path:
        :param output_filename:
        :param processing_mode: str
            "cut-slice-z": series of nilearn.plotting.find_cut_slices along each layer
            "act-series": activation series in some hand-picked regions
        :param reduction_mode: str
        """
        from .Data import Data
        self.source_path = source_path or DATA_ORG_DIR
        self.destination_path = destination_path or DATA_PROC_DIR
        self.source_filenames = self._search_files()
        self.output_filename = output_filename
        self.reduction_mode = reduction_mode
        self.processing_mode = processing_mode
        self._data = Data()

    def _search_files(self):
        filenames = []
        for root, dirs, files in os.walk(self.source_path):
            for file in files:
                if file.endswith(".mat"):
                    filenames.append((
                        file[:-4],
                        os.path.join(root, file),
                    ))
        return sorted(filenames)

    def load_mat(self, filename):
        file = h5py.File(filename, "r")

        data = file["SM_feature"][()]
        file.close()
        # convert to nifti's x, y, z, t
        data = np.transpose(data, (3, 2, 1, 0))
        img = nibabel.Nifti1Image(data, self._data.fmri_mask.affine)
        return img

    def run_conversion(self, num_processes=None):
        pool = Pool(num_processes)
        results = pool.map(self._convert_file, self.source_filenames)

        if not os.path.exists(self.destination_path):
            os.makedirs(self.destination_path)

        output_filename = self.output_filename or f"fmri-{self.processing_mode}"
        self._store_csv(
            os.path.join(self.destination_path, f"{output_filename}.csv"),
            results
        )
        #pd.DataFrame(flat_brains, index=[f[0] for f in self.source_filenames])

    def _store_csv(self, filename, flat_brains):
        printe(f"storing {filename}")
        with open(filename, "wt") as fp:
            writer = csv.writer(fp)
            writer.writerow(["Id"] + list(range(len(flat_brains[0]))))
            for flat_brain in tqdm.tqdm(flat_brains):
                writer.writerow(flat_brain)

    def _convert_file(self, filename):
        printe(f"converting {filename[0]}")

        fmri = self.load_mat(filename[1])

        flat_brain = self._process_fmri(fmri)

        flat_brain = np.round(flat_brain, 6)
        # add ID and convert to normal list to save process-transmission space
        flat_brain = [filename[0]] + list(flat_brain)
        print(flat_brain)
        return flat_brain

    def _process_fmri(self, fmri):
        if self.processing_mode.startswith("cut-slice-"):
            data = [
                find_cut_slices(index_img(fmri, i), n_cuts=1)[0]
                for i in range(fmri.shape[-1])
            ]
            return data

        if self.processing_mode == "act-series":
            series = build_series(fmri)
            return series.reshape(-1)

        else:
            raise ValueError(f"Invalid convert_mode {self.processing_mode}")


def build_series(brain4d):
    coords = [
        (30, -45, -10),
        (0, -85, 10),
        (0, 47, 30),
        (-35, -20, 50),
    ]
    masker = NiftiSpheresMasker(
        coords,
        radius=5,
        detrend=True, standardize=True,
        low_pass=0.1, high_pass=0.01, t_r=2,
    )

    series = masker.fit_transform(brain4d)
    return series