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

from . import printe, DATA_ORG_DIR, DATA_PROC_DIR


class FMRIConverterCSV:

    convolution_weights = [
        [
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0],
        ],
        [
            [0, 1, 0],
            [1, -6, 1],
            [0, 1, 0],
        ],
        [
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0],
        ],
    ]

    def __init__(self, source_path=None, destination_path=None, output_filename=None,
                 convert_mode="1time", processing_mode=None, reduction_mode=None):
        """
        This class converts a directory of .mat files to a single CSV
            by reducing the fmri data in some way

        :param source_path: str, where to search for .mat files
        :param destination_path: str, where to store csv file
        :param output_filename: str, name of csv file
        :param convert_mode: str, how to sample a brain volume from fmri 4d data
            "1time": first time-slice
            "halftime": middle time-slice
            "sum": sum of all time slices
            "1time-diff": first minus second time-slice
        :param processing_mode: str, some additional processing on the volume
            "conv": do 'some' convolution on the brain volume
        :param reduction_mode: str, how to reduce brain volume to csv row
            "bucket-Xth": reduce by summing into N buckets, where N is num_voxels / X
            "center-slice": take the whole z center slice
        """
        from .Data import Data
        self.source_path = source_path or DATA_ORG_DIR
        self.destination_path = destination_path or DATA_PROC_DIR
        self.source_filenames = self._search_files()
        self.output_filename = output_filename
        self.convert_mode = convert_mode
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
        """
        Load mat file as h5py data
        """
        file = h5py.File(filename, "r")

        feature = file["SM_feature"][()]
        file.close()
        return feature

    def run_conversion(self, num_processes=None):
        """
        Run conversion of
        :param num_processes:
        :return:
        """
        pool = Pool(num_processes)
        flat_brains = pool.map(self._convert_file, self.source_filenames)

        if not os.path.exists(self.destination_path):
            os.makedirs(self.destination_path)

        output_filename = self.output_filename or f"fmri-{self.convert_mode}"
        self._store_csv(
            os.path.join(self.destination_path, f"{output_filename}.csv"),
            flat_brains
        )

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

        brain = self._convert_fmri(fmri)
        if self.processing_mode:
            brain = self._process(brain)

        self.print_slice(brain, brain.shape[0] // 2)

        flat_brain = self._reduce_brain(brain)

        flat_brain = np.round(flat_brain, 6)
        # add ID and convert to normal list to save process-transmission space
        flat_brain = [filename[0]] + list(flat_brain)
        return flat_brain

    def _convert_fmri(self, fmri):
        if self.convert_mode == "sum":
            brain = np.sum(fmri, axis=0)
        elif self.convert_mode == "1time":
            brain = fmri[0]
        elif self.convert_mode == "halftime":
            brain = fmri[fmri.shape[0] // 2]
        elif self.convert_mode == "1time-diff":
            brain = fmri[0] - fmri[1]
        else:
            raise ValueError(f"Invalid convert_mode {self.convert_mode}")
        return brain

    def _reduce_brain(self, brain):
        if self.reduction_mode is None:
            flat_brain = brain.reshape(-1)

        elif self.reduction_mode == "center-slice":
            flat_brain = brain[brain.shape[0]//2].reshape(-1)

        elif self.reduction_mode.startswith("bucket-"):
            factor = int(self.reduction_mode[7])
            if 1:
                W, H, D = brain.shape
                SW, SH, SD = W//factor, H//factor, D//factor

                flat_brain = np.zeros(SW*SH*SD, dtype="f8")
                for z in range(D):
                    sz = z // factor
                    if sz < SD:
                        for y in range(H):
                            sy = y // factor
                            if sy < SH:
                                for x in range(W):
                                    sx = x // factor
                                    if sx < SW:
                                        flat_brain[((sz * SH + sy) * SW) + sx] += brain[z][y][x]
        else:
            raise ValueError(f"Invalid reduction_mode {self.convert_mode}")
        return flat_brain

    def _process(self, brain):
        if self.processing_mode == "conv":
            brain = convolve(brain, self.convolution_weights)
        else:
            raise ValueError(f"Invalid processing_mode {self.processing_mode}")
        return brain

    @staticmethod
    def print_slice(brain, idx):
        chars = " .,:;+*#"
        numchars = len(chars)

        mi, ma = np.min(brain), np.max(brain)

        slice = brain[idx]

        for y in range(slice.shape[0]):
            print("".join(
                chars[min(numchars-1, int((slice[y][x]-mi)/(ma-mi)*numchars))]
                for x in range(slice.shape[1])
            ))
