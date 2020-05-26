import os

import pandas as pd
import numpy as np
import nibabel
import h5py

from . import printe, DATA_ORG_DIR, DATA_ADD_DIR, DATA_PROC_DIR


class Data:
    """
    Wrapper for all competition and converted data
    """

    def __init__(self):
        self._fnc = None
        self._smb = None
        self._fmri = dict()
        self._fmri_train = dict()
        self._fmri_test = dict()
        self._train = None
        self._train_ids = None
        self._test_ids = None
        self._fmri_mask = None
        self._desikan = None
        self._desikan_indices = None

    @property
    def connectivity(self):
        if self._fnc is None:
            self._fnc = pd.read_csv(os.path.join(DATA_ORG_DIR, "fnc.csv"), index_col="Id")
        return self._fnc

    @property
    def morphology(self):
        if self._smb is None:
            self._smb = pd.read_csv(os.path.join(DATA_ORG_DIR, "loading.csv"), index_col="Id")
        return self._smb

    def fmri(self, type):
        if type not in self._fmri:
            df1 = self.fmri_train(type)
            df2 = self.fmri_test(type)
            df = df1.append(df2)
            self._fmri[type] = df
        return self._fmri[type]

    def fmri_train(self, type):
        if type not in self._fmri_train:
            self._fmri_train[type] = pd.read_csv(os.path.join(DATA_PROC_DIR, f"fmri-train-{type}.csv"), index_col="Id")
            # TODO: sometimes last row is NaN
            del self._fmri_train[type][self._fmri_train[type].columns[-1]]
        return self._fmri_train[type]

    def fmri_test(self, type):
        if type not in self._fmri_test:
            self._fmri_test[type] = pd.read_csv(os.path.join(DATA_PROC_DIR, f"fmri-test-{type}.csv"), index_col="Id")
            del self._fmri_test[type][self._fmri_test[type].columns[-1]]
        return self._fmri_test[type]

    @property
    def training(self):
        if self._train is None:
            self._train = pd.read_csv(os.path.join(DATA_ORG_DIR, "train_scores.csv"), index_col="Id")
        return self._train

    @property
    def training_ids(self):
        if self._train_ids is None:
            self._train_ids = list(self.training.index)
        return self._train_ids

    @property
    def test_ids(self):
        if self._test_ids is None:
            self._test_ids = sorted(set(self.morphology.index) - set(self.training_ids))
        return self._test_ids

    @property
    def fmri_mask(self):
        """
        The fmri mask .nii file as nibabel image
        :return: Nifti1Image instance
        """
        if self._fmri_mask is None:
            self._fmri_mask = nibabel.load(os.path.join(DATA_ORG_DIR, "fMRI_mask.nii"))
        return self._fmri_mask

    def load_fmri_image(self, subject_id):
        """
        Load one of the fMRI_train/test .mat files as nibabel image
        :param subject_id: str
        :return: Nifti1Image instance
        """
        h5f = None
        for subdir in ("train", "test"):
            try:
                h5f = h5py.File(
                    os.path.join(DATA_ORG_DIR, f"fMRI_{subdir}", f"{subject_id}.mat"),
                    "r"
                )
            except IOError:
                pass
        if not h5f:
            raise IOError(f"Could not load subject {subject_id}")

        data = h5f["SM_feature"]
        # convert to nifti's x, y, z, t
        data = np.transpose(data, (3, 2, 1, 0))
        img = nibabel.Nifti1Image(data, self.fmri_mask.affine)
        return img
