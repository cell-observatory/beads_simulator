from operator import floordiv
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(scope="session")
def kargs():
    repo = Path.cwd()
    num_modes = 15
    digital_rotations = 361

    kargs = dict(
        repo=repo,
        embeddings_shape=(6, 64, 64, 1),
        digital_rotations=digital_rotations,
        rotations_shape=(digital_rotations, 6, 64, 64, 1),
        num_modes=num_modes,
        psf_type=repo/'lattice/YuMB_NAlattice0p35_NAAnnulusMax0p40_NAsigma0p1.mat',
        prediction_filename_pattern=r"*[!_gt|!_realspace|!_noisefree|!_predictions_psf|!_corrected_psf|!_reconstructed_psf].tif",
        prev=None,
        dm_state=None,
        wavelength=.510,
        dm_damping_scalar=1.0,
        lateral_voxel_size=.097,
        axial_voxel_size=.2,
        freq_strength_threshold=.01,
        prediction_threshold=0.,
        confidence_threshold=0.02,
        num_predictions=1,
        batch_size=64,
        plot=True,
        plot_rotations=True,
        ignore_modes=[0, 1, 2, 4],
        # limit the number of cpu workers to hopefully avoid "cupy.cuda.memory.OutOfMemoryError: Out of memory"
        # during "emb = rotate_embeddings(...)"
        big_job_cpu_workers=3,

    )

    return kargs
