from pathlib import Path
import pytest


@pytest.fixture(scope="session")
def kargs():
    repo = Path.cwd()
    num_modes = 15

    kargs = dict(
        repo=repo,
        num_modes=num_modes,
        prediction_filename_pattern=r"*[!_gt|!_realspace|!_noisefree|!_predictions_psf|!_corrected_psf|!_reconstructed_psf].tif",
        wavelength=.510,
        lateral_voxel_size=.097,
        axial_voxel_size=.2,
        freq_strength_threshold=.01,
        prediction_threshold=0.,
        confidence_threshold=0.02,
        batch_size=64,
        ignore_modes=[0, 1, 2, 4],
        big_job_cpu_workers=3,
    )

    return kargs
