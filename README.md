
Beads simulator
====================================================

[![python](https://img.shields.io/badge/python-3.10+-3776AB.svg?style=flat&logo=python&logoColor=3776AB)](https://www.python.org/)
[![license](https://img.shields.io/github/license/cell-observatory/beads_simulator.svg?style=flat&logo=git&logoColor=white)](https://opensource.org/license/bsd-2-clause/)
[![issues](https://img.shields.io/github/issues/cell-observatory/beads_simulator.svg?style=flat&logo=github)](https://github.com/cell-observatory/beads_simulator/issues)
[![pr](https://img.shields.io/github/issues-pr/cell-observatory/beads_simulator.svg?style=flat&logo=github)](https://github.com/cell-observatory/beads_simulator/pulls)

* [Installation](#installation)
* [Features](#features)
* [Testing](#testing)

# Installation

## Clone repository to your host system
```shell
git clone --recurse-submodules https://github.com/cell-observatory/beads_simulator.git
```

To later update to the latest, greatest
```shell
git pull --recurse-submodules
```

## Create conda environment
```shell
conda/mamba create --name beads --file requirements.txt
```
Or use `pip install -r requirements.txt`



# Features

## Wavefront simulation ([wavefront.py](src/wavefront.py))

<div align="center">
  <img class="center" src="https://www.dropbox.com/scl/fi/i5bets9ewgue7sd3n8irl/distributions.png?rlkey=yc8flirdxjx92asbcuk2smuuy&raw=1" width="100%" />
</div>

- Simulation of ideal and aberrated wavefronts where the amplitudes of the zernike modes are drawn from a given distribution:
  - Single
  - Bimodal
  - Multinomial
  - Powerlaw
  - Dirichlet
 
## Fourier embedding ([embedding.py](src/embedding.py))

<div align="center">
  <img class="center" src="https://www.dropbox.com/scl/fi/4miprr445ujle7hu3djei/embedding.png?rlkey=yna1zfy986yr0mehwz1uwpnbn&raw=1" width="100%" />
</div>

## PSF dataset generator ([psf_dataset.py](src/psf_dataset.py))

Supported PSFs ([synthatic.py](src/synthatic.py))
  - Lattice light sheets
  - Widefield
  - 2photon
  - Confocal

```shell
usage: psf_dataset.py [-h] [--filename FILENAME] [--outdir OUTDIR] [--emb] [--iters ITERS] [--kernels] [--noise] [--normalize] [--x_voxel_size X_VOXEL_SIZE] [--y_voxel_size Y_VOXEL_SIZE] [--z_voxel_size Z_VOXEL_SIZE] [--input_shape INPUT_SHAPE] [--modes MODES] [--min_photons MIN_PHOTONS] [--max_photons MAX_PHOTONS] [--psf_type PSF_TYPE] [--dist DIST] [--mode_dist MODE_DIST] [--gamma GAMMA] [--signed] [--rotate] [--min_amplitude MIN_AMPLITUDE]
                      [--max_amplitude MAX_AMPLITUDE] [--min_lls_defocus_offset MIN_LLS_DEFOCUS_OFFSET] [--max_lls_defocus_offset MAX_LLS_DEFOCUS_OFFSET] [--refractive_index REFRACTIVE_INDEX] [--na_detection NA_DETECTION] [--lam_detection LAM_DETECTION] [--cpu_workers CPU_WORKERS] [--use_theoretical_widefield_simulator] [--skip_remove_background]

options:
  -h, --help            show this help message and exit
  --filename FILENAME
  --outdir OUTDIR
  --emb                 toggle to save fourier embeddings only
  --iters ITERS         number of samples (Default: `10`)
  --kernels             toggle to save raw kernels
  --noise               toggle to add random background and shot noise to the generated PSFs
  --normalize           toggle to scale the generated PSFs to 1.0
  --x_voxel_size X_VOXEL_SIZE
                        lateral voxel size in microns for X (Default: `0.125`)
  --y_voxel_size Y_VOXEL_SIZE
                        lateral voxel size in microns for Y (Default: `0.125`)
  --z_voxel_size Z_VOXEL_SIZE
                        axial voxel size in microns for Z (Default: `0.2`)
  --input_shape INPUT_SHAPE
                        PSF input shape (Default: `64`)
  --modes MODES         number of modes to describe aberration (Default: `55`)
  --min_photons MIN_PHOTONS
                        minimum photons for training samples (Default: `5000`)
  --max_photons MAX_PHOTONS
                        maximum photons for training samples (Default: `10000`)
  --psf_type PSF_TYPE   widefield, 2photon, confocal, or a path to an LLS excitation profile  (Default: `../lattice/YuMB_NAlattice0p35_NAAnnulusMax0p40_NAsigma0p1.mat`)
  --dist DIST           distribution of the zernike amplitudes (Default: `single`)
  --mode_dist MODE_DIST
                        distribution of the zernike modes (Default: `pyramid`)
  --gamma GAMMA         exponent for the powerlaw distribution (Default: `0.75`)
  --signed              optional flag to generate a symmetric (pos/neg) semi-distributions for the given range of amplitudes
  --rotate              optional flag to introduce a random radial rotation to each zernike mode
  --min_amplitude MIN_AMPLITUDE
                        min amplitude for the zernike coefficients (Default: `0`)
  --max_amplitude MAX_AMPLITUDE
                        max amplitude for the zernike coefficients (Default: `0.25`)
  --min_lls_defocus_offset MIN_LLS_DEFOCUS_OFFSET
                        min value for the offset between the excitation and detection focal plan (microns) (Default: `0`)
  --max_lls_defocus_offset MAX_LLS_DEFOCUS_OFFSET
                        max value for the offset between the excitation and detection focal plan (microns) (Default: `0`)
  --refractive_index REFRACTIVE_INDEX
                        the quotient of the speed of light as it passes through two media (Default: `1.33`)
  --na_detection NA_DETECTION
                        Numerical aperture (Default: `1.0`)
  --lam_detection LAM_DETECTION
                        wavelength in microns (Default: `0.51`)
  --cpu_workers CPU_WORKERS
                        number of CPU cores to use (Default: `-1`)
  --use_theoretical_widefield_simulator
                        optional toggle to use an experimental complex pupil to estimate amplitude attenuation (cosine factor)
  --skip_remove_background
                        optional toggle to skip preprocessing input data using the DoG filter
```


## Beads dataset generator ([multipoint_dataset.py](src/multipoint_dataset.py))

Synthatic samples with 1 up to $n$ beads randomly placed in a given FOV

```shell
usage: multipoint_dataset.py [-h] [--filename FILENAME] [--npoints NPOINTS] [--outdir OUTDIR] [--emb] [--embedding_option EMBEDDING_OPTION] [--iters ITERS] [--kernels] [--noise] [--normalize] [--x_voxel_size X_VOXEL_SIZE] [--y_voxel_size Y_VOXEL_SIZE] [--z_voxel_size Z_VOXEL_SIZE] [--input_shape INPUT_SHAPE] [--random_crop RANDOM_CROP] [--modes MODES] [--min_photons MIN_PHOTONS] [--max_photons MAX_PHOTONS] [--psf_type PSF_TYPE] [--dist DIST]
                             [--mode_dist MODE_DIST] [--gamma GAMMA] [--signed] [--rotate] [--randomize_object_size] [--min_amplitude MIN_AMPLITUDE] [--max_amplitude MAX_AMPLITUDE] [--min_lls_defocus_offset MIN_LLS_DEFOCUS_OFFSET] [--max_lls_defocus_offset MAX_LLS_DEFOCUS_OFFSET] [--refractive_index REFRACTIVE_INDEX] [--na_detection NA_DETECTION] [--fill_radius FILL_RADIUS] [--object_size OBJECT_SIZE] [--uniform_background UNIFORM_BACKGROUND]
                             [--lam_detection LAM_DETECTION] [--alpha_val ALPHA_VAL] [--phi_val PHI_VAL] [--cpu_workers CPU_WORKERS] [--override] [--plot] [--use_theoretical_widefield_simulator] [--skip_remove_background]

options:
  -h, --help            show this help message and exit
  --filename FILENAME
  --npoints NPOINTS
  --outdir OUTDIR
  --emb                 toggle to save fourier embeddings only
  --embedding_option EMBEDDING_OPTION
                        type of embedding to use: ["spatial_planes", "principle_planes", "rotary_slices", "spatial_quadrants"] (Default: `['spatial_planes']`)
  --iters ITERS         number of samples (Default: `10`)
  --kernels             toggle to save raw kernels
  --noise               toggle to add random background and shot noise to the generated PSFs
  --normalize           toggle to scale the generated PSFs to 1.0
  --x_voxel_size X_VOXEL_SIZE
                        lateral voxel size in microns for X (Default: `0.125`)
  --y_voxel_size Y_VOXEL_SIZE
                        lateral voxel size in microns for Y (Default: `0.125`)
  --z_voxel_size Z_VOXEL_SIZE
                        axial voxel size in microns for Z (Default: `0.2`)
  --input_shape INPUT_SHAPE
                        PSF input shape (Default: `64`)
  --random_crop RANDOM_CROP
  --modes MODES         number of modes to describe aberration (Default: `55`)
  --min_photons MIN_PHOTONS
                        minimum photons for training samples (Default: `5000`)
  --max_photons MAX_PHOTONS
                        maximum photons for training samples (Default: `10000`)
  --psf_type PSF_TYPE   widefield, 2photon, confocal, or a path to an LLS excitation profile (Default: `['../lattice/YuMB_NAlattice0p35_NAAnnulusMax0p40_NAsigma0p1.mat']`)
  --dist DIST           distribution of the zernike amplitudes (Default: `single`)
  --mode_dist MODE_DIST
                        distribution of the zernike modes (Default: `pyramid`)
  --gamma GAMMA         exponent for the powerlaw distribution (Default: `0.75`)
  --signed              optional flag to generate a symmetric (pos/neg) semi-distributions for the given range of amplitudes
  --rotate              optional flag to introduce a random radial rotation to each zernike mode
  --randomize_object_size
                        optional flag to randomize voxel size during training
  --min_amplitude MIN_AMPLITUDE
                        min amplitude for the zernike coefficients (Default: `0`)
  --max_amplitude MAX_AMPLITUDE
                        max amplitude for the zernike coefficients (Default: `0.25`)
  --min_lls_defocus_offset MIN_LLS_DEFOCUS_OFFSET
                        min value for the offset between the excitation and detection focal plan (microns) (Default: `0`)
  --max_lls_defocus_offset MAX_LLS_DEFOCUS_OFFSET
                        max value for the offset between the excitation and detection focal plan (microns) (Default: `0`)
  --refractive_index REFRACTIVE_INDEX
                        the quotient of the speed of light as it passes through two media (Default: `1.33`)
  --na_detection NA_DETECTION
                        Numerical aperture (Default: `1.0`)
  --fill_radius FILL_RADIUS
                        Fractional cylinder radius (0-1) that defines where a bead may be placed in X Y Z. (Default: `0.0`)
  --object_size OBJECT_SIZE
                        optional bead size (Default: 0 for diffraction-limited beads, -1 for beads with random sizes) (Default: `0.0`)
  --uniform_background UNIFORM_BACKGROUND
                        optional uniform background value (Default: `0`)
  --lam_detection LAM_DETECTION
                        wavelength in microns (Default: `0.51`)
  --alpha_val ALPHA_VAL
                        values to use for the `alpha` embedding [options: real, abs] (Default: `abs`)
  --phi_val PHI_VAL     values to use for the `phi` embedding [options: angle, imag, abs] (Default: `angle`)
  --cpu_workers CPU_WORKERS
                        number of CPU cores to use (Default: `-1`)
  --override            optional toggle to override existing data
  --plot                optional toggle to plot preprocessing
  --use_theoretical_widefield_simulator
                        optional toggle to use an experimental complex pupil to estimate amplitude attenuation (cosine factor)
  --skip_remove_background
                        optional toggle to skip preprocessing input data using the DoG filter
```

# Testing

Running `tests/test_datasets.py` will create a few example datasets.
```shell
pytest -s -v --disable-pytest-warnings --color=yes tests/test_datasets.py
```

```shell
platform linux -- Python 3.12.9, pytest-8.3.4, pluggy-1.5.0 -- /home/thayer/miniforge3/envs/beads/bin/python3.12
cachedir: .pytest_cache
rootdir: /home/thayer/Github/beads_simulator
plugins: order-1.3.0
collected 9 items

tests/test_datasets.py::test_theoretical_widefield_simulator Loading cached SyntheticPSF instance from /home/thayer/Github/beads_simulator/SyntheticPSFCache/_home_thayer_Github_beads_simulator_lattice_YuMB_NAlattice0p35_NAAnnulusMax0p40_NAsigma0p1.mat_shape_64-64-64_lam_0.51_na_1.0_ri_1.33_x_0.097_y_0.097_z_0.2_twd_simulator_True
PASSED
tests/test_datasets.py::test_experimental_widefield_simulator Loading cached SyntheticPSF instance from /home/thayer/Github/beads_simulator/SyntheticPSFCache/_home_thayer_Github_beads_simulator_lattice_YuMB_NAlattice0p35_NAAnnulusMax0p40_NAsigma0p1.mat_shape_64-64-64_lam_0.51_na_1.0_ri_1.33_x_0.097_y_0.097_z_0.2_twd_simulator_False
PASSED
tests/test_datasets.py::test_random_aberrated_psf Loading cached SyntheticPSF instance from /home/thayer/Github/beads_simulator/SyntheticPSFCache/_home_thayer_Github_beads_simulator_lattice_YuMB_NAlattice0p35_NAAnnulusMax0p40_NAsigma0p1.mat_shape_64-64-64_lam_0.51_na_1.0_ri_1.33_x_0.097_y_0.097_z_0.2_twd_simulator_False
PASSED
tests/test_datasets.py::test_random_defocused_psf Loading cached SyntheticPSF instance from /home/thayer/Github/beads_simulator/SyntheticPSFCache/_home_thayer_Github_beads_simulator_lattice_YuMB_NAlattice0p35_NAAnnulusMax0p40_NAsigma0p1.mat_shape_64-64-64_lam_0.51_na_1.0_ri_1.33_x_0.097_y_0.097_z_0.2_twd_simulator_False
PASSED
tests/test_datasets.py::test_random_aberrated_defocused_psf Loading cached SyntheticPSF instance from /home/thayer/Github/beads_simulator/SyntheticPSFCache/_home_thayer_Github_beads_simulator_lattice_YuMB_NAlattice0p35_NAAnnulusMax0p40_NAsigma0p1.mat_shape_64-64-64_lam_0.51_na_1.0_ri_1.33_x_0.097_y_0.097_z_0.2_twd_simulator_False
PASSED
tests/test_datasets.py::test_psf_dataset Loading cached SyntheticPSF instance from /home/thayer/Github/beads_simulator/SyntheticPSFCache/_home_thayer_Github_beads_simulator_lattice_YuMB_NAlattice0p35_NAAnnulusMax0p40_NAsigma0p1.mat_shape_64-64-64_lam_0.51_na_1.0_ri_1.33_x_0.097_y_0.097_z_0.2_twd_simulator_False
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10/10 [00:03<00:00,  2.91it/s]
PASSED
tests/test_datasets.py::test_multipoint_dataset Loading cached SyntheticPSF instance from /home/thayer/Github/beads_simulator/SyntheticPSFCache/_home_thayer_Github_beads_simulator_lattice_YuMB_NAlattice0p35_NAAnnulusMax0p40_NAsigma0p1.mat_shape_64-64-64_lam_0.51_na_1.0_ri_1.33_x_0.097_y_0.097_z_0.2_twd_simulator_False
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10/10 [00:13<00:00,  1.40s/it]
PASSED
tests/test_datasets.py::test_randomize_object_size_dataset Loading cached SyntheticPSF instance from /home/thayer/Github/beads_simulator/SyntheticPSFCache/_home_thayer_Github_beads_simulator_lattice_YuMB_NAlattice0p35_NAAnnulusMax0p40_NAsigma0p1.mat_shape_64-64-64_lam_0.51_na_1.0_ri_1.33_x_0.097_y_0.097_z_0.2_twd_simulator_False
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10/10 [00:15<00:00,  1.50s/it]
PASSED
tests/test_datasets.py::test_multimodal_dataset Loading cached SyntheticPSF instance from /home/thayer/Github/beads_simulator/SyntheticPSFCache/.._lattice_YuMB_NAlattice0p35_NAAnnulusMax0p40_NAsigma0p1.mat_shape_64-64-64_lam_0.51_na_1.0_ri_1.33_x_0.097_y_0.097_z_0.2_twd_simulator_False
Loading cached SyntheticPSF instance from /home/thayer/Github/beads_simulator/SyntheticPSFCache/.._lattice_YuMB_NAlattice0p35_NAAnnulusMax0p40_NAsigma0p1.mat_shape_128-128-128_lam_0.51_na_1.0_ri_1.33_x_0.097_y_0.097_z_0.2_twd_simulator_False
Loading cached SyntheticPSF instance from /home/thayer/Github/beads_simulator/SyntheticPSFCache/.._lattice_Gaussian_NAexc0p21_NAsigma0p21_annulus0p4-0p2_crop0p1_FWHM51p0.mat_shape_64-64-64_lam_0.51_na_1.0_ri_1.33_x_0.097_y_0.097_z_0.2_twd_simulator_False
Loading cached SyntheticPSF instance from /home/thayer/Github/beads_simulator/SyntheticPSFCache/.._lattice_Gaussian_NAexc0p21_NAsigma0p21_annulus0p4-0p2_crop0p1_FWHM51p0.mat_shape_128-128-128_lam_0.51_na_1.0_ri_1.33_x_0.097_y_0.097_z_0.2_twd_simulator_False
Loading cached SyntheticPSF instance from /home/thayer/Github/beads_simulator/SyntheticPSFCache/.._lattice_MBSq_NAexc0p30_annulus0p375-0p225_FWHM48p5.mat_shape_64-64-64_lam_0.51_na_1.0_ri_1.33_x_0.097_y_0.097_z_0.2_twd_simulator_False
Loading cached SyntheticPSF instance from /home/thayer/Github/beads_simulator/SyntheticPSFCache/.._lattice_MBSq_NAexc0p30_annulus0p375-0p225_FWHM48p5.mat_shape_128-128-128_lam_0.51_na_1.0_ri_1.33_x_0.097_y_0.097_z_0.2_twd_simulator_False
Loading cached SyntheticPSF instance from /home/thayer/Github/beads_simulator/SyntheticPSFCache/.._lattice_Sinc_by_lateral_SW_NAexc0p32_NAsigma5p0_annulus0p4-0p2_realSLM_FWHM51p5.mat_shape_64-64-64_lam_0.51_na_1.0_ri_1.33_x_0.097_y_0.097_z_0.2_twd_simulator_False
Loading cached SyntheticPSF instance from /home/thayer/Github/beads_simulator/SyntheticPSFCache/.._lattice_Sinc_by_lateral_SW_NAexc0p32_NAsigma5p0_annulus0p4-0p2_realSLM_FWHM51p5.mat_shape_128-128-128_lam_0.51_na_1.0_ri_1.33_x_0.097_y_0.097_z_0.2_twd_simulator_False
Loading cached SyntheticPSF instance from /home/thayer/Github/beads_simulator/SyntheticPSFCache/.._lattice_v2Hex_NAexc0p50_NAsigma0p075_annulus0p60-0p40_FWHM53p0.mat_shape_64-64-64_lam_0.51_na_1.0_ri_1.33_x_0.097_y_0.097_z_0.2_twd_simulator_False
Loading cached SyntheticPSF instance from /home/thayer/Github/beads_simulator/SyntheticPSFCache/.._lattice_v2Hex_NAexc0p50_NAsigma0p075_annulus0p60-0p40_FWHM53p0.mat_shape_128-128-128_lam_0.51_na_1.0_ri_1.33_x_0.097_y_0.097_z_0.2_twd_simulator_False
Loading cached SyntheticPSF instance from /home/thayer/Github/beads_simulator/SyntheticPSFCache/.._lattice_v2HexRect_NAexc0p50_NAsigma0p15_annulus0p60-0p40_FWHM_56p0.mat_shape_64-64-64_lam_0.51_na_1.0_ri_1.33_x_0.097_y_0.097_z_0.2_twd_simulator_False
Loading cached SyntheticPSF instance from /home/thayer/Github/beads_simulator/SyntheticPSFCache/.._lattice_v2HexRect_NAexc0p50_NAsigma0p15_annulus0p60-0p40_FWHM_56p0.mat_shape_128-128-128_lam_0.51_na_1.0_ri_1.33_x_0.097_y_0.097_z_0.2_twd_simulator_False
Loading cached SyntheticPSF instance from /home/thayer/Github/beads_simulator/SyntheticPSFCache/.._lattice_ACHex_NAexc0p40_NAsigma0p075_annulus0p6-0p2_crop0p1_FWHM52p0.mat_shape_64-64-64_lam_0.51_na_1.0_ri_1.33_x_0.097_y_0.097_z_0.2_twd_simulator_False
Loading cached SyntheticPSF instance from /home/thayer/Github/beads_simulator/SyntheticPSFCache/.._lattice_ACHex_NAexc0p40_NAsigma0p075_annulus0p6-0p2_crop0p1_FWHM52p0.mat_shape_128-128-128_lam_0.51_na_1.0_ri_1.33_x_0.097_y_0.097_z_0.2_twd_simulator_False
Loading cached SyntheticPSF instance from /home/thayer/Github/beads_simulator/SyntheticPSFCache/.._lattice_MBHex_NAexc0p43_annulus0p47_0p40_crop0p08_FWHM48p0.mat_shape_64-64-64_lam_0.51_na_1.0_ri_1.33_x_0.097_y_0.097_z_0.2_twd_simulator_False
Loading cached SyntheticPSF instance from /home/thayer/Github/beads_simulator/SyntheticPSFCache/.._lattice_MBHex_NAexc0p43_annulus0p47_0p40_crop0p08_FWHM48p0.mat_shape_128-128-128_lam_0.51_na_1.0_ri_1.33_x_0.097_y_0.097_z_0.2_twd_simulator_False
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10/10 [02:37<00:00, 15.79s/it]
PASSED
```


# BibTeX

```bibtex
comming soon
```

# License 

This work is licensed under the [BSD 2-Clause License](https://github.com/cell-observatory/beads_simulator/blob/main/LICENSE)