
Beads simulator for AOViFT
====================================================

[![arXiv](https://img.shields.io/badge/arXiv-2503.12593-b31b1b.svg)](https://arxiv.org/abs/2503.12593)
[![python](https://img.shields.io/badge/python-3.10+-3776AB.svg?style=flat&logo=python&logoColor=3776AB)](https://www.python.org/)
[![license](https://img.shields.io/github/license/cell-observatory/beads_simulator.svg?style=flat&logo=git&logoColor=white)](https://opensource.org/license/bsd-2-clause/)
[![issues](https://img.shields.io/github/issues/cell-observatory/beads_simulator.svg?style=flat&logo=github)](https://github.com/cell-observatory/beads_simulator/issues)
[![pr](https://img.shields.io/github/issues-pr/cell-observatory/beads_simulator.svg?style=flat&logo=github)](https://github.com/cell-observatory/beads_simulator/pulls)

<div style="text-align: center; width: 100%; display: inline-block; text-align: center;" >
 <h2>Fourier-Based 3D Multistage Transformer for Aberration Correction in Multicellular Specimens</h2>
  <p>
  Thayer Alshaabi<sup>1,2*</sup>, Daniel E. Milkie<sup>1</sup>, Gaoxiang Liu<sup>2</sup>, Cyna Shirazinejad<sup>2</sup>, Jason L. Hong<sup>2</sup>, Kemal Achour<sup>2</sup>, Frederik Görlitz<sup>2</sup>, Ana Milunovic-Jevtic<sup>2</sup>, Cat Simmons<sup>2</sup>, Ibrahim S. Abuzahriyeh<sup>2</sup>, Erin Hong<sup>2</sup>, Samara Erin Williams<sup>2</sup>, Nathanael Harrison<sup>2</sup>, Evan Huang<sup>2</sup>, Eun Seok Bae<sup>2</sup>, Alison N. Killilea<sup>2</sup>, David G. Drubin<sup>2</sup>, Ian A. Swinburne<sup>2</sup>, Srigokul Upadhyayula<sup>2,3,4*</sup>, Eric Betzig<sup>1,2,5*</sup>
  </p>
  <h5>
    <sup>1</sup>HHMI, <sup>2</sup>UC Berkeley, <sup>3</sup>Lawrence Berkeley National Laboratory, <sup>4</sup>Chan Zuckerberg Biohub, <sup>5</sup>Helen Wills Neuroscience Institute
  </h5>
  <div align="center">

  [![arXiv](https://img.shields.io/badge/arXiv-2503.12593-b31b1b.svg?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2503.12593) &nbsp;
  [![Pytest](https://img.shields.io/badge/pytest-suite-%23ffffff.svg?style=for-the-badge&logo=pytest&logoColor=2f9fe3)](https://github.com/cell-observatory/beads_simulator/tree/main/tests) &nbsp;
  [![BibTeX](https://img.shields.io/badge/BibTeX-reference-%23008080.svg?style=for-the-badge&logo=latex&logoColor=white)](#bibtex)

  </div>
</div>


* [Installation](#installation)
* [Features](#features)
  * [Wavefront simulation](#wavefront-simulation)
  * [Fourier embedding](#fourier-embedding)
  * [PSF dataset generator](#psf-dataset-generator)
  * [Beads dataset generator](#beads-dataset-generator)
  * [Multimodal beads dataset generator](#multimodal-beads-dataset-generator)


# Installation

## Dependencies

> [!IMPORTANT] 
> Source code is tested on the following operating systems:
> - **Ubuntu 22.04** 
> - **Rocky Linux 8.10 & 9.3**
> - **Windows 11 Pro 22621**

```requirements
numpy
pandas
cupy
cuda-version==12.8
astropy
seaborn
scikit-image
scikit-spatial
pytest
pytest-order
matplotlib==3.8.4
ujson
zarr
pycudadecon
dphtools
tifffile==2023.9.18
imagecodecs==2023.9.18
nvitop
pycuda
pytest
tqdm
cachetools
line_profiler_pycharm
```

> [!CAUTION] 
> NVIDIA GPU with a driver release **545** or later, and **CUDA 12.8**.


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


## Running the simulator
Activate conda environment
```shell
conda activate beads
```

# Features

* [Wavefront simulation](#wavefront-simulation)
* [Fourier embedding](#fourier-embedding)
* [PSF dataset generator](#psf-dataset-generator)
* [Beads dataset generator](#beads-dataset-generator)
* [Multimodal beads dataset generator](#multimodal-beads-dataset-generator)


## Wavefront simulation

- Simulation of ideal and aberrated wavefronts where the amplitudes of the zernike modes are drawn from a given distribution:
  - Single
  - Bimodal
  - Multinomial
  - Powerlaw
  - Dirichlet
 
> [!TIP]
> For more options, please refer to [wavefront.py](src/wavefront.py)


<div align="center">
  <img class="center" src="https://www.dropbox.com/scl/fi/i5bets9ewgue7sd3n8irl/distributions.png?rlkey=yc8flirdxjx92asbcuk2smuuy&raw=1" width="100%" />
</div>



## Fourier embedding

> [!TIP]
> For more options, please refer to [embedding.py](src/embedding.py)

<div align="center">
  <img class="center" src="https://www.dropbox.com/scl/fi/4miprr445ujle7hu3djei/embedding.png?rlkey=yna1zfy986yr0mehwz1uwpnbn&raw=1" width="100%" />
</div>


### Aberrated PSF 
```shell
pytest -s -v --disable-pytest-warnings --color=yes tests/test_datasets.py -k test_random_aberrated_psf
```

```shell
=========================================================== test session starts ===========================================================
platform linux -- Python 3.12.9, pytest-8.3.4, pluggy-1.5.0 -- /home/thayer/miniforge3/envs/beads/bin/python3.12
cachedir: .pytest_cache
rootdir: /home/thayer/Github/beads_simulator
plugins: order-1.3.0
collected 9 items / 8 deselected / 1 selected

tests/test_datasets.py::test_random_aberrated_psf Loading cached SyntheticPSF instance from /home/thayer/Github/beads_simulator/SyntheticPSFCache/_home_thayer_Github_beads_simulator_lattice_YuMB_NAlattice0p35_NAAnnulusMax0p40_NAsigma0p1.mat_shape_64-64-64_lam_0.51_na_1.0_ri_1.33_x_0.097_y_0.097_z_0.2_twd_simulator_False
PASSED
=============================================== 1 passed, 8 deselected, 1 warning in 7.68s ================================================
```


### LLS defocused PSF 

```shell
pytest -s -v --disable-pytest-warnings --color=yes tests/test_datasets.py -k test_random_defocused_psf
```

```shell
=========================================================== test session starts ===========================================================
platform linux -- Python 3.12.9, pytest-8.3.4, pluggy-1.5.0 -- /home/thayer/miniforge3/envs/beads/bin/python3.12
cachedir: .pytest_cache
rootdir: /home/thayer/Github/beads_simulator
plugins: order-1.3.0
collected 9 items / 8 deselected / 1 selected

tests/test_datasets.py::test_random_defocused_psf Loading cached SyntheticPSF instance from /home/thayer/Github/beads_simulator/SyntheticPSFCache/_home_thayer_Github_beads_simulator_lattice_YuMB_NAlattice0p35_NAAnnulusMax0p40_NAsigma0p1.mat_shape_64-64-64_lam_0.51_na_1.0_ri_1.33_x_0.097_y_0.097_z_0.2_twd_simulator_False
PASSED

=============================================== 1 passed, 8 deselected, 1 warning in 7.68s ================================================
```

### Aberrated and LLS defocused PSF 

```shell
pytest -s -v --disable-pytest-warnings --color=yes tests/test_datasets.py -k test_random_aberrated_defocused_psf
```

```shell
=========================================================== test session starts ===========================================================
platform linux -- Python 3.12.9, pytest-8.3.4, pluggy-1.5.0 -- /home/thayer/miniforge3/envs/beads/bin/python3.12
cachedir: .pytest_cache
rootdir: /home/thayer/Github/beads_simulator
plugins: order-1.3.0
collected 9 items / 8 deselected / 1 selected

tests/test_datasets.py::test_random_aberrated_defocused_psf Loading cached SyntheticPSF instance from /home/thayer/Github/beads_simulator/SyntheticPSFCache/_home_thayer_Github_beads_simulator_lattice_YuMB_NAlattice0p35_NAAnnulusMax0p40_NAsigma0p1.mat_shape_64-64-64_lam_0.51_na_1.0_ri_1.33_x_0.097_y_0.097_z_0.2_twd_simulator_False
PASSED
=============================================== 1 passed, 8 deselected, 1 warning in 7.71s ================================================
```


## PSF dataset generator

> [!TIP]
> For more options, please refer to [psf_dataset.py](src/psf_dataset.py)



```shell
pytest -s -v --disable-pytest-warnings --color=yes tests/test_datasets.py -k test_psf_dataset
```

```shell
=========================================================== test session starts ===========================================================
platform linux -- Python 3.12.9, pytest-8.3.4, pluggy-1.5.0 -- /home/thayer/miniforge3/envs/beads/bin/python3.12
cachedir: .pytest_cache
rootdir: /home/thayer/Github/beads_simulator
plugins: order-1.3.0
collected 9 items / 8 deselected / 1 selected

tests/test_datasets.py::test_psf_dataset Loading cached SyntheticPSF instance from /home/thayer/Github/beads_simulator/SyntheticPSFCache/_home_thayer_Github_beads_simulator_lattice_YuMB_NAlattice0p35_NAAnnulusMax0p40_NAsigma0p1.mat_shape_64-64-64_lam_0.51_na_1.0_ri_1.33_x_0.097_y_0.097_z_0.2_twd_simulator_False
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████| 10/10 [00:03<00:00,  2.83it/s]
PASSED
===================================================== 1 passed, 8 deselected in 5.19s =====================================================
```


> [!TIP]
> For more options, please refer to `psf_dataset.py --help`

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


## Beads dataset generator

Synthatic samples with 1 up to $n$ beads randomly placed in a given FOV


```shell
pytest -s -v --disable-pytest-warnings --color=yes tests/test_datasets.py -k test_multipoint_dataset
```
```shell
=========================================================== test session starts ===========================================================
platform linux -- Python 3.12.9, pytest-8.3.4, pluggy-1.5.0 -- /home/thayer/miniforge3/envs/beads/bin/python3.12
cachedir: .pytest_cache
rootdir: /home/thayer/Github/beads_simulator
plugins: order-1.3.0
collected 9 items / 8 deselected / 1 selected

tests/test_datasets.py::test_multipoint_dataset Loading cached SyntheticPSF instance from /home/thayer/Github/beads_simulator/SyntheticPSFCache/_home_thayer_Github_beads_simulator_lattice_YuMB_NAlattice0p35_NAAnnulusMax0p40_NAsigma0p1.mat_shape_64-64-64_lam_0.51_na_1.0_ri_1.33_x_0.097_y_0.097_z_0.2_twd_simulator_False
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████| 10/10 [00:14<00:00,  1.48s/it]
PASSED
=============================================== 1 passed, 8 deselected, 1 warning in 16.47s ===============================================
```


> [!TIP]
> For more options, please refer to [multipoint_dataset.py](src/multipoint_dataset.py)

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



## Multimodal beads dataset generator

Synthatic samples with 1 up to $n$ beads randomly placed in a given FOV

> [!NOTE]
> Supported PSFs ([synthatic.py](src/synthatic.py))
>  - Lattice light sheets
>  - Widefield
>  - 2photon
>  - Confocal


```shell
pytest -s -v --disable-pytest-warnings --color=yes tests/test_datasets.py -k test_multimodal_dataset
```

```shell
=========================================================== test session starts ===========================================================
platform linux -- Python 3.12.9, pytest-8.3.4, pluggy-1.5.0 -- /home/thayer/miniforge3/envs/beads/bin/python3.12
cachedir: .pytest_cache
rootdir: /home/thayer/Github/beads_simulator
plugins: order-1.3.0
collected 9 items / 8 deselected / 1 selected

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
Loading cached SyntheticPSF instance from /home/thayer/Github/beads_simulator/SyntheticPSFCache/widefield_shape_64-64-64_lam_0.51_na_1.0_ri_1.33_x_0.097_y_0.097_z_0.2_twd_simulator_False
Loading cached SyntheticPSF instance from /home/thayer/Github/beads_simulator/SyntheticPSFCache/widefield_shape_128-128-128_lam_0.51_na_1.0_ri_1.33_x_0.097_y_0.097_z_0.2_twd_simulator_False
Loading cached SyntheticPSF instance from /home/thayer/Github/beads_simulator/SyntheticPSFCache/2photon_shape_64-64-64_lam_0.92_na_1.0_ri_1.33_x_0.097_y_0.097_z_0.2_twd_simulator_False
Loading cached SyntheticPSF instance from /home/thayer/Github/beads_simulator/SyntheticPSFCache/2photon_shape_128-128-128_lam_0.92_na_1.0_ri_1.33_x_0.097_y_0.097_z_0.2_twd_simulator_False
Loading cached SyntheticPSF instance from /home/thayer/Github/beads_simulator/SyntheticPSFCache/confocal_shape_64-64-64_lam_0.51_na_1.0_ri_1.33_x_0.097_y_0.097_z_0.2_twd_simulator_False
Loading cached SyntheticPSF instance from /home/thayer/Github/beads_simulator/SyntheticPSFCache/confocal_shape_128-128-128_lam_0.51_na_1.0_ri_1.33_x_0.097_y_0.097_z_0.2_twd_simulator_False
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████| 10/10 [04:48<00:00, 28.83s/it]
PASSED

========================================= 1 passed, 8 deselected, 1 warning in 290.55s (0:04:50) ==========================================
```



# BibTeX

```bibtex
@article{alshaabi2025fourier,
  title={Fourier-Based 3D Multistage Transformer for Aberration Correction in Multicellular Specimens},
  author={Thayer Alshaabi and Daniel E. Milkie and Gaoxiang Liu and Cyna Shirazinejad and Jason L. Hong and Kemal Achour and Frederik Görlitz and Ana Milunovic-Jevtic and Cat Simmons and Ibrahim S. Abuzahriyeh and Erin Hong and Samara Erin Williams and Nathanael Harrison and Evan Huang and Eun Seok Bae and Alison N. Killilea and David G. Drubin and Ian A. Swinburne and Srigokul Upadhyayula and Eric Betzig},
  journal={arXiv preprint arXiv:2503.12593},
  year={2025},
  url={https://arxiv.org/abs/2503.12593},
}
```


# License 

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   [Apache License 2.0](LICENSE)

Copyright 2025 Cell Observatory.