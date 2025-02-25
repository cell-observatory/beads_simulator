import matplotlib

matplotlib.use('Agg')

import pickle
import sys
import logging
import numpy as np
from tqdm.auto import tqdm
from skimage.feature import peak_local_max
from scipy.spatial import KDTree
from astropy import convolution
import multiprocessing as mp
from typing import Any, List, Union, Optional, Generator

try:
    import cupy as cp
except ImportError as e:
    logging.warning(f"Cupy not supported on your system: {e}")

from wavefront import Wavefront

import matplotlib.pyplot as plt

plt.set_loglevel('error')

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)



def multiprocess(
        jobs: Union[Generator, List, np.ndarray],
        func: Any,
        desc: str = 'Processing',
        cores: int = -1,
        unit: str = 'it',
        pool: Optional[mp.Pool] = None,
):
    """ Multiprocess a generic function
    Args:
        func: a python function
        jobs: a list of jobs for function `func`
        desc: description for the progress bar
        cores: number of cores to use

    Returns:
        an array of outputs for every function call
    """

    cores = cores if mp.current_process().name == 'MainProcess' else 1
    # mp.set_start_method('spawn', force=True)
    jobs = list(jobs)

    if cores == 1 or len(jobs) == 1:
        results = []
        for j in tqdm(
                jobs,
                total=len(jobs),
                desc=desc,
                bar_format='{l_bar}{bar}{r_bar} {elapsed_s:.1f}s elapsed',
                unit=unit,
                file=sys.stdout,
        ):
            results.append(func(j))
    elif cores == -1 and len(jobs) > 0:
        with pool if pool is not None else mp.Pool(min(mp.cpu_count(), len(jobs))) as p:
            results = list(tqdm(
                p.imap(func, jobs),
                total=len(jobs),
                desc=desc,
                bar_format='{l_bar}{bar}{r_bar} {elapsed_s:.1f}s elapsed',
                unit=unit,
                file=sys.stdout,
            ))
    elif cores > 1 and len(jobs) > 0:
        with pool if pool is not None else mp.Pool(cores) as p:
            results = list(tqdm(
                p.imap(func, jobs),
                total=len(jobs),
                desc=desc,
                bar_format='{l_bar}{bar}{r_bar} {elapsed_s:.1f}s elapsed',
                unit=unit,
                file=sys.stdout,
            ))
    else:
        raise Exception(f'No data found in {jobs=}')

    return np.array(results)


def photons2electrons(image, quantum_efficiency: float = .82):
    return image * quantum_efficiency


def electrons2photons(image, quantum_efficiency: float = .82):
    return image / quantum_efficiency


def electrons2counts(image, electrons_per_count: float = .22):
    return image / electrons_per_count


def counts2electrons(image, electrons_per_count: float = .22):
    return image * electrons_per_count


def randuniform(var):
    """
    Returns a random number (uniform chance) in the range provided by var. If var is a scalar, var is simply returned.

    Args:
        var : (as scalar) Returned as is.
        var : (as list) Range to provide a random number

    Returns:
        _type_: ndarray or scalar. Random sample from the range provided.

    """
    var = (var, var) if np.isscalar(var) else var

    # star unpacks a list, so that var's values become the separate arguments here
    return np.random.uniform(*var)


def normal_noise(mean: float, sigma: float, size: tuple) -> np.array:
    mean = randuniform(mean)
    sigma = randuniform(sigma)
    return np.random.normal(loc=mean, scale=sigma, size=size).astype(np.float32)


def poisson_noise(image: np.ndarray) -> np.array:
    image = np.nan_to_num(image, nan=0)
    return np.random.poisson(lam=image).astype(np.float32) - image


def add_noise(
        image: np.ndarray,
        mean_background_offset: int = 100,
        sigma_background_noise: int = 40,
        quantum_efficiency: float = .82,
        electrons_per_count: float = .22,
):
    """

    Args:
        image: noise-free image in incident photons
        mean_background_offset: camera background offset
        sigma_background_noise: read noise from the camera
        quantum_efficiency: quantum efficiency of the camera
        electrons_per_count: conversion factor to go from electrons to counts

    Returns:
        noisy image in counts
    """
    image = photons2electrons(image, quantum_efficiency=quantum_efficiency)
    sigma_background_noise *= electrons_per_count  # electrons;  40 counts = 40 * .22 electrons per count
    dark_read_noise = normal_noise(mean=0, sigma=sigma_background_noise, size=image.shape)  # dark image in electrons
    shot_noise = poisson_noise(image)  # shot noise in electrons

    image += shot_noise + dark_read_noise
    image = electrons2counts(image, electrons_per_count=electrons_per_count)

    image += mean_background_offset  # add camera offset (camera offset in counts)
    image[image < 0] = 0
    return image.astype(np.float32)


def microns2waves(a, wavelength):
    return a / wavelength


def waves2microns(a, wavelength):
    return a * wavelength


def peak2valley(w, wavelength: float = .510, na: float = 1.0) -> float:
    if not isinstance(w, Wavefront):
        w = Wavefront(w, lam_detection=wavelength)

    wavefront = w.wave(100)
    center = (int(wavefront.shape[0] / 2), int(wavefront.shape[1] / 2))
    Y, X = np.ogrid[:wavefront.shape[0], :wavefront.shape[1]]
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
    mask = dist_from_center <= (na * wavefront.shape[0]) / 2
    wavefront *= mask
    return abs(np.nanmax(wavefront) - np.nanmin(wavefront))



def mean_min_distance(sample: np.array, voxel_size: tuple, plot: bool = False):
    beads = peak_local_max(
        sample,
        min_distance=0,
        threshold_rel=0,
        exclude_border=False,
        p_norm=2,
    ).astype(np.float32)

    scaled_peaks = np.zeros_like(beads)
    scaled_peaks[:, 0] = beads[:, 0] * voxel_size[0]
    scaled_peaks[:, 1] = beads[:, 1] * voxel_size[1]
    scaled_peaks[:, 2] = beads[:, 2] * voxel_size[2]

    kd = KDTree(scaled_peaks)
    dists, idx = kd.query(scaled_peaks, k=2)

    if plot:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=False, sharex=False)
        for ax in range(3):
            axes[ax].imshow(
                np.nanmax(sample, axis=ax),
                aspect='auto',
                cmap='gray'
            )

            for p in range(dists.shape[0]):
                if ax == 0:
                    axes[ax].plot(beads[p, 2], beads[p, 1], marker='.', ls='', color=f'C{p}')
                elif ax == 1:
                    axes[ax].plot(beads[p, 2], beads[p, 0], marker='.', ls='', color=f'C{p}')
                else:
                    axes[ax].plot(beads[p, 1], beads[p, 0], marker='.', ls='', color=f'C{p}')

        plt.tight_layout()
        plt.show()

    return np.round(np.mean(dists), 1)



def fftconvolution(kernel, sample):
    if kernel.shape[0] == 1 or kernel.shape[-1] == 1:
        kernel = np.squeeze(kernel)

    if sample.shape[0] == 1 or sample.shape[-1] == 1:
        sample = np.squeeze(sample)

    conv = convolution.convolve_fft(
        sample,
        kernel,
        allow_huge=True,
        normalize_kernel=False,
        nan_treatment='fill',
        fill_value=0
    ).astype(sample.dtype)  # otherwise returns as float64
    conv[conv < 0] = 0  # clip negative small values
    return conv


def fft_decon(kernel, sample, iters):
    for k in range(kernel.ndim):
        kernel = np.roll(kernel, kernel.shape[k] // 2, axis=k)

    kernel = cp.array(kernel)
    sample = cp.array(sample)
    deconv = cp.array(sample)

    kernel = cp.fft.rfftn(kernel)

    for _ in range(iters):
        conv = cp.fft.irfftn(cp.fft.rfftn(deconv) * kernel)
        relative_blur = sample / conv
        deconv *= cp.fft.irfftn((cp.fft.rfftn(relative_blur).conj() * kernel).conj())

    return cp.asnumpy(deconv)


def round_to_even(n):
    answer = round(n)
    if not answer % 2:
        return int(answer)
    if abs(answer + 1 - n) < abs(answer - 1 - n):
        return int(answer + 1)
    else:
        return int(answer - 1)


def round_to_odd(n):
    answer = round(n)
    if answer % 2:
        return int(answer)
    if abs(answer + 1 - n) < abs(answer - 1 - n):
        return int(answer + 1)
    else:
        return int(answer - 1)


def gaussian_kernel(kernlen: tuple = (21, 21, 21), std=3):
    """Returns a 3D Gaussian kernel array."""
    x = np.arange((-kernlen[2] // 2) + 1, (-kernlen[2] // 2) + 1 + kernlen[2], 1)
    y = np.arange((-kernlen[1] // 2) + 1, (-kernlen[1] // 2) + 1 + kernlen[1], 1)
    z = np.arange((-kernlen[0] // 2) + 1, (-kernlen[0] // 2) + 1 + kernlen[0], 1)
    zz, yy, xx = np.meshgrid(z, y, x, indexing='ij')
    kernel = np.exp(-(xx ** 2 + yy ** 2 + zz ** 2) / (2 * std ** 2))
    return kernel / np.nansum(kernel)


def fwhm2sigma(w):
    """ convert from full width at half maximum (FWHM) to std """
    return w / (2 * np.sqrt(2 * np.log(2)))


def sigma2fwhm(s):
    """ convert from std to full width at half maximum (FWHM) """
    return s * (2 * np.sqrt(2 * np.log(2)))


def sphere_mask(image_shape, radius=1):
    """
    Args:
        image_shape:
        radius:

    Returns:
        3D Boolean array where True within the sphere
    """
    center = [s // 2 for s in image_shape]
    Z, Y, X = np.ogrid[:image_shape[0], :image_shape[1], :image_shape[2]]
    dist_from_center = np.sqrt((Z - center[0]) ** 2 + (Y - center[1]) ** 2 + (X - center[2]) ** 2)
    mask = dist_from_center <= radius
    return mask



def fft(inputs, padsize=None):
    if padsize is not None:
        shape = inputs.shape[1]
        size = shape * (padsize / shape)
        pad = int((size - shape) // 2)
        inputs = np.pad(inputs, ((pad, pad), (pad, pad), (pad, pad)), 'constant', constant_values=0)

    otf = np.fft.ifftshift(inputs)
    otf = np.fft.fftn(otf)
    otf = np.fft.fftshift(otf)
    return otf



def ifft(otf):
    psf = np.fft.ifftshift(otf)
    psf = np.fft.ifftn(psf)
    psf = np.fft.fftshift(psf)
    return np.abs(psf)



def normalize_otf(otf, freq_strength_threshold: float = 0., percentile: bool = False):

    if percentile:
        otf /= np.nanpercentile(np.abs(otf), 99.99)
    else:
        roi = np.abs(otf[sphere_mask(image_shape=otf.shape, radius=3)])
        dc = np.max(roi)
        otf /= np.mean(roi[roi != dc])

    # since the DC has no bearing on aberration: clamp to -1, +1
    otf[otf > 1] = 1
    otf[otf < -1] = -1

    if freq_strength_threshold != 0.:
        otf[np.abs(otf) < freq_strength_threshold] = 0.

    otf = np.nan_to_num(otf, nan=0, neginf=0, posinf=0)
    return otf


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def save_pickle(x, path):
    with open(path, 'wb') as f:
        pickle.dump(x, f, protocol=pickle.HIGHEST_PROTOCOL)
