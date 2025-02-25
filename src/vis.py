import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
plt.set_loglevel('error')

import re
import warnings
import pandas as pd
from pathlib import Path
from functools import partial
import logging
import sys
import itertools
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import ImageGrid
from numpy.lib.stride_tricks import sliding_window_view
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import FormatStrFormatter, LogFormatterMathtext
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from itertools import cycle

from typing import Any, Union, Optional
import numpy as np
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
from matplotlib import colors

from wavefront import Wavefront
from zernike import Zernike


logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')


def savesvg(
    fig: plt.Figure,
    savepath: Union[Path, str],
    top: float = 0.9,
    bottom: float = 0.1,
    left: float = 0.1,
    right: float = 0.9,
    hspace: float = 0.35,
    wspace: float = 0.1
):

    plt.subplots_adjust(top=top, bottom=bottom, left=left, right=right, hspace=hspace, wspace=wspace)
    plt.savefig(savepath, bbox_inches='tight', dpi=300, pad_inches=.25)

    if Path(savepath).suffix == '.svg':
        # Read in the file
        with open(savepath, 'r', encoding="utf-8") as f:
            filedata = f.read()

        # Replace the target string
        filedata = re.sub('height="[0-9]+(\.[0-9]+)pt"', '', filedata)
        filedata = re.sub('width="[0-9]+(\.[0-9]+)pt"', '', filedata)

        # Write the file out again
        with open(savepath, 'w', encoding="utf-8") as f:
            f.write(filedata)


def plot_mip(
    xy,
    xz,
    yz,
    vol,
    label='',
    gamma=.5,
    cmap='hot',
    dxy=.097,
    dz=.2,
    colorbar=True,
    aspect='auto',
    log=False,
    mip=True,
    ticks=True,
    normalize=False,
    alpha=1.0
):
    def formatter(x, pos, dd):
        return f'{np.ceil(x * dd).astype(int):1d}'

    if log:
        vmin, vmax, step = 1e-4, 1, .025
        norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
        vol[vol < vmin] = vmin
    else:
        vol = vol ** gamma
        vol /= vol.max()
        vol = np.nan_to_num(vol)
        vmin, vmax, step = 0, 1, .025
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    if xy is not None:
        if mip:
            v = np.max(vol, axis=0)
        else:
            v = vol[vol.shape[0]//2, :, :]

        mat_xy = xy.imshow(v, cmap=cmap, aspect=aspect, norm=norm if normalize else None, alpha=alpha)

        xy.set_xlabel(r'XY ($\mu$m)')
        if ticks:
            xy.yaxis.set_ticks_position('right')
            xy.xaxis.set_major_formatter(partial(formatter, dd=dxy))
            xy.yaxis.set_major_formatter(partial(formatter, dd=dxy))
            xy.xaxis.set_major_locator(plt.MaxNLocator(6))
            xy.yaxis.set_major_locator(plt.MaxNLocator(6))
        else:
            xy.axis('off')

    if xz is not None:
        if mip:
            v = np.max(vol, axis=1)
        else:
            v = vol[:, vol.shape[0] // 2, :]

        mat = xz.imshow(v, cmap=cmap, aspect=aspect, norm=norm if normalize else None, alpha=alpha)

        xz.set_xlabel(r'XZ ($\mu$m)')
        if ticks:
            xz.yaxis.set_ticks_position('right')
            xz.xaxis.set_major_formatter(partial(formatter, dd=dxy))
            xz.yaxis.set_major_formatter(partial(formatter, dd=dz))
            xz.xaxis.set_major_locator(plt.MaxNLocator(6))
            xz.yaxis.set_major_locator(plt.MaxNLocator(6))
        else:
            xz.axis('off')

    if yz is not None:
        if mip:
            v = np.max(vol, axis=2)
        else:
            v = vol[:, :, vol.shape[0] // 2]

        mat = yz.imshow(v, cmap=cmap, aspect=aspect, norm=norm if normalize else None, alpha=alpha)

        yz.set_xlabel(r'YZ ($\mu$m)')
        if ticks:
            yz.yaxis.set_ticks_position('right')
            yz.xaxis.set_major_formatter(partial(formatter, dd=dxy))
            yz.yaxis.set_major_formatter(partial(formatter, dd=dz))
            yz.xaxis.set_major_locator(plt.MaxNLocator(6))
            yz.yaxis.set_major_locator(plt.MaxNLocator(6))
        else:
            yz.axis('off')

    if colorbar:
        divider = make_axes_locatable(xy)
        cax = divider.append_axes("left", size="5%", pad=0.1)
        cb = plt.colorbar(
            mat_xy if xy else mat,  # make colorbar out of the xy mip plot
            cax=cax,
            format=LogFormatterMathtext() if log else FormatStrFormatter("%.1f"),
        )

        cb.ax.set_ylabel(f"{label}")
        cb.ax.yaxis.set_label_position("left")
        cb.ax.yaxis.set_ticks_position('left')
        cb.ax.get_xaxis().get_major_formatter().labelOnlyBase = False

    return


def plot_wavefront(
    iax,
    phi,
    rms=None,
    label=None,
    nas=(.85, .95, .99),
    vcolorbar=False,
    hcolorbar=False,
    vmin=None,
    vmax=None,
):
    def formatter(x, pos):
        val_str = '{:.1g}'.format(x)
        if np.abs(x) > 0 and np.abs(x) < 1:
            return val_str.replace("0", "", 1)
        else:
            return val_str

    def na_mask(radius):
        center = (int(phi.shape[0]/2), int(phi.shape[1]/2))
        Y, X = np.ogrid[:phi.shape[0], :phi.shape[1]]
        dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
        mask = dist_from_center <= radius
        return mask

    dlimit = .05

    if vmin is None:
        vmin = np.floor(np.nanmin(phi)*2)/4     # round down to nearest 0.25 wave
        vmin = -1*dlimit if vmin > -0.01 else vmin

    if vmax is None:
        vmax = np.ceil(np.nanmax(phi)*2)/4  # round up to nearest 0.25 wave
        vmax = dlimit if vmax < 0.01 else vmax

    cmap = 'Spectral_r'
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    mat = iax.imshow(phi, cmap=cmap, norm=norm)

    pcts = []
    for d in nas:
        r = (d * phi.shape[0]) / 2
        circle = patches.Circle((50, 50), r, ls='--', ec="dimgrey", fc="none", zorder=3)
        iax.add_patch(circle)

        mask = phi * na_mask(radius=r)
        pcts.append((np.nanquantile(mask, .05), np.nanquantile(mask, .95)))

    phi = phi.flatten()

    if label is not None:
        p2v = abs(np.nanmin(phi) - np.nanmax(phi))
        err = '\n'.join([
            f'$NA_{{{na:.2f}}}$={p2v if na == 1 else abs(p[1]-p[0]):.2f}$\lambda$ (P2V)'
            for na, p in zip(nas, pcts)
        ])
        if label == '':
            iax.set_title(err)
        else:
            if rms is not None:
                iax.set_title(f'{label} RMS[{rms:.2f}$\lambda$]\n{err}\n$NA_{{1.0}}=${p2v:.2f}$\lambda$ (P2V)')
            else:
                iax.set_title(f'{label} [{p2v:.2f}$\lambda$] (P2V)\n{err}')

    iax.axis('off')
    iax.set_aspect("equal")

    if vcolorbar:
        divider = make_axes_locatable(iax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = plt.colorbar(
            mat,
            cax=cax,
            extend='both',
            format=formatter,
        )
        cbar.ax.set_title(r'$\lambda$', pad=10)
        cbar.ax.yaxis.set_ticks_position('right')
        cbar.ax.yaxis.set_label_position('left')

    if hcolorbar:
        divider = make_axes_locatable(iax)
        cax = divider.append_axes("bottom", size="5%", pad=0.1)
        cbar = plt.colorbar(
            mat,
            cax=cax,
            extend='both',
            orientation='horizontal',
            format=formatter,
        )
        cbar.ax.xaxis.set_ticks_position('bottom')
        cbar.ax.xaxis.set_label_position('top')

    return mat



def zernikes(pred: Wavefront, save_path: Path, pred_std: Any = None, lls_defocus: float = 0.):

    plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'axes.autolimit_mode': 'round_numbers'
    })

    pred_wave = pred.wave(size=100)
    pred_rms = np.linalg.norm(pred.amplitudes_noll_waves)

    fig = plt.figure(figsize=(10, 6))
    gs = fig.add_gridspec(4, 3)

    if lls_defocus != 0.:
        ax_wavefront = fig.add_subplot(gs[:-1, -1])
        ax_zcoff = fig.add_subplot(gs[:, :-1])
    else:
        ax_wavefront = fig.add_subplot(gs[:, -1])
        ax_zcoff = fig.add_subplot(gs[:, :-1])

    plot_wavefront(
        ax_wavefront,
        pred_wave,
        rms=pred_rms,
        label='Predicted',
        vcolorbar=True,
    )

    if pred_std is not None:
        ax_zcoff.bar(
            range(len(pred.amplitudes)),
            pred.amplitudes,
            yerr=pred_std.amplitudes,
            capsize=2,
            color='dimgrey',
            alpha=.75,
            align='center',
            ecolor='lightgrey',
        )
    else:
        ax_zcoff.bar(
            range(len(pred.amplitudes)),
            pred.amplitudes,
            capsize=2,
            color='dimgrey',
            alpha=.75,
            align='center',
            ecolor='k',
        )

    ax_zcoff.set_ylabel(f'Zernike coefficients ($\mu$m RMS)')
    ax_zcoff.spines['top'].set_visible(False)
    ax_zcoff.spines['left'].set_visible(False)
    ax_zcoff.spines['right'].set_visible(False)
    ax_zcoff.grid(True, which="both", axis='y', lw=1, ls='--', zorder=0)
    ax_zcoff.set_xticks(range(len(pred.amplitudes)), minor=True)
    ax_zcoff.set_xticks(range(0, len(pred.amplitudes)+5, min(5, int(np.ceil(len(pred.amplitudes)+5)/8))), minor=False) # at least 8 ticks
    ax_zcoff.set_xlim((-.5, len(pred.amplitudes)))
    ax_zcoff.axhline(0, ls='--', color='r', alpha=.5)

    if lls_defocus != 0.:
        ax_defocus = fig.add_subplot(gs[-1, -1])

        data = [lls_defocus]
        bars = ax_defocus.barh(range(len(data)), data)

        for bar in bars:
            bar.set_zorder(1)
            bar.set_facecolor("none")
            x, y = bar.get_xy()
            w, h = bar.get_width(), bar.get_height()
            grad = np.atleast_2d(np.linspace(0, 1 * w / max(data), 256))
            ax_defocus.imshow(
                grad,
                extent=[x, x + w, y, y + h],
                aspect="auto",
                zorder=0,
                cmap='magma'
            )

        ax_defocus.set_title(f'LLS defocus ($\mu$m)')
        ax_defocus.spines['top'].set_visible(False)
        ax_defocus.spines['left'].set_visible(False)
        ax_defocus.spines['right'].set_visible(False)
        ax_defocus.set_yticks([])
        ax_defocus.grid(True, which="both", axis='x', lw=1, ls='--', zorder=0)
        ax_defocus.axvline(0, ls='-', color='k')
        ax_defocus.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        ax_defocus.set_xticklabels(ax_defocus.get_xticks(), rotation=45)
    
    savesvg(fig, f'{save_path}.svg')


def plot_interference(
        plot,
        plot_interference_pattern,
        pois,
        min_distance,
        beads,
        convolved_psf,
        psf_peaks,
        corrected_psf,
        kernel,
        interference_pattern,
        gamma = 0.5,
        high_snr = None,
        estimated_object_gaussian_sigma=0,
):
    fig, axes = plt.subplots(
        nrows=5 if plot_interference_pattern else 4,
        ncols=3,
        figsize=(10, 11),
        sharey=False,
        sharex=False
    )
    transparency = 0.6
    aspect='equal' # 'auto'
    for ax in range(3):
        for p in range(pois.shape[0]):
            if ax == 0:
                axes[0, ax].plot(pois[p, 2], pois[p, 1], marker='x', ls='', color=f'C{p}')
                axes[2, ax].plot(pois[p, 2], pois[p, 1], marker='x', ls='', color=f'C{p}', alpha=transparency)
                axes[2, ax].add_patch(patches.Rectangle(
                    xy=(pois[p, 2] - min_distance, pois[p, 1] - min_distance),
                    width=min_distance * 2,
                    height=min_distance * 2,
                    fill=None,
                    color=f'C{p}',
                    alpha=transparency
                ))
            elif ax == 1:
                axes[0, ax].plot(pois[p, 2], pois[p, 0], marker='x', ls='', color=f'C{p}')
                axes[2, ax].plot(pois[p, 2], pois[p, 0], marker='x', ls='', color=f'C{p}', alpha=transparency)
                axes[2, ax].add_patch(patches.Rectangle(
                    xy=(pois[p, 2] - min_distance, pois[p, 0] - min_distance),
                    width=min_distance * 2,
                    height=min_distance * 2,
                    fill=None,
                    color=f'C{p}',
                    alpha=transparency
                ))

            elif ax == 2:
                axes[0, ax].plot(pois[p, 1], pois[p, 0], marker='x', ls='', color=f'C{p}')
                axes[2, ax].plot(pois[p, 1], pois[p, 0], marker='x', ls='', color=f'C{p}', alpha=transparency)
                axes[2, ax].add_patch(patches.Rectangle(
                    xy=(pois[p, 1] - min_distance, pois[p, 0] - min_distance),
                    width=min_distance * 2,
                    height=min_distance * 2,
                    fill=None,
                    color=f'C{p}',
                    alpha=transparency
                ))
        m1 = axes[0, ax].imshow(np.nanmax(psf_peaks**gamma, axis=ax), cmap='hot', aspect=aspect, vmin=0, vmax=1)
        m2 = axes[1, ax].imshow(np.nanmax(kernel, axis=ax), cmap='hot', aspect=aspect)
        m3 = axes[2, ax].imshow(np.nanmax(convolved_psf, axis=ax), cmap='Greys_r', alpha=.66, aspect=aspect)

        if plot_interference_pattern:
            # interference = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=axes[3, ax], wspace=0.05, hspace=0)
            # ax1 = fig.add_subplot(interference[0])
            # ax1.imshow(np.nanmax(beads, axis=ax), cmap='hot', aspect=aspect)
            # ax1.axis('off')
            # ax1.set_title(r'$\mathcal{S}$')
            #
            # ax2 = fig.add_subplot(interference[1])
            # m4 = ax2.imshow(np.nanmax(abs(interference_pattern), axis=ax), cmap='magma', aspect=aspect)
            # ax2.axis('off')
            # ax2.set_title(r'$|\mathscr{F}(\mathcal{S})|$')

            m4 = axes[3, ax].imshow(np.nanmax(beads, axis=ax), cmap='hot', aspect=aspect)
        m5 = axes[-1, ax].imshow(np.nanmax(corrected_psf**gamma, axis=ax), cmap='hot', aspect=aspect)

    if high_snr:
        kernel_label = 'Template'
    else:
        kernel_label = 'Kernel'

    sigma_gauss = r"$\sigma_{gauss}$"
    for ax, m, label in zip(
        range(5) if plot_interference_pattern else range(4),
        [m1, m2, m3, m4, m5] if plot_interference_pattern else [m1, m2, m3, m5],
        [
            f'Inputs ({pois.shape[0]} peaks)\n[$\gamma$=.5]',
            f'{kernel_label}',
            'Peak detection',
            f'Interference',
            f'Reconstructed\n[$\gamma$=.5 {sigma_gauss}={estimated_object_gaussian_sigma}]'
        ]
        if plot_interference_pattern else [
            f'Inputs ({pois.shape[0]} peaks)\n[$\gamma$=.5]',
            f'{kernel_label}',
            'Peak detection',
            f'Reconstructed\n[$\gamma$=.5 {sigma_gauss}={estimated_object_gaussian_sigma}]'
        ]
    ):
        divider = make_axes_locatable(axes[ax, -1])
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cb = plt.colorbar(m, cax=cax)
        cax.yaxis.set_label_position("right")
        cax.set_ylabel(label)

    for ax in axes.flatten():
        ax.axis('off')

    axes[0, 0].set_title('XY')
    axes[0, 1].set_title('XZ')
    axes[0, 2].set_title('YZ')

    savesvg(fig, f'{plot}_interference_pattern.svg')



def plot_embeddings(
        inputs: np.array,
        emb: np.array,
        save_path: Any,
        gamma: float = .5,
        nrows: Optional[int] = None,
        ncols: Optional[int] = None,
        ztiles: Optional[int] = None,
        icmap: str = 'hot',
        aspect: str = 'auto'
):
    plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'axes.autolimit_mode': 'round_numbers'
    })

    step = .1
    vmin = int(np.floor(np.nanpercentile(emb[0], 1))) if np.any(emb[0] < 0) else 0
    vmax = int(np.ceil(np.nanpercentile(emb[0], 99))) if vmin < 0 else 3
    vcenter = 1 if vmin == 0 else 0

    cmap = np.vstack((
        plt.get_cmap('GnBu_r' if vmin == 0 else 'GnBu_r', 256)(
            np.linspace(0, 1 - step, int(abs(vcenter - vmin) / step))
        ),
        [1, 1, 1, 1],
        plt.get_cmap('YlOrRd' if vmax != 1 else 'OrRd', 256)(
            np.linspace(0, 1 + step, int(abs(vcenter - vmax) / step))
        )
    ))
    cmap = mcolors.ListedColormap(cmap)

    if emb.shape[0] == 3:
        fig, axes = plt.subplots(2, 3, figsize=(8, 6))
    else:
        fig, axes = plt.subplots(3, 3, figsize=(8, 8))

    if inputs.ndim == 4:
        if ncols is None or nrows is None:
            inputs = np.max(inputs, axis=0)  # show max projections of all z-tiles
            for c in range(10, 0, -1):
                if inputs.shape[0] > c and not inputs.shape[0] % c:
                    ncols = c
                    break

            nrows = inputs.shape[0] // ncols

        for proj in range(3):
            grid = gridspec.GridSpecFromSubplotSpec(
                nrows, ncols, subplot_spec=axes[0, proj], wspace=.01, hspace=.01
            )

            for idx, (i, j) in enumerate(itertools.product(range(nrows), range(ncols))):
                ax = fig.add_subplot(grid[i, j])

                try:
                    if np.max(inputs[idx], axis=None) > 0 :
                        m = ax.imshow(np.max(inputs[idx], axis=proj) ** gamma, cmap=icmap, aspect=aspect)

                except IndexError: # if we dropped a tile due to poor SNR
                    m = ax.imshow(np.zeros_like(np.max(inputs[0], axis=proj)), cmap=icmap, aspect=aspect)

                ax.axis('off')
            axes[0, proj].axis('off')

        divider = make_axes_locatable(axes[0, 0])
        cax = divider.append_axes("left", size="5%", pad=0.1)
        cb = plt.colorbar(m, cax=cax)
        cax.yaxis.set_label_position("left")
        cax.yaxis.set_ticks_position('left')
        cax.set_ylabel(rf'Input (MIP) [$\gamma$={gamma}]')
    else:
        m = axes[0, 0].imshow(np.max(inputs**gamma, axis=0), cmap=icmap, aspect=aspect)
        axes[0, 1].imshow(np.max(inputs**gamma, axis=1), cmap=icmap, aspect=aspect)
        axes[0, 2].imshow(np.max(inputs**gamma, axis=2), cmap=icmap, aspect=aspect)

        divider = make_axes_locatable(axes[0, 0])
        cax = divider.append_axes("left", size="5%", pad=0.1)
        cb = plt.colorbar(m, cax=cax)
        cax.yaxis.set_label_position("left")
        cax.yaxis.set_ticks_position('left')
        cax.set_ylabel(rf'Input (MIP) [$\gamma$={gamma}]')

    m = axes[1, 0].imshow(emb[0], cmap=cmap, vmin=vmin, vmax=vmax)
    axes[1, 1].imshow(emb[1], cmap=cmap, vmin=vmin, vmax=vmax)
    axes[1, 2].imshow(emb[2], cmap=cmap, vmin=vmin, vmax=vmax)

    divider = make_axes_locatable(axes[1, 0])
    cax = divider.append_axes("left", size="5%", pad=0.1)
    cb = plt.colorbar(m, cax=cax)
    cax.yaxis.set_label_position("left")
    cax.yaxis.set_ticks_position('left')
    cax.set_ylabel(r'Embedding ($\alpha$)')

    if emb.shape[0] > 3:
        # phase embedding limit = 95th percentile or 0.25, round to nearest 1/2 rad
        p_vmax = max(np.ceil(np.nanpercentile(np.abs(emb[3:]), 95)*2)/2, .25)
        p_vmin = -p_vmax
        p_vcenter = 0
        step = p_vmax/10

        p_cmap = np.vstack((
            plt.get_cmap('GnBu_r' if p_vmin == 0 else 'GnBu_r', 256)(
                np.linspace(0, 1, int(abs(p_vcenter - p_vmin) / step))
            ),
            [1, 1, 1, 1],
            plt.get_cmap('YlOrRd' if p_vmax == 3 else 'OrRd', 256)(
                np.linspace(0, 1, int(abs(p_vcenter - p_vmax) / step))
            )
        ))
        p_cmap = mcolors.ListedColormap(p_cmap)

        m = axes[-1, 0].imshow(emb[3], cmap=p_cmap, vmin=p_vmin, vmax=p_vmax)
        axes[-1, 1].imshow(emb[4], cmap=p_cmap, vmin=p_vmin, vmax=p_vmax)
        axes[-1, 2].imshow(emb[5], cmap=p_cmap, vmin=p_vmin, vmax=p_vmax)

        divider = make_axes_locatable(axes[-1, 0])
        cax = divider.append_axes("left", size="5%", pad=0.1)
        cb = plt.colorbar(m, cax=cax, format=lambda x, _: f"{x:.1f}")
        cax.yaxis.set_label_position("left")
        cax.yaxis.set_ticks_position('left')
        cax.set_ylabel(r'Embedding ($\varphi$, radians)')

    for ax in axes.flatten():
        ax.axis('off')

    if save_path == True:
        plt.show()
    else:
        savesvg(fig, f'{save_path}_embeddings.svg')
        # plt.savefig(f'{save_path}_embeddings.png')
