import matplotlib
from matplotlib import pyplot as plt
from matplotlib import gridspec

import palettable

#CMAP1 = palettable.cartocolors.sequential.Teal_7.mpl_colormap
#CMAP2 = palettable.cartocolors.sequential.Teal_7.mpl_colormap
#COLORS = palettable.cartocolors.sequential.Teal_4_r.mpl_colors

TEXTWIDTH = 7.1014  # inches

CMAP1 = palettable.cmocean.sequential.Ice_20_r.mpl_colormap
CMAP2 = palettable.cmocean.sequential.Ice_20.mpl_colormap
COLORS = palettable.cmocean.sequential.Ice_6_r.mpl_colors
COLOR_FULL = COLORS[4]  # "#B3004A"
COLOR_MASS = COLORS[3]  # "#4CBFAC"
COLOR_ALIGN = COLORS[2]  # "#5B4CFF"
COLOR_FIX = COLORS[1]  # "#B7B7B7"
COLOR_BKG = "0.7"  # "#B7B7B7"


def setup():
    matplotlib.rcParams.update({'text.usetex': True, 'font.size': 10, 'font.family': 'serif'})
    params= {'text.latex.preamble' : [r'\usepackage{amsmath}', r'\usepackage{amssymb}']}
    plt.rcParams.update(params)


def figure(cbar=False, height=TEXTWIDTH*0.4,  large_margin=0.18, mid_margin=0.14, small_margin=0.05, cbar_sep=0.03, cbar_width=0.04):
    if cbar:
        width = height * (1. + cbar_sep + cbar_width + large_margin - small_margin)
        top = small_margin
        bottom = mid_margin
        left = large_margin
        right = large_margin + cbar_width + cbar_sep
        cleft = 1. - (large_margin + cbar_width) * height / width
        cbottom = bottom
        cwidth = cbar_width * height / width
        cheight = 1. - top - bottom

        fig = plt.figure(figsize=(width, height))
        ax = plt.gca()
        plt.subplots_adjust(
            left=left * height / width,
            right=1. - right * height / width,
            bottom=bottom,
            top=1. - top,
            wspace=0.,
            hspace=0.,
        )
        cax = fig.add_axes([cleft, cbottom, cwidth, cheight])

        plt.sca(ax)

        return fig, (ax, cax)
    else:
        width = height
        left = large_margin
        right = small_margin
        top = small_margin
        bottom = mid_margin

        fig = plt.figure(figsize=(width, height))
        ax = plt.gca()
        plt.subplots_adjust(
            left=left,
            right=1. - right,
            bottom=bottom,
            top=1. - top,
            wspace=0.,
            hspace=0.,
        )

        return fig, ax


def grid(nx=4, ny=2, height=6., n_caxes=0, large_margin=0.02, small_margin=0.02, sep=0.02, cbar_width=0.03):
    # Geometry (in multiples of height)
    left = large_margin
    right = small_margin
    top = small_margin
    bottom = large_margin
    panel_size = (1. - top - bottom - (ny - 1)*sep)/ny

    # Absolute width
    width = height*(left + nx*panel_size+ (nx-1)*sep + right)

    # wspace and hspace are complicated beasts
    avg_width_abs = (height*panel_size * nx * ny + n_caxes * cbar_width * height) / (nx * ny + n_caxes)
    avg_height_abs = height*panel_size
    wspace = sep * height / avg_width_abs
    hspace = sep * height / avg_height_abs

    # Set up figure
    fig = plt.figure(figsize=(width, height))
    plt.subplots_adjust(
        left=left * height / width,
        right=1. - right * height / width,
        bottom=bottom,
        top=1. - top,
        wspace=wspace,
        hspace=hspace,
    )

    # Colorbar axes in last panel
    caxes = []
    if n_caxes > 0:
        ax = plt.subplot(ny, nx, nx*ny)
        ax.axis("off")
        pos = ax.get_position()
        cax_total_width=pos.width / n_caxes
        cbar_width_ = cbar_width * height / width
        for i in range(n_caxes):
            cax = fig.add_axes([pos.x0 + i * cax_total_width, pos.y0, cbar_width_, pos.height])
            cax.yaxis.set_ticks_position('right')
            caxes.append(cax)

    return fig, caxes


def grid_width(nx=4, ny=2, width=TEXTWIDTH, n_caxes=0, large_margin=0.025, small_margin=0.025, sep=0.025, cbar_width=0.04):
    left = large_margin
    right = small_margin
    top = small_margin
    bottom = large_margin
    panel_size = (1. - top - bottom - (ny - 1) * sep) / ny
    height = width / (left + nx * panel_size + (nx - 1) * sep + right)
    return grid(nx, ny, height, n_caxes, large_margin, small_margin, sep, cbar_width)


def grid2(nx=4, ny=2, height=6., large_margin=0.14, small_margin=0.03, sep=0.03, cbar_width=0.06):
    # Geometry
    left = large_margin
    right = large_margin
    top = small_margin
    bottom = large_margin

    panel_size = (1. - top - bottom - (ny - 1)*sep)/ny
    width = height*(left + nx*panel_size + cbar_width + nx*sep + right)

    # wspace and hspace are complicated beasts
    avg_width_abs = (height*panel_size * nx * ny + ny * cbar_width * height) / (nx * ny + ny)
    avg_height_abs = height*panel_size
    wspace = sep * height / avg_width_abs
    hspace = sep * height / avg_height_abs

    # Set up figure
    fig = plt.figure(figsize=(width, height))
    gs = gridspec.GridSpec(ny, nx + 1, width_ratios=[1.]*nx + [cbar_width], height_ratios=[1.] * ny)
    plt.subplots_adjust(
        left=left * height / width,
        right=1. - right * height / width,
        bottom=bottom,
        top=1. - top,
        wspace=wspace,
        hspace=hspace,
    )
    return fig, gs


def grid2_width(nx=4, ny=2, width=TEXTWIDTH, large_margin=0.14, small_margin=0.03, sep=0.03, cbar_width=0.06):
    left = large_margin
    right = large_margin
    top = small_margin
    bottom = large_margin
    panel_size = (1. - top - bottom - (ny - 1)*sep)/ny
    height = width / (left + nx*panel_size + cbar_width + nx*sep + right)
    return grid2(nx, ny, height, large_margin, small_margin, sep, cbar_width)




def two_figures(height=TEXTWIDTH*0.4,  large_margin=0.18, small_margin=0.05, sep=0.21,):
    # Geometry (in multiples of height)
    left = large_margin
    right = small_margin
    top = small_margin
    bottom = large_margin
    panel_size = 1. - top - bottom

    # Absolute width
    width = height*(left + 2*panel_size+ sep + right)

    # wspace and hspace are complicated beasts
    avg_width_abs = height*panel_size
    avg_height_abs = height*panel_size
    wspace = sep * height / avg_width_abs
    hspace = sep * height / avg_height_abs

    # Set up figure
    fig = plt.figure(figsize=(width, height))
    plt.subplots_adjust(
        left=left * height / width,
        right=1. - right * height / width,
        bottom=bottom,
        top=1. - top,
        wspace=wspace,
        hspace=hspace,
    )

    ax_left = plt.subplot(1,2,1)
    ax_right = plt.subplot(1,2,2)

    return fig, ax_left, ax_right


def animated_special(height=TEXTWIDTH*0.4, large_margin=0.18, mid_margin=0.14, small_margin=0.05):
    # Geometry (in multiples of height)
    left = large_margin
    right = small_margin
    top = small_margin
    bottom = large_margin
    panel_size = 1. - top - bottom
    sep1 = small_margin + large_margin
    sep2 = small_margin * 2

    # Absolute width
    width = height*(left + 3*panel_size + sep1 + sep2 + right)

    # Set up figure
    fig = plt.figure(figsize=(width, height))

    # Two left axes
    ax_left = fig.add_axes([left*height/width, bottom, panel_size*height/width, panel_size])
    ax_middle = fig.add_axes([(left + panel_size + sep1)*height/width, bottom, panel_size*height/width, panel_size])

    # Space for images
    images_left = (left + 2*panel_size + sep1 + sep2)*height/width
    images_bottom = bottom
    images_width = panel_size*height/width
    images_height = panel_size

    return fig, ax_left, ax_middle, (images_left, images_bottom, images_width, images_height)


def add_image_to_roster(fig, axes, total_coords, sep_fraction=0.08):
    total_left, total_bottom, total_width, total_height = total_coords

    n = len(axes)
    rearrange_all = n in [int(x**2) for x in range(2,100)]
    n_side = max(int(n**0.5) + 1, 2)

    def _coords(i):
        ix = i % n_side
        iy = i // n_side

        panel_width = total_width / (n_side + (n_side - 1) * sep_fraction)
        left = total_left + ix * panel_width * (1.0 + sep_fraction)
        width = panel_width
        panel_height = total_height / (n_side + (n_side - 1) * sep_fraction)
        bottom = total_bottom + (n_side - iy - 1) * panel_height * (1.0 + sep_fraction)
        height = panel_height

        return [left, bottom, width, height]

    if rearrange_all:
        axes_new = []
        for i, ax in enumerate(axes):
            axes_new.append(fig.add_axes(_coords(i)))
            axes_new[-1].axis('off')
    else:
        axes_new = axes
    axes_new.append(fig.add_axes(_coords(n)))
    axes_new[-1].axis('off')
    return axes_new
