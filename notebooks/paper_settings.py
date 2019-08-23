from matplotlib import pyplot as plt
from matplotlib import gridspec


COLOR_FULL = "#B3004A"
COLOR_ALIGN = "#5B4CFF"
COLOR_MASS = "#4CBFAC"
COLOR_FIX = "#B7B7B7"

COLOR_BKG = "#B7B7B7"

def figure(cbar=False, height=4.0,  large_margin=0.15, small_margin=0.04, cbar_sep=0.04, cbar_width=0.04):
    if cbar:
        width = height * (1. + cbar_sep + cbar_width + large_margin - small_margin)
        top = small_margin
        bottom = large_margin
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
        bottom = large_margin

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



def grid2(nx=4, ny=2, height=6., large_margin=0.15, small_margin=0.03, sep=0.03, cbar_width=0.04):
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
