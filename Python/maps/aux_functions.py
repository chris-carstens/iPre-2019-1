# aux_functions:
# - Python version: 
# - Author: Mauro S. Mendoza Elguera
# - Date: 2019-05-14

from mpl_toolkits.axes_grid1 import make_axes_locatable


def colorbar(mappable):
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right",
                              size="5%",
                              pad=.6)

    return fig.colorbar(mappable,
                        cax=cax,
                        shrink=.4,
                        aspect=10)
