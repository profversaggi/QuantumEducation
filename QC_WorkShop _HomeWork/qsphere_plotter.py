# type: ignore

import colorsys
from functools import reduce

import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d import proj3d
from matplotlib import get_backend


class Arrow3D(FancyArrowPatch):
    """Standard 3D arrow."""

    def __init__(self, xs, ys, zs, *args, **kwargs):
        """Create arrow."""
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        """Draw the arrow."""
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, _ = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)


def n_choose_k(n, k):
    """Return the number of combinations for n choose k.

    Args:
        n (int): the total number of options .
        k (int): The number of elements.

    Returns:
        int: returns the binomial coefficient
    """
    if n == 0:
        return 0
    return reduce(lambda x, y: x * y[0] / y[1],
                  zip(range(n - k + 1, n + 1),
                      range(1, k + 1)), 1)


def lex_index(n, k, lst):
    """Return  the lex index of a combination..

    Args:
        n (int): the total number of options .
        k (int): The number of elements.
        lst (list): list

    Returns:
        int: returns int index for lex order

    Raises:
        VisualizationError: if length of list is not equal to k
    """
    if len(lst) != k:
        raise VisualizationError("list should have length k")
    comb = list(map(lambda x: n - 1 - x, lst))
    dualm = sum([n_choose_k(comb[k - 1 - i], i + 1) for i in range(k)])
    return int(dualm)


def _validate_input_state(quantum_state):
    """Validates the input to state visualization functions.

    Args:
        quantum_state (ndarray): Input state / density matrix.
    Returns:
        rho: A 2d numpy array for the density matrix.
    Raises:
        VisualizationError: Invalid input.
    """
    rho = np.asarray(quantum_state)
    if rho.ndim == 1:
        rho = np.outer(rho, np.conj(rho))
    # Check the shape of the input is a square matrix
    shape = np.shape(rho)
    if len(shape) != 2 or shape[0] != shape[1]:
        raise VisualizationError("Input is not a valid quantum state.")
    # Check state is an n-qubit state
    num = int(np.log2(rho.shape[0]))
    if 2 ** num != rho.shape[0]:
        raise VisualizationError("Input is not a multi-qubit quantum state.")
    return rho


def bit_string_index(s):
    """Return the index of a string of 0s and 1s."""
    n = len(s)
    k = s.count("1")
    if s.count("0") != n - k:
        raise VisualizationError("s must be a string of 0 and 1")
    ones = [pos for pos, char in enumerate(s) if char == "1"]
    return lex_index(n, k, ones)


def phase_to_rgb(complex_number):
    """Map a phase of a complexnumber to a color in (r,g,b).

    complex_number is phase is first mapped to angle in the range
    [0, 2pi] and then to the HSL color wheel
    """
    angles = (np.angle(complex_number) + (np.pi * 4)) % (np.pi * 2)
    rgb = colorsys.hls_to_rgb(angles / (np.pi * 2), 0.5, 0.5)
    return rgb


def plot_state_qsphere(rho, figsize=None, ax=None, show_state_labels=False,
                       show_state_angles=False, force_angle=False):
    """Plot the qsphere representation of a quantum state.
    Here, the size of the points is proportional to the probability
    of the corresponding term in the state and the color represents
    the phase.

    Args:
        rho (ndarray): State vector or density matrix representation.
            of quantum state.
        figsize (tuple): Figure size in inches.
        ax (matplotlib.axes.Axes): An optional Axes object to be used for
            the visualization output. If none is specified a new matplotlib
            Figure will be created and used. Additionally, if specified there
            will be no returned Figure since it is redundant.

    Returns:
        Figure: A matplotlib figure instance if the ``ax`` kwag is not set

    Raises:
        ImportError: Requires matplotlib.

    Example:
        .. jupyter-execute::

           from qiskit import QuantumCircuit, BasicAer, execute
           from qiskit.visualization import plot_state_qsphere
           %matplotlib inline

           qc = QuantumCircuit(2, 2)
           qc.h(0)
           qc.cx(0, 1)
           qc.measure([0, 1], [0, 1])

           backend = BasicAer.get_backend('statevector_simulator')
           job = execute(qc, backend).result()
           plot_state_qsphere(job.get_statevector(qc))
    """
    try:
        import seaborn as sns
    except ImportError:
        raise ImportError('Must have seaborn installed to use '
                          'plot_state_qsphere. To install, run "pip install seaborn".')
    rho = _validate_input_state(rho)
    if figsize is None:
        figsize = (7, 7)
    num = int(np.log2(len(rho)))

    # get the eigenvectors and eigenvalues
    we, stateall = linalg.eigh(rho)

    if ax is None:
        return_fig = True
        fig = plt.figure(figsize=figsize)
    else:
        return_fig = False
        fig = ax.get_figure()

    gs = gridspec.GridSpec(nrows=3, ncols=3)

    ax = fig.add_subplot(gs[0:3, 0:3], projection='3d')
    ax.axes.set_xlim3d(-1.0, 1.0)
    ax.axes.set_ylim3d(-1.0, 1.0)
    ax.axes.set_zlim3d(-1.0, 1.0)
    ax.axes.grid(False)
    ax.view_init(elev=5, azim=275)

    for _ in range(2 ** num):
        # start with the max
        probmix = we.max()
        prob_location = we.argmax()
        if probmix > 0.001:
            # get the max eigenvalue
            state = stateall[:, prob_location]
            loc = np.absolute(state).argmax()

            # get the element location closes to lowest bin representation.
            for j in range(2 ** num):
                test = np.absolute(np.absolute(state[j]) -
                                   np.absolute(state[loc]))
                if test < 0.001:
                    loc = j
                    break

            # remove the global phase
            angles = (np.angle(state[loc]) + 2 * np.pi) % (2 * np.pi)
            angleset = np.exp(-1j * angles)
            state = angleset * state
            state.flatten()

            # start the plotting
            # Plot semi-transparent sphere
            u = np.linspace(0, 2 * np.pi, 25)
            v = np.linspace(0, np.pi, 25)
            x = np.outer(np.cos(u), np.sin(v))
            y = np.outer(np.sin(u), np.sin(v))
            z = np.outer(np.ones(np.size(u)), np.cos(v))
            ax.plot_surface(x, y, z, rstride=1, cstride=1, color='k',
                            alpha=0.05, linewidth=0)

            # Get rid of the panes
            ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

            # Get rid of the spines
            ax.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
            ax.w_yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
            ax.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))

            # Get rid of the ticks
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])

            d = num
            for i in range(2 ** num):

                # get x,y,z points
                element = bin(i)[2:].zfill(num)
                weight = element.count("1")
                zvalue = -2 * weight / d + 1
                number_of_divisions = n_choose_k(d, weight)
                weight_order = bit_string_index(element)
                angle = (float(weight) / d) * (np.pi * 2) + \
                        (weight_order * 2 * (np.pi / number_of_divisions))

                if (weight > d / 2) or (((weight == d / 2) and
                                         (weight_order >= number_of_divisions / 2))):
                    angle = np.pi - angle - (2 * np.pi / number_of_divisions)

                if force_angle:
                    angle = (360 / number_of_divisions) * weight_order
                    angle = angle * np.pi/180

                xvalue = np.sqrt(1 - zvalue ** 2) * np.cos(angle)
                yvalue = np.sqrt(1 - zvalue ** 2) * np.sin(angle)

                # get prob and angle - prob will be shade and angle color
                prob = np.real(np.dot(state[i], state[i].conj()))
                colorstate = phase_to_rgb(state[i])

                alfa = 1
                if yvalue >= 0.1:
                    alfa = 1.0 - yvalue

                if prob > 0 and show_state_labels:
                    rprime = 1.3
                    angle_theta = np.arctan2(np.sqrt(1 - zvalue ** 2), zvalue)
                    xvalue_text = rprime * np.sin(angle_theta) * np.cos(angle)
                    yvalue_text = rprime * np.sin(angle_theta) * np.sin(angle)
                    zvalue_text = rprime * np.cos(angle_theta)
                    element_text = '$\\vert' + element + '\\rangle$'
                    if show_state_angles:
                        element_angle = (np.angle(state[i]) + (np.pi * 4)) % (np.pi * 2)
                        element_text += '\n$%.1f^\\circ$' % (element_angle * 180/np.pi)
                    ax.text(xvalue_text, yvalue_text, zvalue_text, element_text,
                            ha='center', va='center', size=12)

                ax.plot([xvalue], [yvalue], [zvalue],
                        markerfacecolor=colorstate,
                        markeredgecolor=colorstate,
                        marker='o', markersize=np.sqrt(prob) * 30, alpha=alfa)

                a = Arrow3D([0, xvalue], [0, yvalue], [0, zvalue],
                            mutation_scale=20, alpha=prob, arrowstyle="-",
                            color=colorstate, lw=2)

                ax.add_artist(a)

            # add weight lines
            for weight in range(d + 1):
                theta = np.linspace(-2 * np.pi, 2 * np.pi, 100)
                z = -2 * weight / d + 1
                r = np.sqrt(1 - z ** 2)
                x = r * np.cos(theta)
                y = r * np.sin(theta)
                ax.plot(x, y, z, color=(.5, .5, .5), lw=1, ls=':', alpha=.5)

            # add center point
            ax.plot([0], [0], [0], markerfacecolor=(.5, .5, .5),
                    markeredgecolor=(.5, .5, .5), marker='o', markersize=3,
                    alpha=1)
            we[prob_location] = 0
        else:
            break

    n = 32
    theta = np.ones(n)

    ax2 = fig.add_subplot(gs[2:, 2:])
    ax2.pie(theta, colors=sns.color_palette("hls", n), radius=0.75)
    ax2.add_artist(Circle((0, 0), 0.5, color='white', zorder=1))
    ax2.text(0, 0, 'Phase\n(Deg)', horizontalalignment='center',
             verticalalignment='center', fontsize=14)

    offset = 0.95  # since radius of sphere is one.

    ax2.text(offset, 0, '$0$', horizontalalignment='center',
             verticalalignment='center', fontsize=14)
    ax2.text(0, offset, '$90$', horizontalalignment='center',
             verticalalignment='center', fontsize=14)

    ax2.text(-offset, 0, '$180$', horizontalalignment='center',
             verticalalignment='center', fontsize=14)

    ax2.text(0, -offset, '$270$', horizontalalignment='center',
             verticalalignment='center', fontsize=14)

    if return_fig:
        if get_backend() in ['module://ipykernel.pylab.backend_inline',
                             'nbAgg']:
            plt.close(fig)
        return fig
