# File: plot_circles.py
#
#   This code is used with Qiskit to display the circle plot representation of the state vector
#   of a quantum register. The state vector for an N-qubit quantum register has 2^N components,
#   which are called amplitudes. Each amplitude is a complex number, and the squared magnitudes
#   add up to 1.
#
#   The circle plot has a circle of radius 1 for each amplitude, with a filled-in circle inside it.
#   The radius of the filled-in circle is equal to the magnitude. The circle also contains a line
#   segment from the center of the circle to the circumference. The line segment is rotated to show
#   the relative phase.
#
#   EXAMPLE: (Bell state)
#
#       from qiskit import QuantumCircuit, QuantumRegister, execute, BasicAer
#       from circle_plot import plot_circles
#
#       a = QuantumRegister(1, name='a')
#       b = QuantumRegister(1, name='b')
#       qc = QuantumCircuit(a, b)
#       qc.h(a)
#       qc.cx(a, b)
#       backend = BasicAer.get_backend('statevector_simulator')
#       job = execute(qc, backend)
#       result = job.result()
#       outputstate = result.get_statevector(qc, decimals=3)
#       plot_circles(outputstate)
#
#  The circle plot representation is described in "Programming Quantum Computers"
#  by Eric R. Johnson, Nic Harrigan, and Mercedes Gimeno-Segovia. It is also used
#  in the QCEngine quantum simulator that accompanies the book.
#
#  See https://oreilly-qc.github.io


import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.lines as mlines
import numpy as np

def plot_circle(axis, amplitude, index, **kwargs):
    """Generates a circle representation for a single amplitude."""
    r = abs(amplitude)
    if kwargs.get('long_phase') and r > 1e-6:
        amplitude /= r
    color = kwargs.get('color', 'red')
    fill = kwargs.get('fill', True)
    axis.set_aspect(1)
    axis.set_xlim(-1.25, 1.25)
    axis.set_ylim(-1.5, 1.25)
    axis.axis('off')
    outer_circle = plt.Circle((0, 0), 1, color='black', fill=False)
    inner_circle = plt.Circle((0, 0), r, color=color, fill=fill)
    radius = mlines.Line2D([0, -np.imag(amplitude)], [0, np.real(amplitude)], color='black')
    axis.add_artist(inner_circle)
    axis.add_artist(outer_circle)
    axis.add_artist(radius)
    axis.text(-0.2, -1.5, f'|{index}\u27e9')


def plot_circles(state_vector, nrows=1, **kwargs):
    """Generates the circle-plot representation for the state vector of a quantum register."""
    fontsize = kwargs.get('fontsize', 12)
    matplotlib.rc('font', family='serif', weight='bold', size=fontsize)
    N = len(state_vector)
    assert 0 < nrows and N % nrows == 0, 'invalid value of nrows'
    ncols = N // nrows
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(2 * ncols, nrows + 1))
    for index in range(N):
        if nrows == 1 or ncols == 1:
            axis = axs[index]
        else:
            row = index // ncols
            col = index % ncols
            axis = axs[row][col]

        amplitude = state_vector[index]
        plot_circle(axis, amplitude, index, **kwargs)
    plt.show()

