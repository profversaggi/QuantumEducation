# type: ignore

import ipywidgets as widgets
from qiskit import QuantumCircuit, Aer, execute
from qiskit.visualization import plot_bloch_multivector
from IPython.display import display, clear_output, HTML, Code
import numpy as np

from may4_challenge_common import vec_in_braket
from qsphere_plotter import plot_state_qsphere


def vec_in_text_braket(vec):
    return '$$\\text{{State:\n $|\\Psi\\rangle = $}}{}$$'.format(vec_in_braket(vec))


def minicomposer(nqubits=5, bloch=False, dirac=False, qsphere=False):
    out = widgets.Output
    single_gates = ['I', 'X', 'Y', 'Z', 'H', 'T', 'Tdg', 'S', 'Sdg']
    multi_gates = ['CX', 'CZ', 'SWAP']
    qc = QuantumCircuit(nqubits)
    if bloch or dirac or qsphere:
        backend = Aer.get_backend('statevector_simulator')

    class CircuitWidget:
        def __init__(self):
            self.waiting_for = 'gate'
            self.current_gate = ''
            self.qubits = ['']
            self.code = ""
            self.statevec = []

    widget_state = CircuitWidget()
    cell_pretext = """def create_circuit():\n    qc = QuantumCircuit({})\n""".format(nqubits)
    cell_ending = "    return qc"

    def on_sqg_click(btn):
        """On single-qubit-gate button click"""
        if widget_state.waiting_for == 'gate':
            widget_state.waiting_for = 'sqg_qubit'
            update_output()
            widget_state.current_gate = btn.description

    def on_mqg_click(btn):
        """On multi-qubit-gate button click"""
        if widget_state.waiting_for == 'gate':
            widget_state.waiting_for = 'mqg_qubit_0'
            update_output()
            widget_state.current_gate = btn.description

    def on_qubit_click(btn):
        """On qubit button click"""
        if widget_state.waiting_for == 'sqg_qubit':
            widget_state.qubits[0] = int(btn.description)
            apply_gate()
            widget_state.waiting_for = 'gate'
            update_output()
        elif widget_state.waiting_for == 'mqg_qubit_0':
            widget_state.qubits[0] = int(btn.description)
            widget_state.waiting_for = 'mqg_qubit_1'
            update_output()
        elif widget_state.waiting_for == 'mqg_qubit_1':
            widget_state.qubits.append(int(btn.description))
            widget_state.waiting_for = 'gate'
            apply_gate()
            update_output()

    def on_clear_click(btn):
        """On Clear button click"""
        widget_state.current_gate = 'Clear'
        widget_state.waiting_for = 'gate'
        apply_gate()
        update_output()

    def apply_gate():
        """Uses widget_state to apply the last selected gate, update
        the code cell and prepare widget_state for the next selection"""
        functionmap = {
            'I': 'qc.iden',
            'X': 'qc.x',
            'Y': 'qc.y',
            'Z': 'qc.z',
            'H': 'qc.h',
            'S': 'qc.s',
            'T': 'qc.t',
            'Sdg': 'qc.sdg',
            'Tdg': 'qc.tdg',
            'CX': 'qc.cx',
            'CZ': 'qc.cz',
            'SWAP': 'qc.swap'
        }
        gate = widget_state.current_gate
        qubits = widget_state.qubits
        widget_state.code += "    "
        if len(qubits) == 2:
            widget_state.code += functionmap[gate]
            widget_state.code += "({0}, {1})\n".format(qubits[0], qubits[1])
            widget_state.qubits.pop()
        elif widget_state.current_gate == 'Clear':
            widget_state.code = ""
        else:
            widget_state.code += functionmap[gate] + "({})\n".format(qubits[0])
        qc = QuantumCircuit(nqubits)
        # This is especially awful I know, please don't judge me
        exec(widget_state.code.replace("    ", ""))
        qc.draw('mpl').savefig('circuit_widget_temp.svg', format='svg')
        if bloch or dirac or qsphere:
            ket = execute(qc, backend).result().get_statevector()
            if bloch:
                plot_bloch_multivector(
                    ket, show_state_labels=True).savefig('circuit_widget_temp_bs.svg', format='svg')
            if qsphere:
                plot_state_qsphere(
                    ket, show_state_labels=True).savefig('circuit_widget_temp_qs.svg', format='svg')
            if dirac:
                widget_state.statevec = ket

    # Create buttons for single qubit gates
    sqg_btns = [widgets.Button(description=gate) for gate in single_gates]
    # Link these to the on_sqg_click function
    for button in sqg_btns:
        button.on_click(on_sqg_click)
    # Create buttons for qubits
    qubit_btns = [widgets.Button(description=str(qubit)) for qubit in range(nqubits)]
    # Register these too
    for button in qubit_btns:
        button.on_click(on_qubit_click)
    # Create & register buttons for multi-qubit gates, clear
    mqg_btns = [widgets.Button(description=gate) for gate in multi_gates]
    for button in mqg_btns:
        button.on_click(on_mqg_click)
    clear_btn = widgets.Button(description="Clear")
    clear_btn.on_click(on_clear_click)

    instruction = widgets.Label(value="Select a gate to add to the circuit:")
    qc.draw('mpl').savefig('circuit_widget_temp.svg', format='svg')
    if bloch or dirac or qsphere:
        ket = execute(qc, backend).result().get_statevector()
        if bloch:
            plot_bloch_multivector(ket).savefig('circuit_widget_temp_bs.svg', format='svg')
            with open('circuit_widget_temp_bs.svg', 'r') as img:
                bloch_sphere = widgets.HTML(value=img.read())
        if qsphere:
            plot_state_qsphere(ket).savefig('circuit_widget_temp_qs.svg', format='svg')
            with open('circuit_widget_temp_qs.svg', 'r') as img:
                qsphere = widgets.HTML(value=img.read())
        if dirac:
            widget_state.statevec = ket
            latex_statevec = widgets.HTMLMath(vec_in_text_braket(ket))

    qiskit_code = widgets.HTML(value='')

    with open('circuit_widget_temp.svg', 'r') as img:
        drawing = widgets.HTML(value=img.read())

    def display_widget():
        sqg_box = widgets.HBox(sqg_btns)
        mqg_box = widgets.HBox(mqg_btns+[clear_btn])
        qubit_box = widgets.HBox(qubit_btns)
        main_box = widgets.VBox([sqg_box, mqg_box, qubit_box])
        visuals = [drawing]
        if bloch:
            visuals.append(bloch_sphere)
        if qsphere:
            visuals.append(qsphere)
        if dirac:
            visuals.append(latex_statevec)
        vis_box = widgets.VBox(visuals)
        display(instruction, main_box, vis_box)
        display(qiskit_code)

    def update_output():
        """Changes the availability of buttons depending on the state
        of widget_state.waiting_for, updates displayed image"""
        if widget_state.waiting_for == 'gate':
            for button in sqg_btns:
                button.disabled = False
            for button in mqg_btns:
                if nqubits > 1:
                    button.disabled = False
                else:
                    button.disabled = True
            for button in qubit_btns:
                button.disabled = True
            instruction.value = "Select a gate to add to the circuit:"
        else:
            for button in sqg_btns:
                button.disabled = True
            for button in mqg_btns:
                button.disabled = True
            for button in qubit_btns:
                button.disabled = False
            if widget_state.waiting_for == 'sqg_qubit':
                instruction.value = "Select a qubit to perform the gate on:"
            elif widget_state.waiting_for == 'mqg_qubit_0':
                instruction.value = "Select the control qubit:"
            elif widget_state.waiting_for == 'mqg_qubit_1':
                instruction.value = "Select the target qubit:"
                qubit_btns[widget_state.qubits[0]].disabled = True
        with open('circuit_widget_temp.svg', 'r') as img:
            drawing.value = img.read()
        if bloch:
            with open('circuit_widget_temp_bs.svg', 'r') as img:
                bloch_sphere.value = img.read()
        if qsphere:
            with open('circuit_widget_temp_qs.svg', 'r') as img:
                qsphere.value = img.read()
        if dirac:
            latex_statevec.value = vec_in_text_braket(widget_state.statevec)

        complete_code = cell_pretext + widget_state.code + cell_ending
        qiskit_code.value = f"""
            <div class="output_html" style="line-height: 1.21429em; font-size: 14px">
                {Code(complete_code, language='python')._repr_html_()}
            </div>
        """

    display_widget()
    update_output()
