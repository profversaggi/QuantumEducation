import json
import os
from typing import Any, Mapping

import numpy as np
from scipy.stats import unitary_group


import qiskit.quantum_info as qi
from qiskit import QuantumCircuit, Aer, execute, assemble
from qiskit.assembler import disassemble
from qiskit.qobj import QasmQobj

EX1_SUBEXERCISE_COUNT = 8
EX4_N = 4
EX4_REFERENCE_UNITARY: np.ndarray = np.load(
    os.path.join(os.path.dirname(__file__), 'U.npy'))


def bloch_vec(qc: QuantumCircuit) -> np.ndarray:
    backend = Aer.get_backend('statevector_simulator')
    ket = without_global_phase(execute(qc, backend).result().get_statevector())
    if ket[0] != 0:
        theta = 2*np.arctan(np.abs(ket[1]/ket[0]))
        phi = np.angle(ket[1]/ket[0])
    else:
        theta = np.pi
        phi = 0
    bloch_vector = np.round([
        np.sin(theta)*np.cos(phi),
        np.sin(theta)*np.sin(phi),
        np.cos(theta)], 5)
    return bloch_vector


def vec_in_braket(vec: np.ndarray) -> str:
    nqubits = int(np.log2(len(vec)))
    state = ''
    for i in range(len(vec)):
        rounded = round(vec[i], 3)
        if rounded != 0:
            basis = format(i, 'b').zfill(nqubits)
            state += np.str(rounded).replace('-0j', '+0j')
            state += '|' + basis + '\\rangle + '
    state = state.replace("j", "i")
    return state[0:-2].strip()


def statevec(qc: QuantumCircuit) -> qi.Statevector:
    return qi.Statevector.from_instruction(qc)


def return_state(qc: QuantumCircuit) -> str:
    return vec_in_braket(without_global_phase(statevec(qc)))


def circuit_to_dict(qc: QuantumCircuit) -> dict:
    qobj = assemble(qc)
    return qobj.to_dict()


def circuit_to_json(qc: QuantumCircuit) -> str:
    class _QobjEncoder(json.encoder.JSONEncoder):
        def default(self, obj: Any) -> Any:
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, complex):
                return (obj.real, obj.imag)
            return json.JSONEncoder.default(self, obj)

    return json.dumps(circuit_to_dict(qc), cls=_QobjEncoder)


def dict_to_circuit(dict_: dict) -> QuantumCircuit:
    qobj = QasmQobj.from_dict(dict_)
    return disassemble(qobj)[0][0]


def json_to_circuit(json_: str) -> QuantumCircuit:
    return dict_to_circuit(json.loads(json_))


def without_global_phase(matrix: np.ndarray, atol: float = 1e-8) -> np.ndarray:
    phases1 = np.angle(matrix[abs(matrix) > atol].ravel(order='F'))
    if len(phases1) > 0:
        matrix = np.exp(-1j * phases1[0]) * matrix
    return matrix
