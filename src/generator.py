import pennylane as qml
import numpy as np
from pennylane import transforms

# setup
n_qubits = 2
dev = qml.device("default.qubit", wires=n_qubits, shots=1000)

def generator_ansatz(params):
    """Parameterized quantum circuit with entanglement"""
    for i in range(n_qubits):
        qml.RY(params[i][0], wires=i)
        qml.RZ(params[i][1], wires=i)
    for i in range(n_qubits - 1):
        qml.CNOT(wires=[i, i + 1])

@qml.qnode(dev)
def generate_sample(params):
    generator_ansatz(params)
    return qml.sample(wires=range(n_qubits))

def get_real_space_samples(params, shots=1000):
    """converts quantum bitstring outputs to real-valued 2D points"""
    dev.shots = shots
    raw_samples = generate_sample(params)
    if len(raw_samples.shape) == 1:
        raw_samples = raw_samples.reshape(1, -1)

    # Map bitstrings to real values: 00→[-1,-1], 01→[-1,1], etc.
    mapping = {
        (0, 0): [-1, -1],
        (0, 1): [-1, 1],
        (1, 0): [1, -1],
        (1, 1): [1, 1],
    }

    real_samples = np.array([mapping[tuple(b)] for b in raw_samples])
    return real_samples

# initialization
def initialize_params(seed=None):
    if seed:
        np.random.seed(seed)
    return np.random.uniform(0, 2 * np.pi, size=(n_qubits, 2))
