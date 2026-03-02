import math
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector


class QuantumLayer(nn.Module):
    """
    Quantum layer implemented with Qiskit Machine Learning.

    Circuit structure per block:
        RZ-RY-RZ (trainable) → circular CNOT
        → RZ-RY-RZ (trainable) → circular CNOT
        (+ RZ data re-encoding between blocks)

    Measurement: <Z> expectation value on each qubit independently.
    """

    def __init__(self, n_wires: int, n_blocks: int):
        super().__init__()
        self.n_wires = n_wires
        self.n_blocks = n_blocks

        # --- build parameterized quantum circuit ---
        input_params = ParameterVector('x', n_wires)
        weight_params = ParameterVector('θ', n_blocks * 6 * n_wires)

        qc = QuantumCircuit(n_wires)
        p = 0  # weight parameter index

        for k in range(n_blocks):
            # RZ-RY-RZ trainable block 1  (mirrors rx1/ry1/rz1 layers)
            for i in range(n_wires):
                qc.rz(weight_params[p + i], i)
            p += n_wires
            for i in range(n_wires):
                qc.ry(weight_params[p + i], i)
            p += n_wires
            for i in range(n_wires):
                qc.rz(weight_params[p + i], i)
            p += n_wires

            # circular CNOT entanglement  (cnot_layers[k], first call)
            for i in range(n_wires):
                qc.cx(i, (i + 1) % n_wires)

            # RZ-RY-RZ trainable block 2  (mirrors rx2/ry2/rz2 layers)
            for i in range(n_wires):
                qc.rz(weight_params[p + i], i)
            p += n_wires
            for i in range(n_wires):
                qc.ry(weight_params[p + i], i)
            p += n_wires
            for i in range(n_wires):
                qc.rz(weight_params[p + i], i)
            p += n_wires

            # circular CNOT entanglement  (cnot_layers[k], second call)
            for i in range(n_wires):
                qc.cx(i, (i + 1) % n_wires)

            # data re-encoding between blocks (skipped after the last block)
            if k < n_blocks - 1:
                for i in range(n_wires):
                    qc.rz(input_params[i], i)

        # PauliZ observable on each qubit independently.
        # Qiskit uses little-endian qubit ordering: qubit 0 is the rightmost
        # character in a Pauli string, so 'IZ' measures qubit 0 on a 2-qubit circuit.
        observables = [
            SparsePauliOp('I' * (n_wires - 1 - i) + 'Z' + 'I' * i)
            for i in range(n_wires)
        ]

        qnn = EstimatorQNN(
            circuit=qc,
            input_params=list(input_params),
            weight_params=list(weight_params),
            observables=observables,
        )
        self.quantum = TorchConnector(qnn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor of shape (..., n_wires)  — last dim carries qubit features.
        Returns:
            tensor of the same shape with PauliZ expectation values per qubit.
        """
        shape = x.shape
        x_flat = x.reshape(-1, self.n_wires)   # (N, n_wires)
        out = self.quantum(x_flat)              # (N, n_wires)
        return out.reshape(shape)


class HybridLayer(nn.Module):
    def __init__(self, in_features, hidden_features, spectrum_layer):
        super().__init__()
        self.clayer = nn.Linear(in_features, hidden_features)
        self.norm = nn.BatchNorm1d(hidden_features)
        self.qlayer1 = QuantumLayer(hidden_features, spectrum_layer)

    def forward(self, x):
        x = self.clayer(x)
        x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.qlayer1(x)
        return x


class Hybridren(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, spectrum_layer):
        super().__init__()

        self.net = []
        self.net.append(HybridLayer(in_features, hidden_features, spectrum_layer))
        final_linear = nn.Linear(hidden_features, out_features)
        self.net.append(final_linear)
        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        coords = torch.unsqueeze(coords, dim=0)
        coords = coords.clone().requires_grad_(True)
        output = nn.Sigmoid()(self.net(coords).squeeze(dim=0))
        return output


# ---------------------------------------------------------------------------
# Classical SIREN baseline (unchanged)
# ---------------------------------------------------------------------------

def exists(val):
    return val is not None


def cast_tuple(val, repeat=1):
    return val if isinstance(val, tuple) else ((val,) * repeat)


class Sine(nn.Module):
    def __init__(self, w0=1.):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)


class Siren(nn.Module):
    def __init__(self, dim_in, dim_out, w0=1., c=6., is_first=False,
                 use_bias=True, activation='sine'):
        super().__init__()
        self.dim_in = dim_in
        self.is_first = is_first
        self.act = activation

        weight = torch.zeros(dim_out, dim_in)
        bias = torch.zeros(dim_out) if use_bias else None
        self.init_(weight, bias, c=c, w0=w0)

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias) if use_bias else None

        if activation == 'sine':
            self.activation = Sine(w0)
        elif activation == 'relu':
            self.activation = nn.ReLU(inplace=False)
        elif activation == 'id':
            self.activation = nn.Identity()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            raise ValueError('No mlp activation specified')

    def init_(self, weight, bias, c, w0):
        dim = self.dim_in
        act = self.act
        if act == 'relu':
            w_std = math.sqrt(1 / dim)
        else:
            w_std = (1 / dim) if self.is_first else (math.sqrt(c / dim) / w0)
        weight.uniform_(-w_std, w_std)
        if exists(bias):
            bias.uniform_(-w_std, w_std)

    def forward(self, x):
        out = F.linear(x, self.weight, self.bias)
        out = self.activation(out)
        return out


class SirenNet(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out, num_layers, w0=1.,
                 w0_initial=30., use_bias=True, activation='relu',
                 final_activation='sigmoid'):
        super().__init__()

        self.dim_hidden = dim_hidden
        self.num_layers = len(dim_hidden)
        self.layers = nn.ModuleList([])
        for ind in range(self.num_layers):
            is_first = ind == 0
            layer_w0 = w0_initial if is_first else w0
            layer_dim_in = dim_in if is_first else dim_hidden[ind - 1]
            self.layers.append(Siren(
                dim_in=layer_dim_in,
                dim_out=dim_hidden[ind],
                w0=layer_w0,
                use_bias=use_bias,
                is_first=is_first,
                activation=activation,
            ))

        final_activation = 'id' if not exists(final_activation) else final_activation
        self.last_layer = Siren(dim_in=dim_hidden[num_layers - 1], dim_out=dim_out,
                                w0=w0, use_bias=use_bias, activation=final_activation)

    def forward(self, x, mods=None):
        mods = cast_tuple(mods, self.num_layers)
        for layer, mod in zip(self.layers, mods):
            x = layer(x)
            if exists(mod):
                x = x * rearrange(mod, 'd -> () d')
        return self.last_layer(x)
