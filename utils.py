import numpy as np
import qutip as qt
from typing import Dict, List, Tuple, Any, Optional
import json
import pickle

class QuantumUtils:
    """
    Utility functions for quantum state manipulation and analysis.
    """
    
    @staticmethod
    def pauli_matrices() -> Dict[str, np.ndarray]:
        """Return dictionary of Pauli matrices."""
        return {
            'I': np.array([[1, 0], [0, 1]], dtype=complex),
            'X': np.array([[0, 1], [1, 0]], dtype=complex),
            'Y': np.array([[0, -1j], [1j, 0]], dtype=complex),
            'Z': np.array([[1, 0], [0, -1]], dtype=complex)
        }
    
    @staticmethod
    def create_bell_states() -> Dict[str, qt.Qobj]:
        """Create the four Bell states."""
        # Computational basis states
        b00 = qt.tensor(qt.basis(2, 0), qt.basis(2, 0))
        b01 = qt.tensor(qt.basis(2, 0), qt.basis(2, 1))
        b10 = qt.tensor(qt.basis(2, 1), qt.basis(2, 0))
        b11 = qt.tensor(qt.basis(2, 1), qt.basis(2, 1))
        
        return {
            'phi_plus': (b00 + b11).unit(),
            'phi_minus': (b00 - b11).unit(),
            'psi_plus': (b01 + b10).unit(),
            'psi_minus': (b01 - b10).unit()
        }
    
    @staticmethod
    def create_ghz_state(n_qubits: int) -> qt.Qobj:
        """Create n-qubit GHZ state."""
        if n_qubits < 2:
            raise ValueError("GHZ state requires at least 2 qubits")
        
        # |000...0⟩ state
        state_0 = qt.tensor(*[qt.basis(2, 0) for _ in range(n_qubits)])
        # |111...1⟩ state
        state_1 = qt.tensor(*[qt.basis(2, 1) for _ in range(n_qubits)])
        
        return (state_0 + state_1).unit()
    
    @staticmethod
    def compute_fidelity(rho1: qt.Qobj, rho2: qt.Qobj) -> float:
        """Compute quantum fidelity between two density matrices."""
        # For density matrices: F = Tr(√(√ρ1 ρ2 √ρ1))
        sqrt_rho1 = rho1.sqrtm()
        temp = sqrt_rho1 * rho2 * sqrt_rho1
        return np.real((temp.sqrtm()).tr())
    
    @staticmethod
    def compute_trace_distance(rho1: qt.Qobj, rho2: qt.Qobj) -> float:
        """Compute trace distance between quantum states."""
        diff = rho1 - rho2
        return 0.5 * np.real((diff.dag() * diff).sqrtm().tr())
    
    @staticmethod
    def partial_transpose(rho: qt.Qobj, subsystem: int) -> qt.Qobj:
        """Compute partial transpose of density matrix."""
        # This is a simplified implementation
        # For full implementation, would need to specify dimensions
        dims = rho.dims[0]
        if len(dims) != 2:
            raise ValueError("Only 2-qubit systems supported in this implementation")
        
        rho_mat = rho.full()
        n = int(np.sqrt(rho_mat.shape[0]))
        
        if subsystem == 0:
            # Transpose first subsystem
            pt_mat = np.zeros_like(rho_mat)
            for i in range(n):
                for j in range(n):
                    for k in range(n):
                        for l in range(n):
                            pt_mat[i*n + k, j*n + l] = rho_mat[j*n + k, i*n + l]
        else:
            # Transpose second subsystem
            pt_mat = np.zeros_like(rho_mat)
            for i in range(n):
                for j in range(n):
                    for k in range(n):
                        for l in range(n):
                            pt_mat[i*n + k, j*n + l] = rho_mat[i*n + l, j*n + k]
        
        return qt.Qobj(pt_mat, dims=rho.dims)
    
    @staticmethod
    def negativity(rho: qt.Qobj, subsystem: int = 0) -> float:
        """Compute negativity of partially transposed state."""
        rho_pt = QuantumUtils.partial_transpose(rho, subsystem)
        eigenvals = rho_pt.eigenenergies()
        return sum(abs(ev) for ev in eigenvals if ev < 0)
    
    @staticmethod
    def quantum_mutual_information(rho: qt.Qobj) -> float:
        """Compute quantum mutual information for bipartite state."""
        # I(A:B) = S(A) + S(B) - S(AB)
        
        # Full system entropy
        s_ab = qt.entropy_vn(rho)
        
        # Reduced state entropies
        rho_a = rho.ptrace(0)
        rho_b = rho.ptrace(1)
        
        s_a = qt.entropy_vn(rho_a)
        s_b = qt.entropy_vn(rho_b)
        
        return s_a + s_b - s_ab

class SimulationState:
    """
    Class for managing simulation state and history.
    """
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.reset()
    
    def reset(self):
        """Reset simulation state."""
        self.time_history = []
        self.state_history = []
        self.parameter_history = []
        self.metrics_history = []
    
    def add_state(self, t: float, state: qt.Qobj, params: Dict[str, Any], metrics: Dict[str, float]):
        """Add state to history."""
        self.time_history.append(t)
        self.state_history.append(state.copy())
        self.parameter_history.append(params.copy())
        self.metrics_history.append(metrics.copy())
        
        # Limit history size
        if len(self.time_history) > self.max_history:
            self.time_history = self.time_history[-self.max_history//2:]
            self.state_history = self.state_history[-self.max_history//2:]
            self.parameter_history = self.parameter_history[-self.max_history//2:]
            self.metrics_history = self.metrics_history[-self.max_history//2:]
    
    def get_history_data(self) -> Dict[str, List]:
        """Get complete history data."""
        return {
            'time': self.time_history,
            'states': self.state_history,
            'parameters': self.parameter_history,
            'metrics': self.metrics_history
        }
    
    def export_data(self, filename: str):
        """Export simulation data to file."""
        data = {
            'time_history': self.time_history,
            'parameter_history': self.parameter_history,
            'metrics_history': self.metrics_history
        }
        
        if filename.endswith('.json'):
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        elif filename.endswith('.pkl'):
            with open(filename, 'wb') as f:
                pickle.dump(data, f)
        else:
            raise ValueError("Unsupported file format. Use .json or .pkl")

class ParameterValidator:
    """
    Utility class for validating simulation parameters.
    """
    
    @staticmethod
    def validate_coherence_params(params: Dict[str, float]) -> bool:
        """Validate coherence parameters."""
        for key, value in params.items():
            if not (0 <= value <= 1):
                return False
        return True
    
    @staticmethod
    def validate_observer_params(params: Dict[str, float]) -> bool:
        """Validate observer time dilation parameters."""
        for key, value in params.items():
            if not (0.1 <= value <= 10.0):
                return False
        return True
    
    @staticmethod
    def validate_emf_params(A_em: float, omega_em: float) -> bool:
        """Validate EMF field parameters."""
        return (0 <= A_em <= 10.0) and (0.1 <= omega_em <= 100.0)
    
    @staticmethod
    def validate_entropy_params(gradient: float, beta_wave: float, beta_particle: float) -> bool:
        """Validate entropy dynamics parameters."""
        return (
            (-10.0 <= gradient <= 10.0) and
            (0 <= beta_wave <= 1.0) and
            (0 <= beta_particle <= 1.0)
        )
    
    @staticmethod
    def validate_all_params(params: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate all simulation parameters."""
        errors = []
        
        # Coherence validation
        if 'coherence' in params:
            if not ParameterValidator.validate_coherence_params(params['coherence']):
                errors.append("Invalid coherence parameters")
        
        # Observer validation
        if 'observer' in params:
            if not ParameterValidator.validate_observer_params(params['observer']):
                errors.append("Invalid observer parameters")
        
        # EMF validation
        if 'A_em' in params and 'omega_em' in params:
            if not ParameterValidator.validate_emf_params(params['A_em'], params['omega_em']):
                errors.append("Invalid EMF parameters")
        
        # Entropy validation
        if all(key in params for key in ['entropy_gradient', 'beta_wave', 'beta_particle']):
            if not ParameterValidator.validate_entropy_params(
                params['entropy_gradient'], 
                params['beta_wave'], 
                params['beta_particle']
            ):
                errors.append("Invalid entropy parameters")
        
        return len(errors) == 0, errors

# Mathematical constants and helper functions
class QuantumConstants:
    """Physical and mathematical constants for quantum simulations."""
    
    HBAR = 1.0  # Reduced Planck constant (normalized)
    PI = np.pi
    E = np.e
    
    # Common quantum gates
    PAULI_I = np.array([[1, 0], [0, 1]], dtype=complex)
    PAULI_X = np.array([[0, 1], [1, 0]], dtype=complex)
    PAULI_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    PAULI_Z = np.array([[1, 0], [0, -1]], dtype=complex)
    
    HADAMARD = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    CNOT = np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 0, 1],
                     [0, 0, 1, 0]], dtype=complex)

def format_complex_number(z: complex, precision: int = 3) -> str:
    """Format complex number for display."""
    real = np.real(z)
    imag = np.imag(z)
    
    if abs(imag) < 1e-10:
        return f"{real:.{precision}f}"
    elif abs(real) < 1e-10:
        return f"{imag:.{precision}f}i"
    else:
        sign = "+" if imag >= 0 else "-"
        return f"{real:.{precision}f}{sign}{abs(imag):.{precision}f}i"

def create_rotation_matrix(axis: str, angle: float) -> np.ndarray:
    """Create rotation matrix for given axis and angle."""
    if axis.upper() == 'X':
        return np.array([[np.cos(angle/2), -1j*np.sin(angle/2)],
                        [-1j*np.sin(angle/2), np.cos(angle/2)]], dtype=complex)
    elif axis.upper() == 'Y':
        return np.array([[np.cos(angle/2), -np.sin(angle/2)],
                        [np.sin(angle/2), np.cos(angle/2)]], dtype=complex)
    elif axis.upper() == 'Z':
        return np.array([[np.exp(-1j*angle/2), 0],
                        [0, np.exp(1j*angle/2)]], dtype=complex)
    else:
        raise ValueError("Axis must be 'X', 'Y', or 'Z'")
