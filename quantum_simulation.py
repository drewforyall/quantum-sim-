import numpy as np
from scipy.integrate import odeint
from scipy.linalg import expm
import qutip as qt
from typing import Dict, List, Tuple, Any
import copy

class QuantumConsciousnessSimulator:
    """
    Core quantum consciousness simulation implementing the governing differential equation
    for entangled qubit dynamics with observer effects and environmental interactions.
    """
   
    def __init__(self):
        self.n_qubits = 2
        self.current_state = None
        self.state_history = []
        self.time_history = []
        self.parameters = {}
        self.reset()
   
    def reset(self):
        """Reset the simulation to initial conditions."""
        self.current_state = self._initialize_quantum_state()
        self.state_history = []
        self.time_history = []
   
    def _initialize_quantum_state(self) -> qt.Qobj:
        """Initialize the quantum system in a superposition state."""
        if self.n_qubits == 2:
            # Bell state |00⟩ + |11⟩
            psi = (qt.tensor(qt.basis(2, 0), qt.basis(2, 0)) +
                   qt.tensor(qt.basis(2, 1), qt.basis(2, 1))).unit()
        elif self.n_qubits == 4:
            # 4-qubit GHZ state
            psi = (qt.tensor(qt.basis(2, 0), qt.basis(2, 0), qt.basis(2, 0), qt.basis(2, 0)) +
                   qt.tensor(qt.basis(2, 1), qt.basis(2, 1), qt.basis(2, 1), qt.basis(2, 1))).unit()
        else:
            # Default 2-qubit case
            psi = qt.tensor(qt.basis(2, 0), qt.basis(2, 0))
       
        return qt.ket2dm(psi)  # Convert to density matrix
   
    def update_parameters(self, params: Dict[str, Any]):
        """Update simulation parameters."""
        self.parameters = copy.deepcopy(params)
        if params['n_qubits'] != self.n_qubits:
            self.n_qubits = params['n_qubits']
            self.reset()
   
    def _construct_hamiltonian(self, t: float) -> qt.Qobj:
        """Construct the time-dependent Hamiltonian."""
        # Initialize total Hamiltonian as zero operator
        dims = [2] * self.n_qubits
        H_total = qt.Qobj(np.zeros((2**self.n_qubits, 2**self.n_qubits)), dims=[dims, dims])
        
        # Individual qubit Hamiltonians
        for i in range(self.n_qubits):
            # Coherence-modulated Pauli-Z terms
            coherence = self.parameters.get('coherence', {}).get(f'C_{i+1}', 0.8)
            
            # Build single qubit operator for position i
            ops = [qt.qeye(2) for _ in range(self.n_qubits)]
            ops[i] = coherence * qt.sigmaz()
            H_total += qt.tensor(*ops)
        
        # Interaction terms (entanglement)
        if self.n_qubits >= 2:
            for i in range(self.n_qubits - 1):
                # XX interaction
                ops_xx = [qt.qeye(2) for _ in range(self.n_qubits)]
                ops_xx[i] = qt.sigmax()
                ops_xx[i + 1] = qt.sigmax()
                
                # YY interaction
                ops_yy = [qt.qeye(2) for _ in range(self.n_qubits)]
                ops_yy[i] = qt.sigmay()
                ops_yy[i + 1] = qt.sigmay()
                
                H_total += 0.5 * (qt.tensor(*ops_xx) + qt.tensor(*ops_yy))
        
        # Observer time dilation effects
        omega_obs = self.parameters.get('omega_obs', 1.0)
        for i in range(self.n_qubits):
            T_obs = self.parameters.get('observer', {}).get(f'T_{i+1}', 1.0)
            phase_factor = T_obs * np.cos(omega_obs * t)
            
            ops = [qt.qeye(2) for _ in range(self.n_qubits)]
            ops[i] = phase_factor * qt.sigmaz()
            H_total += qt.tensor(*ops)
        
        # EMF field coupling
        A_em = self.parameters.get('A_em', 0.5)
        omega_em = self.parameters.get('omega_em', 2.0)
        emf_amplitude = A_em * np.sin(omega_em * t)
        
        for i in range(self.n_qubits):
            ops = [qt.qeye(2) for _ in range(self.n_qubits)]
            ops[i] = emf_amplitude * qt.sigmax()
            H_total += qt.tensor(*ops)
        
        return H_total
   
    def _construct_lindblad_operators(self, t: float) -> List[qt.Qobj]:
        """Construct Lindblad collapse operators for decoherence."""
        L_ops = []
        
        # Dephasing operators
        for i in range(self.n_qubits):
            coherence = self.parameters.get('coherence', {}).get(f'C_{i+1}', 0.8)
            gamma_dephase = 0.1 * (1 - coherence)  # Decoherence rate
            
            ops = [qt.qeye(2) for _ in range(self.n_qubits)]
            ops[i] = np.sqrt(gamma_dephase) * qt.sigmaz()
            L_ops.append(qt.tensor(*ops))
        
        # Amplitude damping
        for i in range(self.n_qubits):
            gamma_relax = 0.05  # Relaxation rate
            
            ops = [qt.qeye(2) for _ in range(self.n_qubits)]
            ops[i] = np.sqrt(gamma_relax) * qt.sigmam()
            L_ops.append(qt.tensor(*ops))
        
        return L_ops
   
    def evolve_step(self, t: float) -> Dict[str, Any]:
        """Evolve the quantum system by one time step."""
        dt = self.parameters.get('dt', 0.01)
        
        # Ensure current state is initialized
        if self.current_state is None:
            self.current_state = self._initialize_quantum_state()
        
        # Construct time-dependent Hamiltonian
        H = self._construct_hamiltonian(t)
        
        # Construct Lindblad operators
        L_ops = self._construct_lindblad_operators(t)
        
        # Evolve using Lindblad master equation
        if L_ops:
            result = qt.mesolve(H, self.current_state, [0, dt], c_ops=L_ops)
            self.current_state = result.states[-1]
        else:
            # Unitary evolution if no collapse operators
            U = (-1j * H * dt).expm()
            self.current_state = U * self.current_state * U.dag()
        
        # Apply entropy dynamics and wave-particle state transitions
        self._apply_entropy_dynamics(t)
        
        # Store history
        if self.current_state is not None:
            self.state_history.append(self.current_state.copy())
            self.time_history.append(t)
        
        # Prepare output data
        state_data = self._extract_state_data()
        
        return state_data
   
    def _apply_entropy_dynamics(self, t: float):
        """Apply entropy gradient effects and wave-particle transitions."""
        if self.current_state is None:
            return
            
        entropy_gradient = self.parameters.get('entropy_gradient', 0.0)
        beta_wave = self.parameters.get('beta_wave', 0.6)
        beta_particle = self.parameters.get('beta_particle', 0.4)
        
        try:
            # Calculate von Neumann entropy
            eigenvals = self.current_state.eigenenergies()
            eigenvals = eigenvals[eigenvals > 1e-12]  # Remove zero eigenvalues
            entropy = -np.sum(eigenvals * np.log2(eigenvals + 1e-12))
            
            # Apply entropy-based state mixing
            wave_prob = beta_wave / (1 + np.exp(-entropy_gradient))
            particle_prob = beta_particle / (1 + np.exp(entropy_gradient))
            
            # Create mixed state based on wave-particle probabilities
            if self.n_qubits == 2:
                wave_state = qt.ket2dm((qt.tensor(qt.basis(2, 0), qt.basis(2, 0)) +
                                       qt.tensor(qt.basis(2, 1), qt.basis(2, 1))).unit())
                particle_state = qt.ket2dm(qt.tensor(qt.basis(2, 0), qt.basis(2, 0)))
            else:
                # 4-qubit case
                wave_state = qt.ket2dm((qt.tensor(qt.basis(2, 0), qt.basis(2, 0), qt.basis(2, 0), qt.basis(2, 0)) +
                                       qt.tensor(qt.basis(2, 1), qt.basis(2, 1), qt.basis(2, 1), qt.basis(2, 1))).unit())
                particle_state = qt.ket2dm(qt.tensor(qt.basis(2, 0), qt.basis(2, 0), qt.basis(2, 0), qt.basis(2, 0)))
            
            # Mix current state with wave/particle states
            mixing_strength = 0.01  # Small mixing to maintain stability
            self.current_state = ((1 - mixing_strength) * self.current_state +
                                 mixing_strength * wave_prob * wave_state +
                                 mixing_strength * particle_prob * particle_state)
            
            # Renormalize
            trace_val = self.current_state.tr()
            if trace_val != 0:
                self.current_state = self.current_state / trace_val
        except Exception as e:
            # Skip entropy dynamics if calculation fails
            pass
   
    def _extract_state_data(self) -> Dict[str, Any]:
        """Extract relevant data from the current quantum state."""
        if self.current_state is None:
            return {
                'density_matrix': np.zeros((2**self.n_qubits, 2**self.n_qubits)),
                'pauli_expectations': {},
                'entanglement': 0.0,
                'entropy': 0.0,
                'eigenvalues': np.array([1.0]),
                'n_qubits': self.n_qubits
            }
        
        try:
            # Density matrix elements
            rho = self.current_state.full()
            
            # Expectation values of Pauli operators
            pauli_expectations = {}
            for i in range(self.n_qubits):
                for pauli_name, pauli_op in [('X', qt.sigmax()), ('Y', qt.sigmay()), ('Z', qt.sigmaz())]:
                    try:
                        ops = [qt.qeye(2) for _ in range(self.n_qubits)]
                        ops[i] = pauli_op
                        operator = qt.tensor(*ops)
                        expectation = qt.expect(operator, self.current_state)
                        pauli_expectations[f'qubit_{i}_{pauli_name}'] = expectation
                    except:
                        pauli_expectations[f'qubit_{i}_{pauli_name}'] = 0.0
            
            # Entanglement measure (concurrence for 2 qubits, tangle for more)
            entanglement = self._calculate_entanglement()
            
            # von Neumann entropy
            try:
                eigenvals = self.current_state.eigenenergies()
                eigenvals = eigenvals[eigenvals > 1e-12]
                entropy = -np.sum(eigenvals * np.log2(eigenvals + 1e-12))
            except:
                eigenvals = np.array([1.0])
                entropy = 0.0
            
            return {
                'density_matrix': rho,
                'pauli_expectations': pauli_expectations,
                'entanglement': entanglement,
                'entropy': entropy,
                'eigenvalues': eigenvals,
                'n_qubits': self.n_qubits
            }
        except Exception as e:
            # Return default values if extraction fails
            return {
                'density_matrix': np.eye(2**self.n_qubits),
                'pauli_expectations': {f'qubit_{i}_{p}': 0.0 for i in range(self.n_qubits) for p in ['X', 'Y', 'Z']},
                'entanglement': 0.0,
                'entropy': 0.0,
                'eigenvalues': np.array([1.0]),
                'n_qubits': self.n_qubits
            }
   
    def _calculate_entanglement(self) -> float:
        """Calculate entanglement measure."""
        if self.current_state is None:
            return 0.0
            
        try:
            if self.n_qubits == 2:
                # Concurrence for 2-qubit systems
                rho = self.current_state.full()
                
                # Pauli-Y tensor product
                sigma_y = np.array([[0, -1j], [1j, 0]])
                sigma_y_tensor = np.kron(sigma_y, sigma_y)
                
                # Time-reversed state
                rho_tilde = sigma_y_tensor @ np.conj(rho) @ sigma_y_tensor
                
                # R matrix
                R = rho @ rho_tilde
                eigenvals = np.linalg.eigvals(R)
                eigenvals = np.sqrt(np.maximum(0, np.real(eigenvals)))
                eigenvals = np.sort(eigenvals)[::-1]
                
                if len(eigenvals) >= 4:
                    concurrence = max(0, eigenvals[0] - eigenvals[1] - eigenvals[2] - eigenvals[3])
                else:
                    concurrence = 0.0
                return concurrence
            else:
                # For more qubits, use linear entropy of reduced states
                entanglement = 0
                for i in range(self.n_qubits):
                    try:
                        # Get reduced state for qubit i
                        reduced_state = self.current_state.ptrace(i)
                        # Calculate linear entropy: 1 - Tr(ρ²)
                        rho_squared = reduced_state * reduced_state
                        linear_entropy = 1 - rho_squared.tr().real
                        entanglement += linear_entropy
                    except:
                        # Fallback calculation if ptrace fails
                        entanglement += 0.5
                
                return entanglement / self.n_qubits
        except Exception as e:
            # Return a reasonable default value if calculation fails
            return 0.5
   
    def get_system_metrics(self) -> Dict[str, float]:
        """Get current system metrics."""
        if not hasattr(self, 'current_state') or self.current_state is None:
            return {
                'entanglement': 0.0,
                'total_coherence': 0.0,
                'entropy': 0.0,
                'wave_prob': 0.0
            }
        
        state_data = self._extract_state_data()
        
        # Total coherence
        total_coherence = 0
        for i in range(self.n_qubits):
            coherence = self.parameters.get('coherence', {}).get(f'C_{i+1}', 0.8)
            total_coherence += coherence
        total_coherence /= self.n_qubits
        
        # Wave probability estimate
        beta_wave = self.parameters.get('beta_wave', 0.6)
        entropy_gradient = self.parameters.get('entropy_gradient', 0.0)
        wave_prob = beta_wave / (1 + np.exp(-entropy_gradient))
        
        return {
            'entanglement': state_data['entanglement'],
            'total_coherence': total_coherence,
            'entropy': state_data['entropy'],
            'wave_prob': wave_prob
        }