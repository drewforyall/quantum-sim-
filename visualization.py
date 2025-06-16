import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx
from typing import Dict, Any, List, Tuple
import colorsys

class QuantumVisualizer:
    """
    Visualization component for quantum consciousness simulation,
    creating interactive plots for quantum states, entanglement networks,
    and environmental field effects.
    """
    
    def __init__(self):
        self.color_schemes = {
            'quantum': px.colors.sequential.Viridis,
            'entanglement': px.colors.sequential.Plasma,
            'observer': px.colors.sequential.Turbo,
            'emf': px.colors.sequential.Electric
        }
    
    def create_quantum_state_plot(self, state_data: Dict[str, Any], t: float) -> go.Figure:
        """Create quantum state evolution visualization."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Population Dynamics', 'Coherence Matrix',
                'Entropy Evolution', 'Eigenvalue Spectrum'
            ],
            specs=[[{"secondary_y": False}, {"type": "heatmap"}],
                   [{"secondary_y": False}, {"type": "bar"}]]
        )
        
        # Population dynamics
        populations = state_data['populations']
        n_states = len(populations)
        state_labels = [f"|{format(i, f'0{int(np.log2(n_states))}b')}⟩" for i in range(n_states)]
        
        # Create pulsing effect based on time
        pulse_factor = 0.8 + 0.2 * np.sin(5 * t)
        colors = [f'rgba({int(255*p*pulse_factor)}, {int(100*p)}, {int(200*(1-p))}, 0.8)' 
                  for p in populations]
        
        fig.add_trace(
            go.Bar(
                x=state_labels,
                y=populations,
                marker_color=colors,
                name="Population",
                showlegend=False
            ),
            row=1, col=1
        )
        
        # Coherence matrix heatmap
        coherence_matrix = np.zeros((n_states, n_states))
        coherence_matrix[np.diag_indices(n_states)] = populations
        
        for key, value in state_data['coherences'].items():
            i, j = map(int, key.split('_'))
            coherence_matrix[i, j] = abs(value)
            coherence_matrix[j, i] = abs(value)
        
        fig.add_trace(
            go.Heatmap(
                z=coherence_matrix,
                colorscale='Viridis',
                showscale=False,
                name="Coherences"
            ),
            row=1, col=2
        )
        
        # Entropy evolution (placeholder - would need history)
        entropy_history = [state_data['entropy']] * 50  # Simplified
        time_points = np.linspace(t-5, t, 50)
        
        fig.add_trace(
            go.Scatter(
                x=time_points,
                y=entropy_history,
                mode='lines',
                line=dict(color='orange', width=3),
                name="Entropy",
                showlegend=False
            ),
            row=2, col=1
        )
        
        # Eigenvalue spectrum
        eigenvals = state_data['eigenvalues']
        fig.add_trace(
            go.Bar(
                x=[f"λ_{i+1}" for i in range(len(eigenvals))],
                y=eigenvals,
                marker_color='lightblue',
                name="Eigenvalues",
                showlegend=False
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title=f"Quantum State at t = {t:.3f}",
            height=600,
            showlegend=False
        )
        
        return fig
    
    def create_entanglement_network(self, state_data: Dict[str, Any], params: Dict[str, Any]) -> go.Figure:
        """Create dynamic entanglement network visualization."""
        n_qubits = params['n_qubits']
        
        # Create network graph
        G = nx.Graph()
        
        # Add nodes (qubits)
        positions = {}
        for i in range(n_qubits):
            angle = 2 * np.pi * i / n_qubits
            positions[i] = (np.cos(angle), np.sin(angle))
            G.add_node(i)
        
        # Add edges based on entanglement strength
        entanglement_threshold = 0.1
        edge_weights = []
        
        for i in range(n_qubits):
            for j in range(i + 1, n_qubits):
                # Calculate pairwise entanglement (simplified)
                coherence_key = f'{i}_{j}'
                if coherence_key in state_data['coherences']:
                    strength = abs(state_data['coherences'][coherence_key])
                    if strength > entanglement_threshold:
                        G.add_edge(i, j, weight=strength)
                        edge_weights.append(strength)
        
        # Create plotly traces
        edge_x, edge_y = [], []
        edge_colors = []
        
        for edge in G.edges():
            x0, y0 = positions[edge[0]]
            x1, y1 = positions[edge[1]]
            weight = G[edge[0]][edge[1]]['weight']
            
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_colors.extend([weight, weight, None])
        
        node_x = [positions[node][0] for node in G.nodes()]
        node_y = [positions[node][1] for node in G.nodes()]
        
        # Node colors based on coherence
        node_colors = []
        for i in range(n_qubits):
            coherence = params.get('coherence', {}).get(f'C_{i+1}', 0.8)
            node_colors.append(coherence)
        
        fig = go.Figure()
        
        # Add edges
        if edge_x:
            fig.add_trace(go.Scatter(
                x=edge_x, y=edge_y,
                mode='lines',
                line=dict(width=3, color='rgba(50, 150, 250, 0.6)'),
                hoverinfo='none',
                showlegend=False
            ))
        
        # Add nodes
        fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            marker=dict(
                size=30,
                color=node_colors,
                colorscale='Viridis',
                line=dict(width=2, color='white'),
                cmin=0, cmax=1
            ),
            text=[f'Q{i+1}' for i in range(n_qubits)],
            textposition='middle center',
            textfont=dict(color='white', size=12),
            hovertemplate='Qubit %{text}<br>Coherence: %{marker.color:.3f}<extra></extra>',
            showlegend=False
        ))
        
        fig.update_layout(
            title="Entanglement Network",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=400,
            plot_bgcolor='rgba(0,0,0,0.1)'
        )
        
        return fig
    
    def create_observer_animation(self, state_data: Dict[str, Any], t: float, params: Dict[str, Any]) -> go.Figure:
        """Create observer role oscillation visualization."""
        n_qubits = params['n_qubits']
        omega_obs = params.get('omega_obs', 1.0)
        
        fig = go.Figure()
        
        # Observer phases
        phases = []
        amplitudes = []
        
        for i in range(n_qubits):
            T_i = params.get('observer', {}).get(f'T_{i+1}', 1.0)
            phase = omega_obs * t + i * np.pi / 2
            amplitude = T_i * np.sin(phase)
            
            phases.append(phase)
            amplitudes.append(amplitude)
        
        # Create radar chart for observer roles
        categories = [f'Observer {i+1}' for i in range(n_qubits)]
        
        fig.add_trace(go.Scatterpolar(
            r=[abs(amp) for amp in amplitudes],
            theta=categories,
            fill='toself',
            name='Observer Strength',
            line_color='rgba(255, 100, 100, 0.8)',
            fillcolor='rgba(255, 100, 100, 0.3)'
        ))
        
        # Add time dilation visualization
        time_dilations = [params.get('observer', {}).get(f'T_{i+1}', 1.0) for i in range(n_qubits)]
        
        fig.add_trace(go.Scatterpolar(
            r=time_dilations,
            theta=categories,
            mode='markers+lines',
            name='Time Dilation',
            line_color='rgba(100, 255, 100, 0.8)',
            marker=dict(size=10)
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 2]
                )
            ),
            title=f"Observer Dynamics (t = {t:.2f})",
            height=300
        )
        
        return fig
    
    def create_emf_field_plot(self, t: float, params: Dict[str, Any]) -> go.Figure:
        """Create EMF field visualization with wave overlay."""
        A_em = params.get('A_em', 0.5)
        omega_em = params.get('omega_em', 2.0)
        g_displacement = params.get('g_displacement', 0.2)
        
        # Generate EMF field data
        x = np.linspace(-np.pi, np.pi, 100)
        emf_field = A_em * np.cos(omega_em * t + x)
        
        # Gravitational displacement effect
        gravity_effect = g_displacement * np.sin(x + t) * emf_field**2
        
        fig = go.Figure()
        
        # EMF field
        fig.add_trace(go.Scatter(
            x=x,
            y=emf_field,
            mode='lines',
            name='EMF Field',
            line=dict(color='blue', width=3)
        ))
        
        # Gravitational displacement
        fig.add_trace(go.Scatter(
            x=x,
            y=gravity_effect,
            mode='lines',
            name='Gravity Effect',
            line=dict(color='red', width=2, dash='dash')
        ))
        
        # Combined field
        combined = emf_field + gravity_effect
        fig.add_trace(go.Scatter(
            x=x,
            y=combined,
            mode='lines',
            name='Combined Field',
            line=dict(color='purple', width=2),
            fill='tonexty'
        ))
        
        fig.update_layout(
            title=f"EMF Field & Gravitational Effects (t = {t:.2f})",
            xaxis_title="Position",
            yaxis_title="Field Strength",
            height=250,
            legend=dict(x=0, y=1)
        )
        
        return fig
    
    def create_entropy_dynamics_plot(self, entropy_history: List[float], time_history: List[float]) -> go.Figure:
        """Create entropy dynamics visualization showing wave-particle transitions."""
        fig = go.Figure()
        
        # Entropy evolution
        fig.add_trace(go.Scatter(
            x=time_history,
            y=entropy_history,
            mode='lines+markers',
            name='Entropy',
            line=dict(color='orange', width=3),
            marker=dict(size=6)
        ))
        
        # Add wave-particle transition indicators
        if len(entropy_history) > 1:
            entropy_diff = np.diff(entropy_history)
            transition_points = []
            for i, diff in enumerate(entropy_diff):
                if abs(diff) > 0.1:  # Threshold for significant change
                    transition_points.append(i)
            
            if transition_points:
                fig.add_trace(go.Scatter(
                    x=[time_history[i] for i in transition_points],
                    y=[entropy_history[i] for i in transition_points],
                    mode='markers',
                    name='Transitions',
                    marker=dict(
                        symbol='star',
                        size=12,
                        color='red'
                    )
                ))
        
        fig.update_layout(
            title="Entropy Dynamics & Wave-Particle Transitions",
            xaxis_title="Time",
            yaxis_title="Entropy",
            height=300
        )
        
        return fig
