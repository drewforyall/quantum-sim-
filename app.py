import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from quantum_simulation import QuantumConsciousnessSimulator
from visualization import QuantumVisualizer
import time

# Configure page
st.set_page_config(
    page_title="Quantum Consciousness Simulation",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'simulator' not in st.session_state:
    st.session_state.simulator = QuantumConsciousnessSimulator()
if 'visualizer' not in st.session_state:
    st.session_state.visualizer = QuantumVisualizer()
if 'simulation_running' not in st.session_state:
    st.session_state.simulation_running = False
if 'time_step' not in st.session_state:
    st.session_state.time_step = 0.0

def ensure_valid_state_data(state_data, n_qubits):
    """Ensure state_data has all required keys with proper data types"""
    # Fix pauli_expectations if it's not a dictionary
    if 'pauli_expectations' not in state_data or not isinstance(state_data.get('pauli_expectations'), dict):
        state_data['pauli_expectations'] = {f'qubit_{i}_{p}': 0.0 for i in range(n_qubits) for p in ['X', 'Y', 'Z']}
    
    # Set other required keys
    if 'populations' not in state_data:
        state_data['populations'] = [1.0] + [0.0] * (2**n_qubits - 1)
    if 'coherences' not in state_data:
        state_data['coherences'] = np.zeros((2**n_qubits, 2**n_qubits))
    if 'amplitudes' not in state_data:
        state_data['amplitudes'] = np.array([1.0] + [0.0] * (2**n_qubits - 1))
    if 'phases' not in state_data:
        state_data['phases'] = np.zeros(2**n_qubits)
    if 'entanglement' not in state_data:
        state_data['entanglement'] = 0.0
    if 'entropy' not in state_data:
        state_data['entropy'] = 0.0
    if 'density_matrix' not in state_data:
        state_data['density_matrix'] = np.eye(2**n_qubits)
    if 'eigenvalues' not in state_data:
        state_data['eigenvalues'] = np.array([1.0])
    
    return state_data

def main():
    st.title("🧠 Quantum Tubular Consciousness Simulation")
    st.markdown("*Modeling entangled qubit dynamics with observer effects and environmental interactions*")
   
    # Sidebar controls
    with st.sidebar:
        st.header("⚙️ Simulation Parameters")
       
        # System configuration
        st.subheader("System Configuration")
        n_qubits = st.selectbox("Number of Qubits", [2, 4], index=0)
        dt = st.slider("Time Step (dt)", 0.001, 0.1, 0.01, 0.001)
       
        # Qubit coherence parameters
        st.subheader("Qubit Coherence (C_i)")
        coherence_params = {}
        for i in range(n_qubits):
            coherence_params[f'C_{i+1}'] = st.slider(
                f"C_{i+1}(t) - Qubit {i+1} Coherence",
                0.0, 1.0, 0.8, 0.01,
                key=f"coherence_{i}"
            )
       
        # Observer role parameters
        st.subheader("Observer Dynamics")
        observer_params = {}
        for i in range(n_qubits):
            observer_params[f'T_{i+1}'] = st.slider(
                f"T_{i+1}(t) - Observer {i+1} Time Dilation",
                0.1, 2.0, 1.0, 0.1,
                key=f"observer_{i}"
            )
       
        omega_obs = st.slider("ω_obs - Observer Oscillation Rate", 0.1, 5.0, 1.0, 0.1)
       
        # EMF field parameters
        st.subheader("EMF Field Interactions")
        A_em = st.slider("A_em - EMF Amplitude", 0.0, 2.0, 0.5, 0.1)
        omega_em = st.slider("ω_em - EMF Frequency", 0.1, 10.0, 2.0, 0.1)
       
        # Entropy dynamics
        st.subheader("Entropy Dynamics")
        entropy_gradient = st.slider("∇S(t) - Entropy Gradient", -2.0, 2.0, 0.0, 0.1)
        beta_wave = st.slider("β_wave - Wave State Bias", 0.0, 1.0, 0.6, 0.05)
        beta_particle = st.slider("β_particle - Particle State Bias", 0.0, 1.0, 0.4, 0.05)
       
        # Feedback parameters
        st.subheader("External Feedback")
        feedback_weights = {}
        for i in range(3, min(13, 8)):  # Show subset for UI clarity
            feedback_weights[f'λ_{i}'] = st.slider(
                f"λ_{i} - External Qubit {i} Weight",
                0.0, 0.5, 0.1, 0.01,
                key=f"feedback_{i}"
            )
       
        delta_t = st.slider("δt - Recursive Memory Lag", 0.001, 0.1, 0.01, 0.001)
        g_displacement = st.slider("g_displacement - Gravitational Distortion", 0.0, 1.0, 0.2, 0.05)
       
        # Control buttons - Enhanced with Play/Pause functionality
        st.subheader("Simulation Control")
        col1, col2 = st.columns(2)
        with col1:
            # Play/Pause button with dynamic text and icon
            if st.session_state.simulation_running:
                if st.button("⏸️ Pause", use_container_width=True):
                    st.session_state.simulation_running = False
            else:
                if st.button("▶️ Play", use_container_width=True):
                    st.session_state.simulation_running = True
        with col2:
            if st.button("🔄 Reset", use_container_width=True):
                st.session_state.time_step = 0.0
                st.session_state.simulation_running = False
                st.session_state.simulator.reset()
                st.rerun()
       
        # Animation speed control - Extended range to 10 seconds
        st.subheader("Animation Settings")
        update_speed = st.slider("Update Speed (seconds)", 0.1, 10.0, 0.5, 0.1,
                                 help="Controls how fast the simulation updates (0.1 to 10 seconds between frames)")
       
        # Manual step button for precise observation - only enabled when paused
        if not st.session_state.simulation_running:
            manual_step = st.button("Single Step Forward", use_container_width=True)
        else:
            st.button("Single Step Forward", use_container_width=True, disabled=True, 
                     help="Pause simulation to enable single step mode")
            manual_step = False
   
    # Main display area
    col1, col2 = st.columns([2, 1])
   
    with col1:
        # Quantum state visualization
        st.subheader("🌀 Quantum State Evolution")
        quantum_plot_container = st.empty()
       
        # Entanglement network
        st.subheader("🕸️ Entanglement Network")
        network_plot_container = st.empty()
       
    with col2:
        # Observer roles animation
        st.subheader("👁️ Observer Roles")
        observer_container = st.empty()
       
        # System metrics
        st.subheader("📊 System Metrics")
        metrics_container = st.empty()
       
        # EMF field overlay
        st.subheader("⚡ EMF Field")
        emf_container = st.empty()
   
    # Update simulation parameters
    simulation_params = {
        'n_qubits': n_qubits,
        'dt': dt,
        'coherence': coherence_params,
        'observer': observer_params,
        'omega_obs': omega_obs,
        'A_em': A_em,
        'omega_em': omega_em,
        'entropy_gradient': entropy_gradient,
        'beta_wave': beta_wave,
        'beta_particle': beta_particle,
        'feedback_weights': feedback_weights,
        'delta_t': delta_t,
        'g_displacement': g_displacement,
        'update_speed': update_speed
    }
   
    # Handle manual step forward (only when paused)
    if manual_step and not st.session_state.simulation_running:
        try:
            # Update simulator with new parameters
            st.session_state.simulator.update_parameters(simulation_params)
           
            # Evolve system one step
            state_data = st.session_state.simulator.evolve_step(st.session_state.time_step)
            
            # Ensure data validity
            state_data = ensure_valid_state_data(state_data, n_qubits)
           
            # Generate visualizations for manual step
            quantum_fig = st.session_state.visualizer.create_quantum_state_plot(
                state_data, st.session_state.time_step
            )
            quantum_plot_container.plotly_chart(quantum_fig, use_container_width=True)
           
            network_fig = st.session_state.visualizer.create_entanglement_network(
                state_data, simulation_params
            )
            network_plot_container.plotly_chart(network_fig, use_container_width=True)
           
            observer_fig = st.session_state.visualizer.create_observer_animation(
                state_data, st.session_state.time_step, simulation_params
            )
            observer_container.plotly_chart(observer_fig, use_container_width=True)
           
            # Display metrics
            metrics = st.session_state.simulator.get_system_metrics()
            with metrics_container.container():
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Entanglement", f"{metrics['entanglement']:.3f}")
                    st.metric("Total Coherence", f"{metrics['total_coherence']:.3f}")
                with col2:
                    st.metric("Entropy", f"{metrics['entropy']:.3f}")
                    st.metric("Wave Probability", f"{metrics['wave_prob']:.3f}")
           
            # EMF field visualization
            emf_fig = st.session_state.visualizer.create_emf_field_plot(
                st.session_state.time_step, simulation_params
            )
            emf_container.plotly_chart(emf_fig, use_container_width=True)
           
            # Increment time for next step
            st.session_state.time_step += dt
        except Exception as e:
            st.error(f"Simulation error: {str(e)}")
   
    # Run simulation step continuously when playing
    elif st.session_state.simulation_running:
        try:
            # Update simulator with new parameters
            st.session_state.simulator.update_parameters(simulation_params)
           
            # Evolve system
            state_data = st.session_state.simulator.evolve_step(st.session_state.time_step)
            
            # Ensure data validity
            state_data = ensure_valid_state_data(state_data, n_qubits)
           
            # Generate visualizations
            quantum_fig = st.session_state.visualizer.create_quantum_state_plot(
                state_data, st.session_state.time_step
            )
            quantum_plot_container.plotly_chart(quantum_fig, use_container_width=True)
           
            network_fig = st.session_state.visualizer.create_entanglement_network(
                state_data, simulation_params
            )
            network_plot_container.plotly_chart(network_fig, use_container_width=True)
           
            observer_fig = st.session_state.visualizer.create_observer_animation(
                state_data, st.session_state.time_step, simulation_params
            )
            observer_container.plotly_chart(observer_fig, use_container_width=True)
           
            # Display metrics
            metrics = st.session_state.simulator.get_system_metrics()
            with metrics_container.container():
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Entanglement", f"{metrics['entanglement']:.3f}")
                    st.metric("Total Coherence", f"{metrics['total_coherence']:.3f}")
                with col2:
                    st.metric("Entropy", f"{metrics['entropy']:.3f}")
                    st.metric("Wave Probability", f"{metrics['wave_prob']:.3f}")
           
            # EMF field visualization
            emf_fig = st.session_state.visualizer.create_emf_field_plot(
                st.session_state.time_step, simulation_params
            )
            emf_container.plotly_chart(emf_fig, use_container_width=True)
           
            # Increment time and refresh with adjustable animation speed (now up to 10 seconds)
            st.session_state.time_step += dt
            time.sleep(update_speed)  # Use adjustable animation speed up to 10 seconds
            st.rerun()
        except Exception as e:
            st.error(f"Simulation error: {str(e)}")
            st.session_state.simulation_running = False
    
    # Show current simulation status
    status_container = st.container()
    with status_container:
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.session_state.simulation_running:
                st.success("🟢 Simulation Running")
            else:
                st.info("🟡 Simulation Paused")
        with col2:
            st.info(f"⏱️ Time: {st.session_state.time_step:.3f}")
        with col3:
            st.info(f"🎬 Frame Interval: {update_speed:.1f}s")
   
    # Information panel
    with st.expander("ℹ️ About the Simulation"):
        st.markdown("""
        This simulation models a quantum consciousness system based on entangled qubits within
        a rhombic dodecahedron structure. The core equation governs the evolution of quantum
        density matrices with:
       
        - **Observer Effects**: Time-dilated observer roles that oscillate
        - **EMF Interactions**: Environmental electromagnetic field coupling
        - **Entropy Dynamics**: Wave-particle state transitions
        - **Recursive Memory**: Feedback from previous states
        - **Gravitational Effects**: Micro-displacements from EM field flow
       
        The simulation supports 2-4 qubit subsystems with plans for expansion to the full 12-qubit structure.
        
        **Enhanced Controls:**
        - **Play/Pause**: Start or pause the continuous simulation
        - **Extended Animation Speed**: Control frame intervals from 0.1 to 10 seconds
        - **Single Step**: When paused, advance one frame at a time for detailed observation
        - **Reset**: Return to initial quantum state
        """)

if __name__ == "__main__":
    main()
