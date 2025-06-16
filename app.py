import streamlit as st
import time
from quantum_simulation import QuantumConsciousnessSimulator
from visualization import QuantumVisualizer

# Initialize state
if 'simulator' not in st.session_state:
    st.session_state.simulator = QuantumConsciousnessSimulator()
if 'vis' not in st.session_state:
    st.session_state.vis = QuantumVisualizer()
if 'running' not in st.session_state:
    st.session_state.running = False
if 't' not in st.session_state:
    st.session_state.t = 0.0

def main():
    st.title("Quantum Simulation Demo")

    update_speed = st.slider("Update speed (s/frame)", 0.1, 10.0, 0.5)

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.session_state.running:
            if st.button("⏸️ Pause"):
                st.session_state.running = False
        else:
            if st.button("▶️ Play"):
                st.session_state.running = True
    with col2:
        if st.button("🔄 Reset"):
            st.session_state.t = 0.0
            st.session_state.simulator.reset()
            st.experimental_rerun()

    if not st.session_state.running:
        manual_step = st.button("➡️ Step")
    else:
        st.button("➡️ Step", disabled=True)
        manual_step = False

    if st.session_state.running or manual_step:
        st.session_state.t += update_speed

        state_data = st.session_state.simulator.evolve_step(st.session_state.t)
        fig = st.session_state.vis.create_quantum_state_plot(state_data, st.session_state.t)
        st.plotly_chart(fig, use_container_width=True)

    st.write("Time:", f"{st.session_state.t:.2f} s")
    if st.session_state.running:
        time.sleep(update_speed)
        st.experimental_rerun()

if __name__ == "__main__":
    main()
