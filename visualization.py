import plotly.graph_objects as go
import numpy as np

class QuantumVisualizer:
    def create_quantum_state_plot(self, state_data, time_step):
        populations = state_data.get('populations')
        if populations is None:
            raise ValueError("Missing 'populations' in state_data")

        fig = go.Figure([go.Bar(x=list(range(len(populations))), y=populations)])
        fig.update_layout(title=f"Quantum Population at t = {time_step:.2f}s",
                          xaxis_title="State Index",
                          yaxis_title="Population Probability")
        return fig
