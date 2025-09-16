import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import io
from networks.HexagonalNetwork import HexagonalNeuralNetwork
from networks.activation.activations import get_available_activation_functions
from networks.loss.loss import get_available_loss_functions




# Helper function to create matplotlib figure and convert to streamlit
def create_matplotlib_figure(fig):
    """Convert matplotlib figure to streamlit-compatible format"""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    return buf



if __name__ == "__main__":
    # variables
    fig = None
    n = 2
    r = 0

    # Rendering
    st.title("HexNet Visualizer")
    st.subheader("Hexagonal Neural Network Visualizer")
    st.set_page_config(
        page_title=f"HexNet Visualizer (n={n}, r={r})",
        page_icon="🔷",
        layout="wide"
    )

    # Create tabs for different n values
    # tab_names = [f"n={i}" for i in range(2, 9)]
    # tabs = st.tabs(tab_names)

    # # Create content for each tab
    # for i, tab in enumerate(tabs):
    #     n = i + 2  # n ranges from 2 to 8
        
        # with tab:
    st.header(f"Parameters")
    
    # Controls section
    col1, col2 = st.columns([2, 1])
    
    with col1:

        n = st.slider(
            "Number of nodes (n)", 
            min_value=2, 
            max_value=8, 
            value=n, 
            step=1, key=f"n_{n}",
            help="Select number of nodes (n)"
        )
        
        r = st.selectbox(
            "Rotation (r)",
            list(range(6)),  # 0 to 5
            key=f"r_{r}",
            help="Select rotation parameter (r=0 to r=5)",
        )

        activation = st.selectbox(
            "Activation",
            get_available_activation_functions(),
            key=f"activation_{n}",
            help="Select activation function",
        )
        loss = st.selectbox(
            "Loss",
            get_available_loss_functions(),
            key=f"loss_{n}",
            help="Select loss function",
        )

        if st.button(f"Generate Graphs (n={n}, r={r})", key=f"generate_{n}"):
            with st.spinner(f"Generating visualization for n={n}, r={r}..."):
                net = HexagonalNeuralNetwork(n=n, r=r, activation="relu", loss="mse")

                col_structure, col_multi_activation = st.columns(2)

                with col_structure:
                    filename, fig = net.graph_structure(output_dir=".", medium="matplotlib")
                    buf = create_matplotlib_figure(fig)
                    st.image(buf, width='stretch')
                    plt.close(fig)  # Clean up memory

                with col_multi_activation:
                    filename, fig = net._graph_multi_activation(detail="", output_dir=".")
                    buf = create_matplotlib_figure(fig)
                    st.image(buf, width='stretch')
                    plt.close(fig)  # Clean up memory
    
    with col2:
        # Display network information
        try:
            # Use the current rotation value from the selectbox
            current_n = st.session_state.get(f"n_{n}", 2)
            current_r = st.session_state.get(f"r_{r}", 0)
            current_activation = st.session_state.get(f"activation_{n}", "relu")
            current_loss = st.session_state.get(f"loss_{n}", "mse")
            net = HexagonalNeuralNetwork(n=current_n, r=current_r, activation=current_activation, loss=current_loss)
            layers = net.dir_W[current_r]["indices"]
            
            st.subheader("Network Information")

            st.write(f"**Total nodes:** {sum(len(layer) for layer in layers)}")
            st.write(f"**Number of layers:** {len(layers)}")
            st.write(f"**Layer sizes:** {[len(layer) for layer in layers]}")
            st.write(f"**Number of nodes (n):** {current_n}")
            st.write(f"**Rotation (r):** {current_r}")
            st.write(f"**Activation:** {current_activation}")
            st.write(f"**Loss:** {current_loss}")

            # Show layer indices
            with st.expander("Layer Indices"):
                for i, layer in enumerate(layers):
                    st.write(f"Layer {i}: {layer}")
                    
        except Exception as e:
            st.error(f"Error loading network information: {str(e)}")

    # Add some basic info about the project
    st.markdown("---")
    st.markdown("### About")
    st.write("HexNets is a tool for working with hexagonal neural networks.")
    st.write("Each tab shows a different network size (n=2 to n=8). Use the controls to generate different types of visualizations.")













