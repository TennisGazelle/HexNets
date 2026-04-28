import streamlit as st

from networks.HexagonalNetwork import HexagonalNeuralNetwork
from networks.activation.activations import get_activation_function
from networks.loss.loss import get_loss_function


def initialize_session_state():
    if "n" not in st.session_state:
        st.session_state.n = 2
    if "r" not in st.session_state:
        st.session_state.r = 0
    if "activation" not in st.session_state:
        st.session_state.activation = "relu"
    if "loss" not in st.session_state:
        st.session_state.loss = "mean_squared_error"
    if "rotation_comparison_n" not in st.session_state:
        st.session_state.rotation_comparison_n = 2
    if "rotation_comparison_r" not in st.session_state:
        st.session_state.rotation_comparison_r = 0
    if "dataset_type" not in st.session_state:
        st.session_state.dataset_type = "identity"
    if "dataset_num_samples" not in st.session_state:
        st.session_state.dataset_num_samples = 100
    if "net" not in st.session_state:
        st.session_state.net = HexagonalNeuralNetwork(
            n=st.session_state.n,
            r=st.session_state.r,
            learning_rate="constant",
            activation=get_activation_function(st.session_state.activation),
            loss=get_loss_function(st.session_state.loss),
        )


def update_network():
    st.session_state.net = HexagonalNeuralNetwork(
        n=st.session_state.n,
        r=st.session_state.r,
        learning_rate="constant",
        activation=get_activation_function(st.session_state.activation),
        loss=get_loss_function(st.session_state.loss),
    )
