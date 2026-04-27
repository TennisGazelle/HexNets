"""Glossary parents for loss, learning rate, and activation registries."""

import networks.activation  # noqa: F401 — register ACTIVATION_FUNCTIONS
import networks.learning_rate  # noqa: F401 — register LEARNING_RATES
import networks.loss  # noqa: F401 — register LOSS_FUNCTIONS

from networks.activation import ACTIVATION_FUNCTIONS
from networks.activation.activations import build_activations_glossary_parent
from networks.learning_rate import LEARNING_RATES
from networks.learning_rate.learning_rate import build_learning_rates_glossary_parent
from networks.loss import LOSS_FUNCTIONS
from networks.loss.loss import build_losses_glossary_parent


def test_build_losses_glossary_parent_matches_registry():
    parent = build_losses_glossary_parent()
    assert parent.title == "Loss functions"
    assert len(parent.children) == len(LOSS_FUNCTIONS)
    for child in parent.children:
        assert child.title
        assert child.english


def test_build_learning_rates_glossary_parent_matches_registry():
    parent = build_learning_rates_glossary_parent()
    assert parent.title == "Learning rates"
    assert len(parent.children) == len(LEARNING_RATES)
    for child in parent.children:
        assert child.title
        assert child.english


def test_build_activations_glossary_parent_matches_registry():
    parent = build_activations_glossary_parent()
    assert parent.title == "Activations"
    assert len(parent.children) == len(ACTIVATION_FUNCTIONS)
    for child in parent.children:
        assert child.title
        assert child.english
