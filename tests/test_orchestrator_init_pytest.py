import pytest

from orchestrator import ABCDEOrchestrator


def test_baseline_initialization():
    """Baseline mode should initialize without raising and expose expected attributes."""
    orch = ABCDEOrchestrator(mode='baseline')
    assert orch is not None
    assert orch.mode == 'baseline'
    # basic API presence
    assert hasattr(orch, 'run_backtest') and callable(orch.run_backtest)


def test_adaptive_initialization():
    """Adaptive mode should initialize and create version tracking objects."""
    orch = ABCDEOrchestrator(mode='adaptive')
    assert orch is not None
    assert orch.mode == 'adaptive'
    # adaptive should have version tracking attributes
    assert hasattr(orch, 'version_tracker')
    assert hasattr(orch, 'param_adopter')
