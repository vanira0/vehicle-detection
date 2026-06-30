"""
Tests for training components: optimizer factory, scheduler factory, callbacks.
"""

import os
import sys
import tempfile

import torch
import torch.nn as nn
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from training.optimizers import OptimizerFactory
from training.schedulers import SchedulerFactory
from training.callbacks import (
    CheckpointCallback,
    EarlyStoppingCallback,
    LoggingCallback,
    CallbackList,
)
from utils.config import Config


class SimpleModel(nn.Module):
    """A simple model for testing."""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 2)

    def forward(self, x):
        return self.linear(x)


class TestOptimizerFactory:
    """Tests for the optimizer factory."""

    def setup_method(self):
        self.model = SimpleModel()

    def test_sgd(self):
        config = Config.from_dict({"name": "sgd", "lr": 0.01, "momentum": 0.9, "weight_decay": 0.0001})
        opt = OptimizerFactory.create(config, self.model)
        assert isinstance(opt, torch.optim.SGD)
        assert opt.param_groups[0]["lr"] == 0.01

    def test_adam(self):
        config = Config.from_dict({"name": "adam", "lr": 0.001, "weight_decay": 0.0001})
        opt = OptimizerFactory.create(config, self.model)
        assert isinstance(opt, torch.optim.Adam)

    def test_adamw(self):
        config = Config.from_dict({"name": "adamw", "lr": 0.001, "weight_decay": 0.01})
        opt = OptimizerFactory.create(config, self.model)
        assert isinstance(opt, torch.optim.AdamW)

    def test_unknown_optimizer_raises(self):
        config = Config.from_dict({"name": "nonexistent", "lr": 0.01})
        with pytest.raises(ValueError, match="Unknown optimizer"):
            OptimizerFactory.create(config, self.model)


class TestSchedulerFactory:
    """Tests for the scheduler factory."""

    def setup_method(self):
        self.model = SimpleModel()
        opt_config = Config.from_dict({"name": "sgd", "lr": 0.01, "momentum": 0.9, "weight_decay": 0.0001})
        self.optimizer = OptimizerFactory.create(opt_config, self.model)

    def test_step_scheduler(self):
        config = Config.from_dict({"name": "step", "step_size": 5, "gamma": 0.1})
        scheduler = SchedulerFactory.create(config, self.optimizer)
        assert scheduler is not None

    def test_cosine_scheduler(self):
        config = Config.from_dict({"name": "cosine", "T_max": 50, "eta_min": 0.0001})
        scheduler = SchedulerFactory.create(config, self.optimizer)
        assert scheduler is not None

    def test_plateau_scheduler(self):
        config = Config.from_dict({"name": "plateau", "patience": 5, "factor": 0.1})
        scheduler = SchedulerFactory.create(config, self.optimizer)
        assert scheduler is not None

    def test_unknown_scheduler_raises(self):
        config = Config.from_dict({"name": "nonexistent"})
        with pytest.raises(ValueError, match="Unknown scheduler"):
            SchedulerFactory.create(config, self.optimizer)


class TestCallbacks:
    """Tests for training callbacks."""

    def test_early_stopping_triggers(self):
        es = EarlyStoppingCallback(patience=3, metric="loss", mode="min")

        class MockTrainer:
            logger = type("", (), {"info": staticmethod(lambda x: None)})()

        trainer = MockTrainer()

        # Simulate decreasing loss then stagnation
        assert not es.on_epoch_end(0, {"loss": 1.0}, {}, trainer)
        assert not es.on_epoch_end(1, {"loss": 0.5}, {}, trainer)
        assert not es.on_epoch_end(2, {"loss": 0.6}, {}, trainer)  # No improvement
        assert not es.on_epoch_end(3, {"loss": 0.7}, {}, trainer)  # No improvement
        assert es.on_epoch_end(4, {"loss": 0.8}, {}, trainer)      # Triggered!

    def test_early_stopping_resets(self):
        es = EarlyStoppingCallback(patience=3, metric="loss", mode="min")

        class MockTrainer:
            logger = type("", (), {"info": staticmethod(lambda x: None)})()

        trainer = MockTrainer()

        assert not es.on_epoch_end(0, {"loss": 1.0}, {}, trainer)
        assert not es.on_epoch_end(1, {"loss": 0.9}, {}, trainer)
        assert not es.on_epoch_end(2, {"loss": 0.95}, {}, trainer)  # No improvement
        assert not es.on_epoch_end(3, {"loss": 0.8}, {}, trainer)   # Improved! Reset counter
        assert es.counter == 0

    def test_checkpoint_callback_saves(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cb = CheckpointCallback(
                checkpoint_dir=tmpdir,
                save_every=1,
                save_best=True,
                metric="loss",
                mode="min",
            )

            class MockTrainer:
                logger = type("", (), {"info": staticmethod(lambda x: None)})()
                best_metric = float("-inf")

                def save_checkpoint(self, path, metrics):
                    torch.save({"test": True}, path)

            trainer = MockTrainer()
            cb.on_epoch_end(0, {"loss": 1.0}, {}, trainer)

            assert os.path.exists(os.path.join(tmpdir, "epoch_0000.pth"))
            assert os.path.exists(os.path.join(tmpdir, "best.pth"))

    def test_callback_list(self):
        cb1 = LoggingCallback()
        cb2 = LoggingCallback()
        cb_list = CallbackList([cb1, cb2])

        class MockTrainer:
            logger = type("", (), {"info": staticmethod(lambda x: None)})()
            optimizer = type("", (), {"param_groups": [{"lr": 0.01}]})()

        trainer = MockTrainer()
        result = cb_list.on_epoch_end(0, {"loss": 1.0}, {}, trainer)
        assert result is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
