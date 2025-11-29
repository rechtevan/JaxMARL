"""
Smoke tests for IPPO baseline algorithms.

These tests validate that IPPO variants (RNN, FF, CNN) can:
1. Load their configuration files successfully
2. Initialize network architectures without errors
3. Run short training loops without crashing

These are NOT learning tests - they only verify the code doesn't crash.
"""

import os
import sys
from pathlib import Path

import hydra
import jax
import jax.numpy as jnp
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


# Add baselines to path
baselines_path = Path(__file__).parent.parent.parent / "baselines"
sys.path.insert(0, str(baselines_path))


class TestIPPOConfigLoading:
    """Test that all IPPO config files can be loaded and validated."""

    @pytest.fixture
    def config_dir(self):
        """Get the absolute path to IPPO config directory."""
        return str(
            Path(__file__).parent.parent.parent / "baselines" / "IPPO" / "config"
        )

    def test_ippo_rnn_smax_config(self, config_dir):
        """Test loading SMAX RNN config."""
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            cfg = compose(config_name="ippo_rnn_smax")
            assert cfg["LR"] > 0
            assert cfg["NUM_ENVS"] > 0
            assert cfg["TOTAL_TIMESTEPS"] > 0
            assert "GRU_HIDDEN_DIM" in cfg
            assert "FC_DIM_SIZE" in cfg

    def test_ippo_rnn_mpe_config(self, config_dir):
        """Test loading MPE RNN config."""
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            cfg = compose(config_name="ippo_rnn_mpe")
            assert cfg["LR"] > 0
            assert cfg["NUM_ENVS"] > 0
            assert "GRU_HIDDEN_DIM" in cfg

    def test_ippo_rnn_hanabi_config(self, config_dir):
        """Test loading Hanabi RNN config."""
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            cfg = compose(config_name="ippo_rnn_hanabi")
            assert cfg["LR"] > 0
            assert cfg["NUM_ENVS"] > 0
            assert "GRU_HIDDEN_DIM" in cfg

    def test_ippo_rnn_overcooked_v2_config(self, config_dir):
        """Test loading Overcooked V2 RNN config."""
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            cfg = compose(config_name="ippo_rnn_overcooked_v2")
            assert cfg["LR"] > 0
            assert cfg["NUM_ENVS"] > 0
            assert "GRU_HIDDEN_DIM" in cfg

    def test_ippo_ff_mpe_config(self, config_dir):
        """Test loading MPE feedforward config."""
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            cfg = compose(config_name="ippo_ff_mpe")
            assert cfg["LR"] > 0
            assert cfg["NUM_ENVS"] > 0
            assert cfg["ACTIVATION"] in ["tanh", "relu"]

    def test_ippo_ff_hanabi_config(self, config_dir):
        """Test loading Hanabi feedforward config."""
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            cfg = compose(config_name="ippo_ff_hanabi")
            assert cfg["LR"] > 0
            assert cfg["NUM_ENVS"] > 0

    def test_ippo_ff_mabrax_config(self, config_dir):
        """Test loading MABrax feedforward config."""
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            cfg = compose(config_name="ippo_ff_mabrax")
            assert cfg["LR"] > 0
            assert cfg["NUM_ENVS"] > 0

    def test_ippo_ff_switch_riddle_config(self, config_dir):
        """Test loading Switch Riddle feedforward config."""
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            cfg = compose(config_name="ippo_ff_switch_riddle")
            assert cfg["LR"] > 0
            assert cfg["NUM_ENVS"] > 0

    def test_ippo_ff_mpe_facmac_config(self, config_dir):
        """Test loading MPE FACMAC feedforward config."""
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            cfg = compose(config_name="ippo_ff_mpe_facmac")
            assert cfg["LR"] > 0
            assert cfg["NUM_ENVS"] > 0

    def test_ippo_ff_overcooked_config(self, config_dir):
        """Test loading Overcooked feedforward config."""
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            cfg = compose(config_name="ippo_ff_overcooked")
            assert cfg["LR"] > 0
            assert cfg["NUM_ENVS"] > 0

    def test_ippo_cnn_overcooked_config(self, config_dir):
        """Test loading Overcooked CNN config."""
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            cfg = compose(config_name="ippo_cnn_overcooked")
            assert cfg["LR"] > 0
            assert cfg["NUM_ENVS"] > 0


class TestIPPONetworkInitialization:
    """Test that IPPO network architectures can be initialized."""

    def test_actorcritic_ff_initialization(self):
        """Test feedforward ActorCritic network initialization."""
        from IPPO.ippo_ff_mpe import ActorCritic

        network = ActorCritic(action_dim=5, activation="tanh")
        rng = jax.random.PRNGKey(0)
        dummy_obs = jnp.zeros((4, 10))  # batch_size=4, obs_dim=10

        params = network.init(rng, dummy_obs)
        assert params is not None

        # Test forward pass
        pi, value = network.apply(params, dummy_obs)
        assert pi is not None
        assert value.shape == (4,)

    def test_actorcritic_rnn_initialization(self):
        """Test RNN ActorCritic network initialization."""
        from IPPO.ippo_rnn_smax import ActorCriticRNN, ScannedRNN

        config = {
            "FC_DIM_SIZE": 64,
            "GRU_HIDDEN_DIM": 64,
        }
        network = ActorCriticRNN(action_dim=6, config=config)
        rng = jax.random.PRNGKey(0)

        # Initialize with proper inputs: (obs, dones, avail_actions)
        batch_size = 4
        seq_len = 10
        obs_dim = 8
        action_dim = 6

        hidden = ScannedRNN.initialize_carry(batch_size, config["GRU_HIDDEN_DIM"])
        dummy_obs = jnp.zeros((seq_len, batch_size, obs_dim))
        dummy_dones = jnp.zeros((seq_len, batch_size))
        dummy_avail = jnp.ones((seq_len, batch_size, action_dim))

        params = network.init(rng, hidden, (dummy_obs, dummy_dones, dummy_avail))
        assert params is not None

        # Test forward pass
        new_hidden, pi, value = network.apply(
            params, hidden, (dummy_obs, dummy_dones, dummy_avail)
        )
        assert new_hidden is not None
        assert pi is not None
        assert value.shape == (seq_len, batch_size)

    def test_actorcritic_cnn_initialization(self):
        """Test CNN ActorCritic network initialization."""
        from IPPO.ippo_cnn_overcooked import CNN, ActorCritic

        network = ActorCritic(action_dim=6, activation="tanh")
        rng = jax.random.PRNGKey(0)

        # Overcooked uses 5x5 grid observations with multiple channels
        batch_size = 4
        dummy_obs = jnp.zeros((batch_size, 5, 5, 26))  # Standard Overcooked obs shape

        params = network.init(rng, dummy_obs)
        assert params is not None

        # Test forward pass
        pi, value = network.apply(params, dummy_obs)
        assert pi is not None
        assert value.shape == (batch_size,)

    def test_cnn_embedding_initialization(self):
        """Test CNN embedding network initialization."""
        from IPPO.ippo_cnn_overcooked import CNN

        cnn = CNN(activation="relu")
        rng = jax.random.PRNGKey(0)

        batch_size = 2
        dummy_obs = jnp.zeros((batch_size, 5, 5, 26))

        params = cnn.init(rng, dummy_obs)
        assert params is not None

        # Test forward pass
        embedding = cnn.apply(params, dummy_obs)
        assert embedding is not None
        assert len(embedding.shape) == 2  # Should be flattened


class TestIPPOShortTraining:
    """Test that IPPO variants can run short training loops without crashing."""

    @pytest.fixture(autouse=True)
    def setup_wandb(self):
        """Initialize wandb in disabled mode before each test."""
        import wandb

        # Initialize wandb in disabled mode for all tests
        wandb.init(mode="disabled", project="test", entity="test")
        yield
        # Finish wandb after test
        wandb.finish()

    def test_ippo_ff_mpe_short_training(self):
        """Test IPPO FF on MPE for 100 timesteps."""
        from IPPO.ippo_ff_mpe import make_train

        config = {
            "LR": 2.5e-4,
            "NUM_ENVS": 4,  # Small for fast testing
            "NUM_STEPS": 10,  # Small for fast testing
            "TOTAL_TIMESTEPS": 100,  # Minimal timesteps
            "UPDATE_EPOCHS": 1,
            "NUM_MINIBATCHES": 2,
            "GAMMA": 0.99,
            "GAE_LAMBDA": 0.95,
            "CLIP_EPS": 0.2,
            "ENT_COEF": 0.01,
            "VF_COEF": 0.5,
            "MAX_GRAD_NORM": 0.5,
            "ACTIVATION": "tanh",
            "ENV_NAME": "MPE_simple_spread_v3",
            "ENV_KWARGS": {},
            "ANNEAL_LR": False,
            "SEED": 0,
        }

        rng = jax.random.PRNGKey(config["SEED"])
        train_fn = make_train(config)
        out = train_fn(rng)

        # Should complete without errors
        assert out is not None

    def test_ippo_ff_switch_riddle_short_training(self):
        """Test IPPO FF on Switch Riddle for 100 timesteps."""
        from IPPO.ippo_ff_switch_riddle import make_train

        config = {
            "LR": 2.5e-4,
            "NUM_ENVS": 4,
            "NUM_STEPS": 10,
            "TOTAL_TIMESTEPS": 100,
            "UPDATE_EPOCHS": 1,
            "NUM_MINIBATCHES": 2,
            "GAMMA": 0.99,
            "GAE_LAMBDA": 0.95,
            "CLIP_EPS": 0.2,
            "ENT_COEF": 0.01,
            "VF_COEF": 0.5,
            "MAX_GRAD_NORM": 0.5,
            "ACTIVATION": "tanh",
            "ENV_NAME": "switch_riddle",
            "NUM_AGENTS": 2,
            "ENV_KWARGS": {},
            "ANNEAL_LR": False,
            "SEED": 0,
        }

        rng = jax.random.PRNGKey(config["SEED"])
        train_fn = make_train(config)
        out = train_fn(rng)

        assert out is not None

    def test_ippo_rnn_smax_short_training(self):
        """Test IPPO RNN on SMAX for 100 timesteps."""
        from IPPO.ippo_rnn_smax import make_train

        config = {
            "LR": 0.004,
            "NUM_ENVS": 4,
            "NUM_STEPS": 10,
            "GRU_HIDDEN_DIM": 64,  # Smaller for faster testing
            "FC_DIM_SIZE": 64,
            "TOTAL_TIMESTEPS": 100,
            "UPDATE_EPOCHS": 1,
            "NUM_MINIBATCHES": 2,
            "GAMMA": 0.99,
            "GAE_LAMBDA": 0.95,
            "CLIP_EPS": 0.05,
            "SCALE_CLIP_EPS": False,
            "ENT_COEF": 0.01,
            "VF_COEF": 0.5,
            "MAX_GRAD_NORM": 0.25,
            "ACTIVATION": "relu",
            "MAP_NAME": "2s3z",
            "SEED": 0,
            "ENV_KWARGS": {
                "see_enemy_actions": True,
                "walls_cause_death": True,
                "attack_mode": "closest",
            },
            "ANNEAL_LR": False,
        }

        rng = jax.random.PRNGKey(config["SEED"])
        train_fn = make_train(config)
        out = train_fn(rng)

        assert out is not None

    def test_ippo_rnn_mpe_short_training(self):
        """Test IPPO RNN on MPE for 100 timesteps."""
        from IPPO.ippo_rnn_mpe import make_train

        config = {
            "LR": 2.5e-4,
            "NUM_ENVS": 4,
            "NUM_STEPS": 10,
            "GRU_HIDDEN_DIM": 64,
            "FC_DIM_SIZE": 64,
            "TOTAL_TIMESTEPS": 100,
            "UPDATE_EPOCHS": 1,
            "NUM_MINIBATCHES": 2,
            "GAMMA": 0.99,
            "GAE_LAMBDA": 0.95,
            "CLIP_EPS": 0.2,
            "SCALE_CLIP_EPS": False,  # Required by ippo_rnn_mpe
            "ENT_COEF": 0.01,
            "VF_COEF": 0.5,
            "MAX_GRAD_NORM": 0.5,
            "ACTIVATION": "tanh",
            "ENV_NAME": "MPE_simple_spread_v3",
            "ENV_KWARGS": {},
            "ANNEAL_LR": False,
            "SEED": 0,
        }

        rng = jax.random.PRNGKey(config["SEED"])
        train_fn = make_train(config)
        out = train_fn(rng)

        assert out is not None

    @pytest.mark.skip(
        reason="Overcooked environment has complex initialization - tested via manual runs"
    )
    def test_ippo_cnn_overcooked_short_training(self):
        """Test IPPO CNN on Overcooked for 100 timesteps."""
        from IPPO.ippo_cnn_overcooked import make_train

        from jaxmarl.environments.overcooked import overcooked_layouts

        config = {
            "LR": 2.5e-4,
            "NUM_ENVS": 4,
            "NUM_STEPS": 10,
            "TOTAL_TIMESTEPS": 100,
            "UPDATE_EPOCHS": 1,
            "NUM_MINIBATCHES": 2,
            "GAMMA": 0.99,
            "GAE_LAMBDA": 0.95,
            "CLIP_EPS": 0.2,
            "ENT_COEF": 0.01,
            "VF_COEF": 0.5,
            "MAX_GRAD_NORM": 0.5,
            "ACTIVATION": "tanh",
            "ENV_NAME": "overcooked",
            "REW_SHAPING_HORIZON": 100,  # Required by overcooked variants
            "ENV_KWARGS": {"layout": overcooked_layouts["cramped_room"]},
            "ANNEAL_LR": False,
            "SEED": 0,
        }

        rng = jax.random.PRNGKey(config["SEED"])
        train_fn = make_train(config)
        out = train_fn(rng)

        assert out is not None

    @pytest.mark.skip(
        reason="Overcooked environment has complex initialization - tested via manual runs"
    )
    def test_ippo_ff_overcooked_short_training(self):
        """Test IPPO FF on Overcooked for 100 timesteps."""
        from IPPO.ippo_ff_overcooked import make_train

        from jaxmarl.environments.overcooked import overcooked_layouts

        config = {
            "LR": 2.5e-4,
            "NUM_ENVS": 4,
            "NUM_STEPS": 10,
            "TOTAL_TIMESTEPS": 100,
            "UPDATE_EPOCHS": 1,
            "NUM_MINIBATCHES": 2,
            "GAMMA": 0.99,
            "GAE_LAMBDA": 0.95,
            "CLIP_EPS": 0.2,
            "ENT_COEF": 0.01,
            "VF_COEF": 0.5,
            "MAX_GRAD_NORM": 0.5,
            "ACTIVATION": "tanh",
            "ENV_NAME": "overcooked",
            "REW_SHAPING_HORIZON": 100,  # Required by overcooked variants
            "ENV_KWARGS": {"layout": overcooked_layouts["cramped_room"]},
            "ANNEAL_LR": False,
            "SEED": 0,
        }

        rng = jax.random.PRNGKey(config["SEED"])
        train_fn = make_train(config)
        out = train_fn(rng)

        assert out is not None


class TestIPPOUtilityFunctions:
    """Test utility functions used across IPPO implementations."""

    def test_batchify_ff(self):
        """Test batchify function from feedforward implementation."""
        from IPPO.ippo_ff_mpe import batchify

        agent_list = ["agent_0", "agent_1", "agent_2"]
        num_envs = 4
        num_actors = len(agent_list) * num_envs  # 3 agents * 4 envs = 12

        # Create sample observation dict - all same dimension for simplicity
        # (batchify pads different dimensions, but for simple test keep them same)
        obs_dict = {
            "agent_0": jnp.ones((num_envs, 10)),
            "agent_1": jnp.ones((num_envs, 10)),
            "agent_2": jnp.ones((num_envs, 10)),
        }

        batched = batchify(obs_dict, agent_list, num_actors)
        assert batched.shape[0] == num_actors
        # After batchify: (num_actors, flattened_obs_dim)
        assert len(batched.shape) == 2

    def test_unbatchify_ff(self):
        """Test unbatchify function from feedforward implementation."""
        from IPPO.ippo_ff_mpe import unbatchify

        agent_list = ["agent_0", "agent_1", "agent_2"]
        num_envs = 4
        num_actors = len(agent_list)
        action_dim = 5

        # unbatchify expects shape (num_actors, num_envs, action_dim)
        # But the input to unbatchify is flattened: (num_actors * num_envs, action_dim)
        batched_actions = jnp.ones((num_actors * num_envs, action_dim))

        unbatched = unbatchify(batched_actions, agent_list, num_envs, num_actors)
        assert len(unbatched) == len(agent_list)
        assert unbatched["agent_0"].shape == (num_envs, action_dim)

    def test_batchify_rnn(self):
        """Test batchify function from RNN implementation."""
        from IPPO.ippo_rnn_smax import batchify

        agent_list = ["agent_0", "agent_1", "agent_2"]
        num_actors = 12

        obs_dict = {
            "agent_0": jnp.ones((4, 10)),
            "agent_1": jnp.ones((4, 10)),
            "agent_2": jnp.ones((4, 10)),
        }

        batched = batchify(obs_dict, agent_list, num_actors)
        assert batched.shape[0] == num_actors

    def test_unbatchify_rnn(self):
        """Test unbatchify function from RNN implementation."""
        from IPPO.ippo_rnn_smax import unbatchify

        agent_list = ["agent_0", "agent_1", "agent_2"]
        num_envs = 4
        num_actors = len(agent_list)
        action_dim = 5

        # RNN unbatchify expects (num_actors * num_envs, action_dim) flattened input
        batched_actions = jnp.ones((num_actors * num_envs, action_dim))

        unbatched = unbatchify(batched_actions, agent_list, num_envs, num_actors)
        assert len(unbatched) == len(agent_list)
        assert unbatched["agent_0"].shape == (num_envs, action_dim)


class TestIPPOHydraIntegration:
    """Test Hydra configuration integration."""

    def test_config_override(self):
        """Test that Hydra overrides work correctly."""
        config_dir = str(
            Path(__file__).parent.parent.parent / "baselines" / "IPPO" / "config"
        )

        with initialize_config_dir(config_dir=config_dir, version_base=None):
            # Load base config
            cfg = compose(config_name="ippo_ff_mpe")
            base_lr = cfg["LR"]

            # Load with override
            cfg_override = compose(config_name="ippo_ff_mpe", overrides=["LR=0.001"])

            assert cfg_override["LR"] == 0.001
            assert cfg_override["LR"] != base_lr

    def test_multiple_overrides(self):
        """Test multiple simultaneous Hydra overrides."""
        config_dir = str(
            Path(__file__).parent.parent.parent / "baselines" / "IPPO" / "config"
        )

        with initialize_config_dir(config_dir=config_dir, version_base=None):
            cfg = compose(
                config_name="ippo_ff_mpe",
                overrides=["LR=0.001", "NUM_ENVS=8", "TOTAL_TIMESTEPS=1000"],
            )

            assert cfg["LR"] == 0.001
            assert cfg["NUM_ENVS"] == 8
            assert cfg["TOTAL_TIMESTEPS"] == 1000


if __name__ == "__main__":
    # Allow running tests directly for debugging
    pytest.main([__file__, "-v"])
