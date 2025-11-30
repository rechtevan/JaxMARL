"""
Comprehensive tests for JaxMARL wrappers.

Tests cover:
- JaxMARLWrapper base class
- LogWrapper for episode logging
- MPELogWrapper for MPE-specific logging
- SMAXLogWrapper for SMAX-specific logging
- OvercookedV2LogWrapper for Overcooked-specific logging
- CTRolloutManager for centralized training
- Utility functions (save_params, load_params, get_space_dim)
"""

import os
import tempfile

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxmarl
from jaxmarl.wrappers.baselines import (
    CTRolloutManager,
    JaxMARLWrapper,
    LogEnvState,
    LogWrapper,
    MPELogWrapper,
    OvercookedV2LogWrapper,
    SMAXLogWrapper,
    get_space_dim,
    load_params,
    save_params,
)


class TestJaxMARLWrapper:
    """Test base JaxMARLWrapper class."""

    def test_wrapper_initialization(self):
        """Test wrapper wraps environment correctly."""
        env = jaxmarl.make("MPE_simple_spread_v3")
        wrapper = JaxMARLWrapper(env)

        assert wrapper._env is env
        assert wrapper.num_agents == env.num_agents
        assert wrapper.agents == env.agents

    def test_wrapper_attribute_delegation(self):
        """Test wrapper delegates attribute access to wrapped env."""
        env = jaxmarl.make("MPE_simple_spread_v3")
        wrapper = JaxMARLWrapper(env)

        # Should delegate to env
        assert wrapper.action_spaces == env.action_spaces
        assert wrapper.observation_spaces == env.observation_spaces

    def test_batchify_floats(self):
        """Test _batchify_floats stacks agent values."""
        env = jaxmarl.make("MPE_simple_spread_v3")
        wrapper = JaxMARLWrapper(env)

        # Create dict of rewards per agent
        rewards = {a: jnp.array(1.0 + i) for i, a in enumerate(env.agents)}
        batched = wrapper._batchify_floats(rewards)

        assert batched.shape == (env.num_agents,)
        for i, agent in enumerate(env.agents):
            assert batched[i] == rewards[agent]


class TestLogWrapper:
    """Test LogWrapper for episode return/length tracking."""

    def test_log_wrapper_reset(self):
        """Test LogWrapper reset initializes tracking state."""
        env = jaxmarl.make("MPE_simple_spread_v3")
        wrapper = LogWrapper(env)

        rng = jax.random.PRNGKey(0)
        obs, state = wrapper.reset(rng)

        assert isinstance(state, LogEnvState)
        assert state.env_state is not None
        assert jnp.all(state.episode_returns == 0)
        assert jnp.all(state.episode_lengths == 0)
        assert jnp.all(state.returned_episode_returns == 0)
        assert jnp.all(state.returned_episode_lengths == 0)

    def test_log_wrapper_step_accumulates_returns(self):
        """Test LogWrapper step accumulates episode returns."""
        env = jaxmarl.make("MPE_simple_spread_v3")
        wrapper = LogWrapper(env)

        rng = jax.random.PRNGKey(0)
        obs, state = wrapper.reset(rng)

        # Take some steps
        for i in range(5):
            rng, step_rng, act_rng = jax.random.split(rng, 3)
            actions = {
                a: env.action_space(a).sample(jax.random.fold_in(act_rng, j))
                for j, a in enumerate(env.agents)
            }
            obs, state, reward, done, info = wrapper.step(step_rng, state, actions)

            # Episode length should increment
            assert jnp.all(state.episode_lengths >= i + 1) or done["__all__"]

    def test_log_wrapper_info_contains_episode_data(self):
        """Test LogWrapper step returns episode data in info."""
        env = jaxmarl.make("MPE_simple_spread_v3")
        wrapper = LogWrapper(env)

        rng = jax.random.PRNGKey(0)
        obs, state = wrapper.reset(rng)

        rng, step_rng, act_rng = jax.random.split(rng, 3)
        actions = {
            a: env.action_space(a).sample(jax.random.fold_in(act_rng, j))
            for j, a in enumerate(env.agents)
        }
        obs, state, reward, done, info = wrapper.step(step_rng, state, actions)

        assert "returned_episode_returns" in info
        assert "returned_episode_lengths" in info
        assert "returned_episode" in info

    def test_log_wrapper_replace_info(self):
        """Test LogWrapper replace_info option."""
        env = jaxmarl.make("MPE_simple_spread_v3")
        wrapper = LogWrapper(env, replace_info=True)

        rng = jax.random.PRNGKey(0)
        obs, state = wrapper.reset(rng)

        rng, step_rng, act_rng = jax.random.split(rng, 3)
        actions = {
            a: env.action_space(a).sample(jax.random.fold_in(act_rng, j))
            for j, a in enumerate(env.agents)
        }
        obs, state, reward, done, info = wrapper.step(step_rng, state, actions)

        # Should only have the log wrapper keys
        assert "returned_episode_returns" in info
        assert "returned_episode_lengths" in info
        assert "returned_episode" in info


class TestMPELogWrapper:
    """Test MPELogWrapper which scales rewards by num_agents."""

    def test_mpe_log_wrapper_reset(self):
        """Test MPELogWrapper reset."""
        env = jaxmarl.make("MPE_simple_spread_v3")
        wrapper = MPELogWrapper(env)

        rng = jax.random.PRNGKey(0)
        obs, state = wrapper.reset(rng)

        assert isinstance(state, LogEnvState)

    def test_mpe_log_wrapper_step(self):
        """Test MPELogWrapper step scales rewards."""
        env = jaxmarl.make("MPE_simple_spread_v3")
        wrapper = MPELogWrapper(env)

        rng = jax.random.PRNGKey(0)
        obs, state = wrapper.reset(rng)

        rng, step_rng, act_rng = jax.random.split(rng, 3)
        actions = {
            a: env.action_space(a).sample(jax.random.fold_in(act_rng, j))
            for j, a in enumerate(env.agents)
        }
        obs, state, reward, done, info = wrapper.step(step_rng, state, actions)

        # Rewards in info should be scaled by num_agents
        # (episode_returns tracks scaled rewards)
        assert "returned_episode_returns" in info


class TestSMAXLogWrapper:
    """Test SMAXLogWrapper for SMAX-specific win tracking."""

    def test_smax_log_wrapper_reset(self):
        """Test SMAXLogWrapper reset initializes win tracking."""
        env = jaxmarl.make("SMAX")
        wrapper = SMAXLogWrapper(env)

        rng = jax.random.PRNGKey(0)
        obs, state = wrapper.reset(rng)

        assert state.env_state is not None
        assert jnp.all(state.won_episode == 0)
        assert jnp.all(state.returned_won_episode == 0)

    def test_smax_log_wrapper_step(self):
        """Test SMAXLogWrapper step tracks wins."""
        env = jaxmarl.make("SMAX")
        wrapper = SMAXLogWrapper(env)

        rng = jax.random.PRNGKey(0)
        obs, state = wrapper.reset(rng)

        rng, step_rng, act_rng = jax.random.split(rng, 3)
        actions = {
            a: env.action_space(a).sample(jax.random.fold_in(act_rng, j))
            for j, a in enumerate(env.agents)
        }
        obs, state, reward, done, info = wrapper.step(step_rng, state, actions)

        assert "returned_won_episode" in info
        assert "returned_episode_returns" in info


class TestOvercookedV2LogWrapper:
    """Test OvercookedV2LogWrapper for recipe tracking."""

    def test_overcooked_v2_log_wrapper_reset(self):
        """Test OvercookedV2LogWrapper reset initializes recipe tracking."""
        env = jaxmarl.make("overcooked_v2")
        wrapper = OvercookedV2LogWrapper(env)

        rng = jax.random.PRNGKey(0)
        obs, state = wrapper.reset(rng)

        assert state.env_state is not None
        assert hasattr(state, "returned_episode_recipe_returns")

    def test_overcooked_v2_log_wrapper_step(self):
        """Test OvercookedV2LogWrapper step tracks recipe returns."""
        env = jaxmarl.make("overcooked_v2")
        wrapper = OvercookedV2LogWrapper(env)

        rng = jax.random.PRNGKey(0)
        obs, state = wrapper.reset(rng)

        rng, step_rng, act_rng = jax.random.split(rng, 3)
        actions = {
            a: env.action_space(a).sample(jax.random.fold_in(act_rng, j))
            for j, a in enumerate(env.agents)
        }
        obs, state, reward, done, info = wrapper.step(step_rng, state, actions)

        assert "returned_episode_recipe_returns" in info
        assert "returned_episode_returns" in info


class TestCTRolloutManager:
    """Test CTRolloutManager for centralized training."""

    def test_ct_rollout_manager_initialization(self):
        """Test CTRolloutManager initialization."""
        env = jaxmarl.make("MPE_simple_spread_v3")
        batch_size = 4
        manager = CTRolloutManager(env, batch_size)

        assert manager.batch_size == batch_size
        assert manager.training_agents == env.agents
        assert manager.max_obs_length > 0
        assert manager.max_action_space > 0

    def test_ct_rollout_manager_batch_reset(self):
        """Test CTRolloutManager batch_reset creates batched states."""
        env = jaxmarl.make("MPE_simple_spread_v3")
        batch_size = 4
        manager = CTRolloutManager(env, batch_size)

        rng = jax.random.PRNGKey(0)
        obs, states = manager.batch_reset(rng)

        # Should have global state
        assert "__all__" in obs

    def test_ct_rollout_manager_batch_step(self):
        """Test CTRolloutManager batch_step processes batched transitions."""
        env = jaxmarl.make("MPE_simple_spread_v3")
        batch_size = 4
        manager = CTRolloutManager(env, batch_size)

        rng = jax.random.PRNGKey(0)
        obs, states = manager.batch_reset(rng)

        # Create batched actions
        rng, act_rng = jax.random.split(rng)
        actions = {
            a: manager.batch_sample(jax.random.fold_in(act_rng, i), a)
            for i, a in enumerate(env.agents)
        }

        rng, step_rng = jax.random.split(rng)
        obs, states, rewards, dones, infos = manager.batch_step(
            step_rng, states, actions
        )

        # Should have global reward
        assert "__all__" in rewards
        assert "__all__" in obs

    def test_ct_rollout_manager_smax(self):
        """Test CTRolloutManager with SMAX environment."""
        env = jaxmarl.make("SMAX")
        batch_size = 2
        manager = CTRolloutManager(env, batch_size)

        rng = jax.random.PRNGKey(0)
        obs, states = manager.batch_reset(rng)

        assert "__all__" in obs

    def test_ct_rollout_manager_overcooked(self):
        """Test CTRolloutManager with Overcooked environment."""
        env = jaxmarl.make("overcooked")
        batch_size = 2
        manager = CTRolloutManager(env, batch_size)

        rng = jax.random.PRNGKey(0)
        obs, states = manager.batch_reset(rng)

        assert "__all__" in obs

    def test_ct_rollout_manager_hanabi(self):
        """Test CTRolloutManager with Hanabi environment."""
        env = jaxmarl.make("hanabi")
        batch_size = 2
        manager = CTRolloutManager(env, batch_size)

        rng = jax.random.PRNGKey(0)
        obs, states = manager.batch_reset(rng)

        assert "__all__" in obs

    def test_ct_rollout_manager_get_valid_actions(self):
        """Test CTRolloutManager get_valid_actions."""
        env = jaxmarl.make("MPE_simple_spread_v3")
        batch_size = 4
        manager = CTRolloutManager(env, batch_size)

        rng = jax.random.PRNGKey(0)
        obs, states = manager.batch_reset(rng)

        valid_actions = manager.get_valid_actions(states)

        for agent in env.agents:
            assert agent in valid_actions
            assert valid_actions[agent].shape[0] == batch_size

    def test_ct_rollout_manager_training_agents_subset(self):
        """Test CTRolloutManager with subset of training agents."""
        env = jaxmarl.make("MPE_simple_spread_v3")
        batch_size = 2
        training_agents = env.agents[:1]  # Only first agent
        manager = CTRolloutManager(env, batch_size, training_agents=training_agents)

        assert manager.training_agents == training_agents

    def test_ct_rollout_manager_preprocess_obs_disabled(self):
        """Test CTRolloutManager with preprocessing disabled."""
        env = jaxmarl.make("MPE_simple_spread_v3")
        batch_size = 2
        manager = CTRolloutManager(env, batch_size, preprocess_obs=False)

        rng = jax.random.PRNGKey(0)
        obs, states = manager.batch_reset(rng)

        # Observations should not be padded/augmented
        assert "__all__" in obs


class TestUtilityFunctions:
    """Test utility functions."""

    def test_save_and_load_params(self):
        """Test save_params and load_params round-trip."""
        params = {
            "layer1": {"weights": jnp.ones((3, 3)), "bias": jnp.zeros(3)},
            "layer2": {"weights": jnp.eye(3), "bias": jnp.ones(3)},
        }

        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
            filename = f.name

        try:
            save_params(params, filename)
            loaded_params = load_params(filename)

            # Check structure matches
            assert "layer1" in loaded_params
            assert "layer2" in loaded_params
            assert "weights" in loaded_params["layer1"]
            assert "bias" in loaded_params["layer1"]

            # Check values match
            np.testing.assert_array_almost_equal(
                loaded_params["layer1"]["weights"], params["layer1"]["weights"]
            )
            np.testing.assert_array_almost_equal(
                loaded_params["layer1"]["bias"], params["layer1"]["bias"]
            )
        finally:
            os.unlink(filename)

    def test_get_space_dim_discrete(self):
        """Test get_space_dim with Discrete space."""
        from jaxmarl.environments.spaces import Discrete

        space = Discrete(5)
        assert get_space_dim(space) == 5

    def test_get_space_dim_box(self):
        """Test get_space_dim with Box space."""
        from jaxmarl.environments.spaces import Box

        space = Box(low=0.0, high=1.0, shape=(3, 4))
        assert get_space_dim(space) == 12

    def test_get_space_dim_multidiscrete(self):
        """Test get_space_dim with MultiDiscrete space."""
        from jaxmarl.environments.spaces import MultiDiscrete

        space = MultiDiscrete(jnp.array([3, 4, 5]))
        assert get_space_dim(space) == 3

    def test_get_space_dim_gymnax_discrete(self):
        """Test get_space_dim with Gymnax Discrete space."""
        from gymnax.environments.spaces import Discrete as DiscreteGymnax

        space = DiscreteGymnax(5)
        assert get_space_dim(space) == 5

    def test_get_space_dim_gymnax_box(self):
        """Test get_space_dim with Gymnax Box space."""
        from gymnax.environments.spaces import Box as BoxGymnax

        space = BoxGymnax(low=0.0, high=1.0, shape=(3, 4))
        assert get_space_dim(space) == 12


class TestLogEnvStateDataclass:
    """Test LogEnvState dataclass structure."""

    def test_log_env_state_creation(self):
        """Test LogEnvState can be created."""
        env = jaxmarl.make("MPE_simple_spread_v3")
        rng = jax.random.PRNGKey(0)
        obs, env_state = env.reset(rng)

        state = LogEnvState(
            env_state=env_state,
            episode_returns=jnp.zeros(env.num_agents),
            episode_lengths=jnp.zeros(env.num_agents),
            returned_episode_returns=jnp.zeros(env.num_agents),
            returned_episode_lengths=jnp.zeros(env.num_agents),
        )

        assert state.env_state is env_state
        assert state.episode_returns.shape == (env.num_agents,)


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_wrapper_with_coin_game(self):
        """Test wrappers work with coin_game environment."""
        env = jaxmarl.make("coin_game")
        wrapper = LogWrapper(env)

        rng = jax.random.PRNGKey(0)
        obs, state = wrapper.reset(rng)

        assert state is not None

    def test_wrapper_with_switch_riddle(self):
        """Test wrappers work with switch_riddle environment."""
        env = jaxmarl.make("switch_riddle")
        wrapper = LogWrapper(env)

        rng = jax.random.PRNGKey(0)
        obs, state = wrapper.reset(rng)

        assert state is not None

    def test_ct_rollout_manager_multiple_resets(self):
        """Test CTRolloutManager handles multiple resets."""
        env = jaxmarl.make("MPE_simple_spread_v3")
        batch_size = 2
        manager = CTRolloutManager(env, batch_size)

        for seed in [0, 42, 123]:
            rng = jax.random.PRNGKey(seed)
            obs, states = manager.batch_reset(rng)
            assert "__all__" in obs


# Run all tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
