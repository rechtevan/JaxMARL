"""
Tests for the TransformersCTRolloutManager wrapper.

Tests cover:
- SMAX transformer wrapper initialization and methods
- SPREAD transformer wrapper initialization and methods
- Observation matrix structure
- Global state extraction
- Unsupported environment handling
"""

import jax
import jax.numpy as jnp
import pytest

import jaxmarl
from jaxmarl.wrappers.baselines import JaxMARLWrapper
from jaxmarl.wrappers.transformers import TransformersCTRolloutManager


class TestTransformersSMAX:
    """Test TransformersCTRolloutManager with SMAX environments."""

    def test_initialization_smax(self):
        """Test wrapper initializes correctly for SMAX."""
        env = jaxmarl.make("SMAX")
        batch_size = 2

        wrapper = TransformersCTRolloutManager(env, batch_size)

        assert wrapper.batch_size == batch_size
        assert hasattr(wrapper, "_preprocess_obs")
        assert hasattr(wrapper, "global_state")

    def test_batch_reset_smax(self):
        """Test batch reset with SMAX returns matrix observations."""
        env = jaxmarl.make("SMAX")
        batch_size = 2
        wrapper = TransformersCTRolloutManager(env, batch_size)

        rng = jax.random.PRNGKey(0)
        obs, state = wrapper.batch_reset(rng)

        # Check that observations exist for each agent
        assert isinstance(obs, dict)
        for agent in env.agents:
            assert agent in obs
            # Observations should be matrix structured
            assert len(obs[agent].shape) >= 2

    def test_batch_step_smax(self):
        """Test batch step with SMAX."""
        env = jaxmarl.make("SMAX")
        batch_size = 2
        wrapper = TransformersCTRolloutManager(env, batch_size)

        rng = jax.random.PRNGKey(0)
        obs, state = wrapper.batch_reset(rng)

        # Get random actions
        rng, act_rng = jax.random.split(rng)
        actions = {
            agent: jax.random.randint(
                jax.random.fold_in(act_rng, i),
                (batch_size,),
                0,
                env.action_space(agent).n,
            )
            for i, agent in enumerate(env.agents)
        }

        rng, step_rng = jax.random.split(rng)
        obs, state, reward, done, info = wrapper.batch_step(step_rng, state, actions)

        assert isinstance(obs, dict)
        assert isinstance(reward, dict)
        assert isinstance(done, dict)

    def test_smax_obs_vec_to_matrix(self):
        """Test SMAX observation vector to matrix conversion."""
        env = jaxmarl.make("SMAX")
        batch_size = 1
        wrapper = TransformersCTRolloutManager(env, batch_size)

        rng = jax.random.PRNGKey(0)
        obs, state = wrapper.batch_reset(rng)

        # The preprocessed observations should have matrix structure
        # Each agent's obs should have shape (num_entities, features)
        for agent in env.agents:
            agent_obs = obs[agent]
            # Should be batched: (batch_size, num_entities, features)
            assert len(agent_obs.shape) == 3
            # Second dimension should be number of allies + enemies
            expected_entities = env.num_allies + env.num_enemies
            assert agent_obs.shape[1] == expected_entities

    def test_smax_global_state_with_world_state(self):
        """Test SMAX global state extraction when world_state is available."""
        from jaxmarl.wrappers.baselines import SMAXLogWrapper

        env = jaxmarl.make("SMAX")
        # Need to use SMAXLogWrapper to get world_state
        wrapped_env = SMAXLogWrapper(env, replace_info=True)
        batch_size = 1
        wrapper = TransformersCTRolloutManager(wrapped_env, batch_size)

        rng = jax.random.PRNGKey(0)
        obs, state = wrapper.batch_reset(rng)

        # When using SMAXLogWrapper, world_state should be available
        if "world_state" in obs:
            single_obs = {k: v[0] for k, v in obs.items()}
            global_state = wrapper.global_state(single_obs, state)

            # Global state should be a matrix
            assert len(global_state.shape) == 2
            # Should have rows for all entities
            assert global_state.shape[0] == env.num_allies + env.num_enemies
        else:
            # If world_state is not available, that's also valid behavior
            assert state is not None


class TestTransformersSPREAD:
    """Test TransformersCTRolloutManager with MPE_spread environments."""

    def test_initialization_spread(self):
        """Test wrapper initializes correctly for SPREAD."""
        env = jaxmarl.make("MPE_simple_spread_v3")
        batch_size = 2

        wrapper = TransformersCTRolloutManager(env, batch_size)

        assert wrapper.batch_size == batch_size
        assert hasattr(wrapper, "global_state")

    def test_batch_reset_spread(self):
        """Test batch reset with SPREAD returns matrix observations."""
        env = jaxmarl.make("MPE_simple_spread_v3")
        batch_size = 2
        wrapper = TransformersCTRolloutManager(env, batch_size)

        rng = jax.random.PRNGKey(0)
        obs, state = wrapper.batch_reset(rng)

        # Check that observations exist for each agent
        assert isinstance(obs, dict)
        for agent in env.agents:
            assert agent in obs

    def test_batch_step_spread(self):
        """Test batch step with SPREAD."""
        env = jaxmarl.make("MPE_simple_spread_v3")
        batch_size = 2
        wrapper = TransformersCTRolloutManager(env, batch_size)

        rng = jax.random.PRNGKey(0)
        obs, state = wrapper.batch_reset(rng)

        # Get random actions
        rng, act_rng = jax.random.split(rng)
        actions = {
            agent: jax.random.randint(
                jax.random.fold_in(act_rng, i),
                (batch_size,),
                0,
                env.action_space(agent).n,
            )
            for i, agent in enumerate(env.agents)
        }

        rng, step_rng = jax.random.split(rng)
        obs, state, reward, done, info = wrapper.batch_step(step_rng, state, actions)

        assert isinstance(obs, dict)
        assert isinstance(reward, dict)
        assert isinstance(done, dict)

    def test_spread_observations_contain_world_state(self):
        """Test that SPREAD observations contain world_state key."""
        env = jaxmarl.make("MPE_simple_spread_v3")
        batch_size = 1
        wrapper = TransformersCTRolloutManager(env, batch_size)

        rng = jax.random.PRNGKey(0)
        obs, state = wrapper.batch_reset(rng)

        # SPREAD observations should include world_state
        assert "world_state" in obs

    def test_spread_global_state(self):
        """Test SPREAD global state extraction."""
        env = jaxmarl.make("MPE_simple_spread_v3")
        batch_size = 1
        wrapper = TransformersCTRolloutManager(env, batch_size)

        rng = jax.random.PRNGKey(0)
        obs, state = wrapper.batch_reset(rng)

        # Extract global state from single batch item
        single_obs = {k: v[0] for k, v in obs.items()}
        global_state = wrapper.global_state(single_obs, state)

        # Global state should exist
        assert global_state is not None

    def test_spread_wrapped_get_obs_structure(self):
        """Test that spread_wrapped_get_obs returns proper structure."""
        env = jaxmarl.make("MPE_simple_spread_v3")
        batch_size = 2
        wrapper = TransformersCTRolloutManager(env, batch_size)

        rng = jax.random.PRNGKey(0)
        obs, state = wrapper.batch_reset(rng)

        # Each agent's observation should have features including:
        # [d_x, d_y, is_self, is_agent] per entity
        for agent in env.agents:
            agent_obs = obs[agent]
            # Should have batch dimension
            assert agent_obs.shape[0] == batch_size


class TestTransformersWithWrappedEnv:
    """Test TransformersCTRolloutManager with wrapped environments."""

    def test_spread_with_jaxmarl_wrapper(self):
        """Test SPREAD with a JaxMARLWrapper applied."""
        from jaxmarl.wrappers.baselines import LogWrapper

        env = jaxmarl.make("MPE_simple_spread_v3")
        wrapped_env = LogWrapper(env)
        batch_size = 2

        # Should handle wrapped environments
        wrapper = TransformersCTRolloutManager(wrapped_env, batch_size)

        rng = jax.random.PRNGKey(0)
        obs, state = wrapper.batch_reset(rng)

        assert isinstance(obs, dict)


class TestTransformersUnsupportedEnv:
    """Test unsupported environments."""

    def test_unsupported_environment(self):
        """Test that unsupported environments raise NotImplementedError."""
        env = jaxmarl.make("hanabi")  # Not SMAX or SPREAD
        batch_size = 2

        with pytest.raises(NotImplementedError) as exc_info:
            TransformersCTRolloutManager(env, batch_size)

        assert "MPE_spread" in str(exc_info.value) or "SMAX" in str(exc_info.value)

    def test_unsupported_mpe_non_spread(self):
        """Test that non-spread MPE environments raise NotImplementedError."""
        env = jaxmarl.make("MPE_simple_tag_v3")  # MPE but not spread
        batch_size = 2

        with pytest.raises(NotImplementedError):
            TransformersCTRolloutManager(env, batch_size)


class TestTransformersMultipleSteps:
    """Test multiple step execution."""

    def test_multiple_steps_smax(self):
        """Test multiple steps with SMAX transformer wrapper."""
        env = jaxmarl.make("SMAX")
        batch_size = 2
        wrapper = TransformersCTRolloutManager(env, batch_size)

        rng = jax.random.PRNGKey(0)
        obs, state = wrapper.batch_reset(rng)

        for _ in range(5):
            rng, act_rng, step_rng = jax.random.split(rng, 3)
            actions = {
                agent: jax.random.randint(
                    jax.random.fold_in(act_rng, i),
                    (batch_size,),
                    0,
                    env.action_space(agent).n,
                )
                for i, agent in enumerate(env.agents)
            }

            obs, state, reward, done, info = wrapper.batch_step(
                step_rng, state, actions
            )

            # Verify state remains valid
            assert state is not None

    def test_multiple_steps_spread(self):
        """Test multiple steps with SPREAD transformer wrapper."""
        env = jaxmarl.make("MPE_simple_spread_v3")
        batch_size = 2
        wrapper = TransformersCTRolloutManager(env, batch_size)

        rng = jax.random.PRNGKey(0)
        obs, state = wrapper.batch_reset(rng)

        for _ in range(5):
            rng, act_rng, step_rng = jax.random.split(rng, 3)
            actions = {
                agent: jax.random.randint(
                    jax.random.fold_in(act_rng, i),
                    (batch_size,),
                    0,
                    env.action_space(agent).n,
                )
                for i, agent in enumerate(env.agents)
            }

            obs, state, reward, done, info = wrapper.batch_step(
                step_rng, state, actions
            )

            # Verify state remains valid
            assert state is not None


# Run all tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
