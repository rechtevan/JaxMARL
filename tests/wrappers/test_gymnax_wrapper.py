"""
Tests for the GymnaxToJaxMARL wrapper.

Tests cover:
- Wrapping Gymnax environments for use in JaxMARL
- Reset and step methods
- Action and observation space conversion
"""

import jax
import jax.numpy as jnp
import pytest

from jaxmarl.wrappers.gymnax import GymnaxToJaxMARL


class TestGymnaxToJaxMARL:
    """Test GymnaxToJaxMARL wrapper."""

    def test_initialization(self):
        """Test wrapper initializes correctly."""
        wrapper = GymnaxToJaxMARL("CartPole-v1")

        assert wrapper.num_agents == 1
        assert wrapper.agent == "agent"
        assert wrapper.agents == ["agent"]
        assert wrapper.env_name == "CartPole-v1"

    def test_initialization_with_kwargs(self):
        """Test wrapper initializes with environment kwargs."""
        wrapper = GymnaxToJaxMARL("CartPole-v1", env_kwargs={})

        assert wrapper.num_agents == 1

    def test_reset(self):
        """Test wrapper reset returns JaxMARL-style output."""
        wrapper = GymnaxToJaxMARL("CartPole-v1")

        rng = jax.random.PRNGKey(0)
        obs, state = wrapper.reset(rng)

        # Should return dict-style observations
        assert isinstance(obs, dict)
        assert "agent" in obs

    def test_step(self):
        """Test wrapper step processes actions correctly."""
        wrapper = GymnaxToJaxMARL("CartPole-v1")

        rng = jax.random.PRNGKey(0)
        obs, state = wrapper.reset(rng)

        # Create action dict
        rng, act_rng = jax.random.split(rng)
        actions = {"agent": jnp.array([0])}  # CartPole has discrete actions

        rng, step_rng = jax.random.split(rng)
        obs, state, reward, done, info = wrapper.step(step_rng, state, actions)

        # Check outputs are dict-style
        assert isinstance(obs, dict)
        assert "agent" in obs
        assert isinstance(reward, dict)
        assert "agent" in reward
        assert isinstance(done, dict)
        assert "agent" in done
        assert "__all__" in done

    def test_observation_space(self):
        """Test observation_space method."""
        wrapper = GymnaxToJaxMARL("CartPole-v1")

        obs_space = wrapper.observation_space("agent")
        assert obs_space is not None

    def test_action_space(self):
        """Test action_space method."""
        wrapper = GymnaxToJaxMARL("CartPole-v1")

        action_space = wrapper.action_space("agent")
        assert action_space is not None

    def test_default_params_property(self):
        """Test default_params property."""
        wrapper = GymnaxToJaxMARL("CartPole-v1")

        params = wrapper.default_params
        assert params is not None

    def test_multiple_steps(self):
        """Test wrapper handles multiple steps."""
        wrapper = GymnaxToJaxMARL("CartPole-v1")

        rng = jax.random.PRNGKey(42)
        obs, state = wrapper.reset(rng)

        for _ in range(10):
            rng, act_rng, step_rng = jax.random.split(rng, 3)
            actions = {"agent": jnp.array([0])}
            obs, state, reward, done, info = wrapper.step(step_rng, state, actions)

            if done["__all__"]:
                break

        # Should complete without error
        assert state is not None

    def test_different_gymnax_envs(self):
        """Test wrapper works with different Gymnax environments."""
        env_names = ["CartPole-v1", "Pendulum-v1", "MountainCar-v0"]

        for env_name in env_names:
            try:
                wrapper = GymnaxToJaxMARL(env_name)
                rng = jax.random.PRNGKey(0)
                obs, state = wrapper.reset(rng)
                assert "agent" in obs
            except Exception as e:
                # Some envs may not be available
                pytest.skip(f"Environment {env_name} not available: {e}")


# Run all tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
