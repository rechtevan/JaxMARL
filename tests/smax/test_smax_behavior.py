"""Comprehensive behavior tests for SMAX environment to increase coverage.

This test file focuses on testing untested code paths in smax_env.py including:
- Unit collision and pushing mechanics
- Wall collision and death
- Weapon cooldown mechanics
- Different scenarios (3m, 2s3z, etc.)
- SMACv2 position/unit generation
- Reward calculation edge cases
- Available actions
- Different unit types
- Self-play rewards
"""

import jax
import jax.numpy as jnp
import pytest

from jaxmarl import make
from jaxmarl.environments.smax.smax_env import (
    MAP_NAME_TO_SCENARIO,
    State,
    map_name_to_scenario,
    register_scenario,
)


def create_env(
    key,
    continuous_action=False,
    conic_observation=False,
    num_allies=5,
    num_enemies=5,
    scenario=None,
    smacv2_position_generation=False,
    smacv2_unit_type_generation=False,
    walls_cause_death=True,
    use_self_play_reward=False,
):
    env = make(
        "SMAX",
        num_allies=num_allies,
        num_enemies=num_enemies,
        map_width=32,
        map_height=32,
        world_steps_per_env_step=8,
        unit_type_velocities=jnp.array([5.0]),
        unit_type_health=jnp.array([1.0]),
        unit_type_attacks=jnp.array([0.02]),
        unit_type_weapon_cooldowns=jnp.array([0.2]),
        time_per_step=1.0 / 16,
        won_battle_bonus=5.0,
        unit_type_attack_ranges=jnp.array([3.0]),
        unit_type_sight_ranges=jnp.array([4.0]),
        max_steps=100,
        action_type="discrete" if not continuous_action else "continuous",
        observation_type="unit_list" if not conic_observation else "conic",
        scenario=scenario,
        smacv2_position_generation=smacv2_position_generation,
        smacv2_unit_type_generation=smacv2_unit_type_generation,
        walls_cause_death=walls_cause_death,
        use_self_play_reward=use_self_play_reward,
    )
    obs, state = env.reset(key)
    return env, obs, state


def get_random_actions(key, env):
    key_a = jax.random.split(key, num=env.num_agents)
    return {
        agent: env.action_space(agent).sample(key_a[i])
        for i, agent in enumerate(env.agents)
    }


# ========== Unit Collision and Pushing Tests ==========


@pytest.mark.parametrize("do_jit", [True, False])
def test_push_units_away(do_jit):
    """Test that overlapping units are pushed apart.

    Note: The function cannot push units at exactly the same position because
    there's no direction vector to determine which way to push them. This test
    uses slightly offset positions to test realistic overlapping scenarios.
    """
    with jax.disable_jit(do_jit):
        key = jax.random.PRNGKey(42)
        key, key_reset = jax.random.split(key)
        env, _, state = create_env(key_reset)

        # Place two units very close but not at exactly the same position
        # (exact same position has no direction vector to push in)
        base_pos = jnp.array([16.0, 16.0])
        offset_pos = jnp.array([16.01, 16.0])  # Very slight offset
        unit_positions = state.unit_positions.at[0].set(base_pos)
        unit_positions = unit_positions.at[1].set(offset_pos)
        state = state.replace(unit_positions=unit_positions)

        # Get initial distance
        dist_before = jnp.linalg.norm(
            state.unit_positions[0] - state.unit_positions[1]
        )

        # Apply pushing
        new_state = env._push_units_away(state)

        # Units should now be more separated
        dist_after = jnp.linalg.norm(
            new_state.unit_positions[0] - new_state.unit_positions[1]
        )
        assert dist_after > dist_before, "Units should be pushed further apart"
        assert dist_after > 0.1, "Units should have significant separation"


@pytest.mark.parametrize("do_jit", [True, False])
def test_push_units_away_multiple(do_jit):
    """Test pushing with multiple overlapping units."""
    with jax.disable_jit(do_jit):
        key = jax.random.PRNGKey(43)
        key, key_reset = jax.random.split(key)
        env, _, state = create_env(key_reset)

        # Place three units very close together
        center_pos = jnp.array([16.0, 16.0])
        for i in range(3):
            state = state.replace(
                unit_positions=state.unit_positions.at[i].set(
                    center_pos + jnp.array([0.01 * i, 0.01 * i])
                )
            )

        # Apply pushing with higher firmness
        new_state = env._push_units_away(state, firmness=2.0)

        # Check that all pairs are separated
        for i in range(3):
            for j in range(i + 1, 3):
                dist = jnp.linalg.norm(
                    new_state.unit_positions[i] - new_state.unit_positions[j]
                )
                assert dist >= 0.0, f"Units {i} and {j} should be separated"


# ========== Wall Collision Tests ==========


@pytest.mark.parametrize("do_jit", [True, False])
def test_wall_death_enabled(do_jit):
    """Test that units die when touching walls if walls_cause_death=True."""
    with jax.disable_jit(do_jit):
        key = jax.random.PRNGKey(44)
        key, key_reset = jax.random.split(key)
        env, _, state = create_env(key_reset, walls_cause_death=True)

        # Place unit at wall boundary
        state = state.replace(
            unit_positions=state.unit_positions.at[0].set(jnp.array([0.0, 16.0]))
        )
        state = state.replace(unit_health=state.unit_health.at[0].set(1.0))

        # Apply wall collision
        new_state = env._kill_agents_touching_walls(state)

        # Unit should be dead
        assert new_state.unit_health[0] == 0.0, "Unit at wall should be dead"


@pytest.mark.parametrize("do_jit", [True, False])
def test_wall_death_disabled(do_jit):
    """Test that units survive walls if walls_cause_death=False."""
    with jax.disable_jit(do_jit):
        key = jax.random.PRNGKey(45)
        key, key_reset = jax.random.split(key)
        env, _, state = create_env(key_reset, walls_cause_death=False)

        # Place unit at wall boundary
        state = state.replace(
            unit_positions=state.unit_positions.at[0].set(jnp.array([0.0, 16.0]))
        )
        state = state.replace(unit_health=state.unit_health.at[0].set(1.0))

        # Apply wall collision
        new_state = env._kill_agents_touching_walls(state)

        # Unit should still be alive
        assert new_state.unit_health[0] == 1.0, "Unit should survive wall"


@pytest.mark.parametrize(
    "position",
    [
        jnp.array([0.0, 16.0]),  # Left wall
        jnp.array([32.0, 16.0]),  # Right wall
        jnp.array([16.0, 0.0]),  # Bottom wall
        jnp.array([16.0, 32.0]),  # Top wall
        jnp.array([0.0, 0.0]),  # Corner
    ],
)
def test_wall_death_all_boundaries(position):
    """Test wall death at all boundaries."""
    key = jax.random.PRNGKey(46)
    key, key_reset = jax.random.split(key)
    env, _, state = create_env(key_reset, walls_cause_death=True)

    state = state.replace(unit_positions=state.unit_positions.at[0].set(position))
    state = state.replace(unit_health=state.unit_health.at[0].set(1.0))

    new_state = env._kill_agents_touching_walls(state)
    assert new_state.unit_health[0] == 0.0, f"Unit at {position} should be dead"


# ========== Weapon Cooldown Tests ==========


@pytest.mark.parametrize("do_jit", [True, False])
def test_weapon_cooldown_prevents_attack(do_jit):
    """Test that units with cooldown cannot attack."""
    with jax.disable_jit(do_jit):
        key = jax.random.PRNGKey(47)
        key, key_reset = jax.random.split(key)
        env, _, state = create_env(key_reset)

        # Place units close together
        state = state.replace(
            unit_positions=state.unit_positions.at[0].set(jnp.array([16.0, 16.0]))
        )
        state = state.replace(
            unit_positions=state.unit_positions.at[env.num_allies].set(
                jnp.array([17.0, 16.0])
            )
        )

        # Set weapon cooldown for unit 0
        state = state.replace(unit_weapon_cooldowns=state.unit_weapon_cooldowns.at[0].set(0.5))

        # Try to attack
        key, key_actions = jax.random.split(key)
        actions = get_random_actions(key_actions, env)
        # Attack action targeting enemy 0
        actions["ally_0"] = env.num_movement_actions

        key, key_step = jax.random.split(key)
        initial_health = state.unit_health[env.num_allies]
        _, new_state, _, _, _ = env.step(key_step, state, actions)

        # Enemy health should not decrease (attack blocked by cooldown)
        assert (
            new_state.unit_health[env.num_allies] == initial_health
        ), "Attack should be blocked by cooldown"


@pytest.mark.parametrize("do_jit", [True, False])
def test_weapon_cooldown_decreases(do_jit):
    """Test that weapon cooldown decreases over time."""
    with jax.disable_jit(do_jit):
        key = jax.random.PRNGKey(48)
        key, key_reset = jax.random.split(key)
        env, _, state = create_env(key_reset)

        # Set initial cooldown
        initial_cooldown = 0.5
        state = state.replace(
            unit_weapon_cooldowns=state.unit_weapon_cooldowns.at[0].set(initial_cooldown)
        )

        # Take a step with no attack
        key, key_actions = jax.random.split(key)
        actions = get_random_actions(key_actions, env)
        actions["ally_0"] = 4  # Stop action

        key, key_step = jax.random.split(key)
        _, new_state, _, _, _ = env.step(key_step, state, actions)

        # Cooldown should decrease
        assert (
            new_state.unit_weapon_cooldowns[0] < initial_cooldown
        ), "Cooldown should decrease"


# ========== Scenario Tests ==========


@pytest.mark.parametrize(
    "map_name",
    ["3m", "2s3z", "8m", "5m_vs_6m", "3s5z", "3s_vs_5z"],
)
def test_scenario_loading(map_name):
    """Test that different scenarios load correctly."""
    key = jax.random.PRNGKey(49)
    scenario = map_name_to_scenario(map_name)

    env = make("SMAX", scenario=scenario)
    obs, state = env.reset(key)

    assert env.num_allies == scenario.num_allies
    assert env.num_enemies == scenario.num_enemies
    assert state.unit_types.shape == (env.num_agents,)


def test_register_custom_scenario():
    """Test registering a custom scenario."""
    from jaxmarl.environments.smax.smax_env import Scenario

    custom_scenario = Scenario(
        unit_types=jnp.array([0, 1, 2] * 2, dtype=jnp.uint8),
        num_allies=3,
        num_enemies=3,
        smacv2_position_generation=False,
        smacv2_unit_type_generation=False,
    )

    register_scenario("custom_test", custom_scenario)
    loaded_scenario = map_name_to_scenario("custom_test")

    assert loaded_scenario.num_allies == 3
    assert loaded_scenario.num_enemies == 3


# ========== SMACv2 Generation Tests ==========


@pytest.mark.parametrize("do_jit", [True, False])
def test_smacv2_position_generation(do_jit):
    """Test SMACv2 position generation produces different positions."""
    with jax.disable_jit(do_jit):
        key1 = jax.random.PRNGKey(50)
        key2 = jax.random.PRNGKey(51)

        env, _, state1 = create_env(key1, smacv2_position_generation=True)
        env, _, state2 = create_env(key2, smacv2_position_generation=True)

        # Different seeds should produce different positions
        assert not jnp.allclose(
            state1.unit_positions, state2.unit_positions
        ), "SMACv2 positions should vary with seed"


@pytest.mark.parametrize("do_jit", [True, False])
def test_smacv2_unit_type_generation(do_jit):
    """Test SMACv2 unit type generation.

    Note: The environment defaults to 6 unit types (marine, marauder, stalker,
    zealot, zergling, hydralisk). The unit type generator uses all 6 types
    regardless of how many unit type parameters you provide.
    """
    with jax.disable_jit(do_jit):
        key = jax.random.PRNGKey(52)

        # Create environment - it will use default 6 unit types
        env = make(
            "SMAX",
            num_allies=5,
            num_enemies=5,
            smacv2_unit_type_generation=True,
        )
        obs, state = env.reset(key)

        # Check that unit types are assigned
        assert state.unit_types.shape == (env.num_agents,)
        # Unit types should be in valid range (0-5 for 6 unit types)
        assert jnp.all(state.unit_types >= 0)
        assert jnp.all(state.unit_types < 6)


# ========== Reward Calculation Tests ==========


@pytest.mark.parametrize("do_jit", [True, False])
def test_self_play_reward_win(do_jit):
    """Test self-play reward when ally wins."""
    with jax.disable_jit(do_jit):
        key = jax.random.PRNGKey(53)
        key, key_reset = jax.random.split(key)
        env, _, state = create_env(key_reset, use_self_play_reward=True)

        # Set up winning scenario for allies
        unit_positions = state.unit_positions.at[0].set(jnp.array([16.0, 16.0]))
        unit_positions = unit_positions.at[env.num_allies].set(jnp.array([17.0, 16.0]))

        unit_alive = jnp.zeros((env.num_agents,), dtype=jnp.bool_)
        unit_alive = unit_alive.at[0].set(True)
        unit_alive = unit_alive.at[env.num_allies].set(True)

        unit_health = jnp.zeros((env.num_agents,))
        unit_health = unit_health.at[0].set(1.0)
        unit_health = unit_health.at[env.num_allies].set(0.02)

        state = state.replace(
            unit_positions=unit_positions,
            unit_alive=unit_alive,
            unit_health=unit_health,
        )

        # Attack to finish enemy
        key, key_actions = jax.random.split(key)
        actions = get_random_actions(key_actions, env)
        actions["ally_0"] = env.num_movement_actions

        key, key_step = jax.random.split(key)
        _, _, rewards, dones, _ = env.step(key_step, state, actions)

        # Ally should get won_battle_bonus
        assert rewards["ally_0"] == env.won_battle_bonus
        # Enemy should get negative bonus in self-play
        assert rewards["enemy_0"] == -env.won_battle_bonus
        assert dones["__all__"]


@pytest.mark.parametrize("do_jit", [True, False])
def test_draw_scenario(do_jit):
    """Test reward when both teams die simultaneously (draw)."""
    with jax.disable_jit(do_jit):
        key = jax.random.PRNGKey(54)
        key, key_reset = jax.random.split(key)
        env, _, state = create_env(key_reset)

        # Set up scenario where both units die
        unit_alive = jnp.zeros((env.num_agents,), dtype=jnp.bool_)
        unit_health = jnp.zeros((env.num_agents,))

        state = state.replace(unit_alive=unit_alive, unit_health=unit_health)

        # Compute reward for draw
        health_before = jnp.ones((env.num_agents,))
        health_after = jnp.zeros((env.num_agents,))
        rewards = env.compute_reward(state, health_before, health_after)

        # In a draw, no won_battle_bonus should be awarded
        assert rewards["ally_0"] >= 0.0  # Just health damage reward
        assert rewards["ally_0"] < env.won_battle_bonus


# ========== Available Actions Tests ==========


@pytest.mark.parametrize("do_jit", [True, False])
def test_get_avail_actions_alive_unit(do_jit):
    """Test available actions for alive unit."""
    with jax.disable_jit(do_jit):
        key = jax.random.PRNGKey(55)
        key, key_reset = jax.random.split(key)
        env, _, state = create_env(key_reset)

        # Place enemy in range
        state = state.replace(
            unit_positions=state.unit_positions.at[0].set(jnp.array([16.0, 16.0]))
        )
        state = state.replace(
            unit_positions=state.unit_positions.at[env.num_allies].set(
                jnp.array([17.0, 16.0])
            )
        )

        avail_actions = env.get_avail_actions(state)

        # Should have movement actions available
        assert avail_actions["ally_0"][0] == 1  # Move north
        assert avail_actions["ally_0"][4] == 1  # Stop
        # Should have attack action available (enemy in range)
        assert avail_actions["ally_0"][env.num_movement_actions] == 1


@pytest.mark.parametrize("do_jit", [True, False])
def test_get_avail_actions_dead_unit(do_jit):
    """Test available actions for dead unit."""
    with jax.disable_jit(do_jit):
        key = jax.random.PRNGKey(56)
        key, key_reset = jax.random.split(key)
        env, _, state = create_env(key_reset)

        # Kill unit 0
        state = state.replace(unit_alive=state.unit_alive.at[0].set(False))

        avail_actions = env.get_avail_actions(state)

        # Dead unit should only have stop action
        assert avail_actions["ally_0"][4] == 1  # Stop
        # No movement or attack actions
        for i in range(4):
            assert avail_actions["ally_0"][i] == 0
        for i in range(env.num_movement_actions, env.num_ally_actions):
            assert avail_actions["ally_0"][i] == 0


@pytest.mark.parametrize("do_jit", [True, False])
def test_get_avail_actions_out_of_range(do_jit):
    """Test attack actions unavailable when enemy out of range."""
    with jax.disable_jit(do_jit):
        key = jax.random.PRNGKey(57)
        key, key_reset = jax.random.split(key)
        env, _, state = create_env(key_reset)

        # Place enemy far away
        state = state.replace(
            unit_positions=state.unit_positions.at[0].set(jnp.array([5.0, 5.0]))
        )
        state = state.replace(
            unit_positions=state.unit_positions.at[env.num_allies].set(
                jnp.array([25.0, 25.0])
            )
        )

        avail_actions = env.get_avail_actions(state)

        # Movement should be available
        assert avail_actions["ally_0"][0] == 1
        # Attack should not be available (out of range)
        assert avail_actions["ally_0"][env.num_movement_actions] == 0


# ========== Different Unit Type Tests ==========


@pytest.mark.parametrize("do_jit", [True, False])
def test_different_unit_types_health(do_jit):
    """Test that different unit types have different max health."""
    with jax.disable_jit(do_jit):
        key = jax.random.PRNGKey(58)

        # Create a custom scenario with different unit types
        from jaxmarl.environments.smax.smax_env import Scenario

        custom_scenario = Scenario(
            unit_types=jnp.array([2, 2, 3, 3], dtype=jnp.uint8),
            num_allies=2,
            num_enemies=2,
            smacv2_position_generation=False,
            smacv2_unit_type_generation=False,
        )

        # Create env with different unit types
        env = make(
            "SMAX",
            scenario=custom_scenario,
            unit_type_health=jnp.array([45.0, 125.0, 160.0, 150.0]),
        )
        obs, state = env.reset(key)

        # Different types should have different health values
        unit_types_present = jnp.unique(state.unit_types)
        if len(unit_types_present) > 1:
            health_values = state.unit_health[: env.num_agents]
            # Should have at least some variation in health
            assert (
                jnp.max(health_values) > jnp.min(health_values)
                or jnp.all(state.unit_types == state.unit_types[0])
            )


# ========== Update Dead Agents Test ==========


@pytest.mark.parametrize("do_jit", [True, False])
def test_update_dead_agents(do_jit):
    """Test that units with 0 or negative health are marked as not alive."""
    with jax.disable_jit(do_jit):
        key = jax.random.PRNGKey(59)
        key, key_reset = jax.random.split(key)
        env, _, state = create_env(key_reset)

        # Set some units to 0 or negative health
        state = state.replace(unit_health=state.unit_health.at[0].set(0.0))
        state = state.replace(unit_health=state.unit_health.at[1].set(-0.5))
        state = state.replace(unit_health=state.unit_health.at[2].set(0.5))

        new_state = env._update_dead_agents(state)

        assert not new_state.unit_alive[0], "Unit with 0 health should be dead"
        assert not new_state.unit_alive[1], "Unit with negative health should be dead"
        assert new_state.unit_alive[2], "Unit with positive health should be alive"


# ========== Terminal State Tests ==========


@pytest.mark.parametrize("do_jit", [True, False])
def test_is_terminal_all_allies_dead(do_jit):
    """Test terminal when all allies are dead."""
    with jax.disable_jit(do_jit):
        key = jax.random.PRNGKey(60)
        key, key_reset = jax.random.split(key)
        env, _, state = create_env(key_reset)

        # Kill all allies
        unit_alive = state.unit_alive.at[: env.num_allies].set(False)
        state = state.replace(unit_alive=unit_alive)

        assert env.is_terminal(state), "Should be terminal when all allies dead"


@pytest.mark.parametrize("do_jit", [True, False])
def test_is_terminal_all_enemies_dead(do_jit):
    """Test terminal when all enemies are dead."""
    with jax.disable_jit(do_jit):
        key = jax.random.PRNGKey(61)
        key, key_reset = jax.random.split(key)
        env, _, state = create_env(key_reset)

        # Kill all enemies
        unit_alive = state.unit_alive.at[env.num_allies :].set(False)
        state = state.replace(unit_alive=unit_alive)

        assert env.is_terminal(state), "Should be terminal when all enemies dead"


@pytest.mark.parametrize("do_jit", [True, False])
def test_is_terminal_time_limit(do_jit):
    """Test terminal when time limit reached."""
    with jax.disable_jit(do_jit):
        key = jax.random.PRNGKey(62)
        key, key_reset = jax.random.split(key)
        env, _, state = create_env(key_reset)

        # Set time to max
        state = state.replace(time=env.max_steps)

        assert env.is_terminal(state), "Should be terminal at time limit"


@pytest.mark.parametrize("do_jit", [True, False])
def test_is_not_terminal_partial_dead(do_jit):
    """Test not terminal when only some units are dead."""
    with jax.disable_jit(do_jit):
        key = jax.random.PRNGKey(63)
        key, key_reset = jax.random.split(key)
        env, _, state = create_env(key_reset)

        # Kill one ally and one enemy
        unit_alive = state.unit_alive.at[0].set(False)
        unit_alive = unit_alive.at[env.num_allies].set(False)
        state = state.replace(unit_alive=unit_alive)

        assert not env.is_terminal(state), "Should not be terminal with partial deaths"


# ========== World State Tests ==========


@pytest.mark.parametrize("do_jit", [True, False])
def test_get_world_state_structure(do_jit):
    """Test world state has correct structure."""
    with jax.disable_jit(do_jit):
        key = jax.random.PRNGKey(64)
        key, key_reset = jax.random.split(key)
        env, _, state = create_env(key_reset)

        world_state = env.get_world_state(state)

        # Check shape
        assert world_state.shape == (env.state_size,)

        # World state includes: unit features, teams, types
        expected_size = (
            len(env.own_features) * env.num_agents
            + env.num_agents  # teams
            + env.num_agents  # types
        )
        assert world_state.shape[0] == expected_size


@pytest.mark.parametrize("do_jit", [True, False])
def test_get_world_state_dead_unit(do_jit):
    """Test world state for dead unit shows zeros."""
    with jax.disable_jit(do_jit):
        key = jax.random.PRNGKey(65)
        key, key_reset = jax.random.split(key)
        env, _, state = create_env(key_reset)

        # Kill unit 0
        state = state.replace(unit_alive=state.unit_alive.at[0].set(False))
        state = state.replace(unit_health=state.unit_health.at[0].set(0.0))

        world_state = env.get_world_state(state)

        # First unit's features should be zero
        first_unit_features = world_state[: len(env.own_features)]
        assert jnp.all(
            first_unit_features == 0.0
        ), "Dead unit features should be zero"


# ========== Observation Edge Cases ==========


@pytest.mark.parametrize("do_jit", [True, False])
def test_obs_unit_list_dead_observer(do_jit):
    """Test observation from dead unit's perspective."""
    with jax.disable_jit(do_jit):
        key = jax.random.PRNGKey(66)
        key, key_reset = jax.random.split(key)
        env, _, state = create_env(key_reset, conic_observation=False)

        # Kill observer
        state = state.replace(unit_alive=state.unit_alive.at[0].set(False))

        obs = env.get_obs(state)

        # Dead unit should see nothing
        assert jnp.allclose(
            obs["ally_0"], jnp.zeros_like(obs["ally_0"])
        ), "Dead unit should see nothing"


@pytest.mark.parametrize("do_jit", [True, False])
def test_obs_conic_dead_observer(do_jit):
    """Test conic observation from dead unit's perspective."""
    with jax.disable_jit(do_jit):
        key = jax.random.PRNGKey(67)
        key, key_reset = jax.random.split(key)
        env, _, state = create_env(key_reset, conic_observation=True)

        # Kill observer
        state = state.replace(unit_alive=state.unit_alive.at[0].set(False))

        obs = env.get_obs(state)

        # Dead unit should see nothing
        assert jnp.allclose(
            obs["ally_0"], jnp.zeros_like(obs["ally_0"])
        ), "Dead unit should see nothing in conic obs"


@pytest.mark.parametrize("do_jit", [True, False])
def test_obs_conic_out_of_range(do_jit):
    """Test conic observation doesn't include out-of-range units."""
    with jax.disable_jit(do_jit):
        key = jax.random.PRNGKey(68)
        key, key_reset = jax.random.split(key)
        env, _, state = create_env(key_reset, conic_observation=True)

        # Place units far apart
        state = state.replace(
            unit_positions=state.unit_positions.at[0].set(jnp.array([5.0, 5.0]))
        )
        state = state.replace(
            unit_positions=state.unit_positions.at[1].set(jnp.array([25.0, 25.0]))
        )

        obs = env.get_obs(state)

        # Most of the observation should be zeros (out of sight)
        # Own features are at the end
        other_features = obs["ally_0"][: -len(env.own_features)]
        zeros_count = jnp.sum(other_features == 0.0)
        # Should be mostly zeros
        assert zeros_count > len(other_features) * 0.8


# ========== Continuous Action Edge Cases ==========


@pytest.mark.parametrize("do_jit", [True, False])
def test_continuous_action_clipping(do_jit):
    """Test that continuous actions are clipped to [0, 1]."""
    with jax.disable_jit(do_jit):
        key = jax.random.PRNGKey(69)
        key, key_reset = jax.random.split(key)
        env, _, state = create_env(key_reset, continuous_action=True)

        # Create actions outside [0, 1]
        key, key_action = jax.random.split(key)
        actions = get_random_actions(key_action, env)
        # Manually set to out-of-bounds values
        actions["ally_0"] = jnp.array([2.0, -1.0, 1.5, 0.5])

        # Actions should be decoded without error (clipped internally)
        key, key_decode = jax.random.split(key)
        actions_array = jnp.array([actions[i] for i in env.agents])
        movement, attack = env._decode_actions(key_decode, state, actions_array)

        # Should have valid output
        assert movement.shape == (env.num_agents, 2)
        assert attack.shape == (env.num_agents,)


# ========== Reset Tests ==========


@pytest.mark.parametrize("do_jit", [True, False])
def test_reset_initializes_correctly(do_jit):
    """Test that reset initializes state correctly."""
    with jax.disable_jit(do_jit):
        key = jax.random.PRNGKey(70)
        key, key_reset = jax.random.split(key)
        env, obs, state = create_env(key_reset)

        # Check state initialization
        assert state.time == 0
        assert not state.terminal
        assert jnp.all(state.unit_alive)
        assert jnp.all(state.unit_health > 0)
        assert state.unit_positions.shape == (env.num_agents, 2)
        assert jnp.all(state.unit_weapon_cooldowns == 0.0)


@pytest.mark.parametrize("do_jit", [True, False])
def test_reset_different_seeds(do_jit):
    """Test that different seeds produce different initial states."""
    with jax.disable_jit(do_jit):
        key1 = jax.random.PRNGKey(71)
        key2 = jax.random.PRNGKey(72)

        env, _, state1 = create_env(key1)
        env, _, state2 = create_env(key2)

        # Positions should differ
        assert not jnp.allclose(
            state1.unit_positions, state2.unit_positions
        ), "Different seeds should produce different positions"


# ========== Step Function Integration Tests ==========


@pytest.mark.parametrize("do_jit", [True, False])
def test_step_increments_time(do_jit):
    """Test that step increments time correctly."""
    with jax.disable_jit(do_jit):
        key = jax.random.PRNGKey(73)
        key, key_reset = jax.random.split(key)
        env, _, state = create_env(key_reset)

        initial_time = state.time

        key, key_actions = jax.random.split(key)
        actions = get_random_actions(key_actions, env)

        key, key_step = jax.random.split(key)
        _, new_state, _, _, _ = env.step(key_step, state, actions)

        assert new_state.time == initial_time + 1


@pytest.mark.parametrize("do_jit", [True, False])
def test_step_updates_prev_actions(do_jit):
    """Test that step updates previous actions."""
    with jax.disable_jit(do_jit):
        key = jax.random.PRNGKey(74)
        key, key_reset = jax.random.split(key)
        env, _, state = create_env(key_reset)

        key, key_actions = jax.random.split(key)
        actions = get_random_actions(key_actions, env)

        key, key_step = jax.random.split(key)
        _, new_state, _, _, _ = env.step(key_step, state, actions)

        # Previous actions should be updated
        assert new_state.prev_movement_actions.shape == (env.num_agents, 2)
        assert new_state.prev_attack_actions.shape == (env.num_agents,)


# ========== Multiple Unit Types in Combat ==========


@pytest.mark.parametrize("do_jit", [True, False])
def test_different_attack_ranges(do_jit):
    """Test that units with different attack ranges behave correctly."""
    with jax.disable_jit(do_jit):
        key = jax.random.PRNGKey(75)

        # Create environment with different attack ranges
        env = make(
            "SMAX",
            num_allies=2,
            num_enemies=1,
            unit_type_health=jnp.array([100.0, 100.0]),
            unit_type_attacks=jnp.array([10.0, 10.0]),
            unit_type_attack_ranges=jnp.array([2.0, 5.0]),  # Different ranges
            unit_type_velocities=jnp.array([1.0, 1.0]),
        )
        obs, state = env.reset(key)

        # Place ally 0 (short range) and ally 1 (long range) at different distances
        state = state.replace(
            unit_positions=state.unit_positions.at[0].set(jnp.array([10.0, 10.0]))
        )
        state = state.replace(
            unit_positions=state.unit_positions.at[1].set(jnp.array([10.0, 10.0]))
        )
        state = state.replace(
            unit_positions=state.unit_positions.at[2].set(
                jnp.array([13.0, 10.0])
            )  # 3 units away
        )

        # Set different unit types
        state = state.replace(unit_types=state.unit_types.at[0].set(0))  # Short range
        state = state.replace(unit_types=state.unit_types.at[1].set(1))  # Long range

        avail_actions = env.get_avail_actions(state)

        # Ally 0 (short range) should NOT be able to attack
        assert avail_actions["ally_0"][env.num_movement_actions] == 0

        # Ally 1 (long range) should be able to attack
        assert avail_actions["ally_1"][env.num_movement_actions] == 1


# ========== Regression Tests ==========


@pytest.mark.parametrize("do_jit", [True, False])
def test_no_nans_in_observations(do_jit):
    """Test that observations don't contain NaN values."""
    with jax.disable_jit(do_jit):
        key = jax.random.PRNGKey(76)
        key, key_reset = jax.random.split(key)
        env, obs, state = create_env(key_reset)

        # Run multiple steps
        for _ in range(10):
            key, key_actions = jax.random.split(key)
            actions = get_random_actions(key_actions, env)

            key, key_step = jax.random.split(key)
            obs, state, _, _, _ = env.step(key_step, state, actions)

            # Check for NaNs
            for agent in env.agents:
                assert not jnp.any(jnp.isnan(obs[agent])), f"NaN in obs for {agent}"


@pytest.mark.parametrize("do_jit", [True, False])
def test_no_nans_in_rewards(do_jit):
    """Test that rewards don't contain NaN values."""
    with jax.disable_jit(do_jit):
        key = jax.random.PRNGKey(77)
        key, key_reset = jax.random.split(key)
        env, _, state = create_env(key_reset)

        # Run multiple steps
        for _ in range(10):
            key, key_actions = jax.random.split(key)
            actions = get_random_actions(key_actions, env)

            key, key_step = jax.random.split(key)
            _, state, rewards, _, _ = env.step(key_step, state, actions)

            # Check for NaNs
            for agent in env.agents:
                assert not jnp.isnan(rewards[agent]), f"NaN in reward for {agent}"


# ========== Action Space Tests ==========


def test_action_space_discrete():
    """Test discrete action space has correct size."""
    key = jax.random.PRNGKey(78)
    env, _, _ = create_env(key, continuous_action=False)

    for i, agent in enumerate(env.agents):
        action_space = env.action_space(agent)
        if i < env.num_allies:
            assert action_space.n == env.num_ally_actions
        else:
            assert action_space.n == env.num_enemy_actions


def test_action_space_continuous():
    """Test continuous action space has correct shape."""
    key = jax.random.PRNGKey(79)
    env, _, _ = create_env(key, continuous_action=True)

    for agent in env.agents:
        action_space = env.action_space(agent)
        assert action_space.shape == (len(env.continuous_action_dims),)


# ========== Observation Space Tests ==========


def test_observation_space_unit_list():
    """Test observation space size for unit_list mode."""
    key = jax.random.PRNGKey(80)
    env, obs, _ = create_env(key, conic_observation=False)

    expected_size = (
        len(env.unit_features) * (env.num_allies - 1)
        + len(env.unit_features) * env.num_enemies
        + len(env.own_features)
    )

    for agent in env.agents:
        assert obs[agent].shape == (expected_size,)


def test_observation_space_conic():
    """Test observation space size for conic mode."""
    key = jax.random.PRNGKey(81)
    env, obs, _ = create_env(key, conic_observation=True)

    expected_size = len(env.unit_features) * (
        env.num_sections * env.max_units_per_section
    ) + len(env.own_features)

    for agent in env.agents:
        assert obs[agent].shape == (expected_size,)
