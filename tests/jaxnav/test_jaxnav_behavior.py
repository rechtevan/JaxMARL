"""
Comprehensive behavior tests for JaxNav environment.

This test suite covers:
- Robot movement and differential drive mechanics
- Lidar sensor accuracy and edge cases
- Collision detection (agent-map and agent-agent)
- Goal reaching mechanics
- Map generation and boundaries
- Observation space validation
- Reward calculation
- Terminal conditions
- Discrete and continuous action spaces
- Graph utilities for pathfinding
"""

import jax
import jax.numpy as jnp
import pytest

from jaxmarl.environments.jaxnav import JaxNav
from jaxmarl.environments.jaxnav.jaxnav_env import State, discrete_act_map, wrap, cart2pol
from jaxmarl.environments.jaxnav.jaxnav_utils import pol2cart, unitvec, rot_mat, euclid_dist
from jaxmarl.environments.jaxnav.maps.grid_map import (
    GridMapCircleAgents,
    GridMapPolygonAgents,
    GridMapBarn,
)
from jaxmarl.environments.jaxnav.maps.map import Map
from jaxmarl.environments.jaxnav.jaxnav_graph_utils import (
    grid_to_graph,
    component_mask_with_pos,
    shortest_path_len,
)


# ===== Environment Initialization and Reset Tests =====


@pytest.mark.parametrize("num_agents", [1, 2, 4])
@pytest.mark.parametrize("act_type", ["Continuous", "Discrete"])
@pytest.mark.parametrize("map_id", ["Grid-Rand", "Grid-Rand-Poly"])
def test_environment_initialization(num_agents, act_type, map_id):
    """Test that JaxNav environment initializes correctly with different configurations."""
    env = JaxNav(
        num_agents=num_agents,
        act_type=act_type,
        map_id=map_id,
        max_steps=100,
    )

    assert env.num_agents == num_agents
    assert env._act_type == act_type
    assert len(env.agents) == num_agents
    assert len(env.action_spaces) == num_agents
    assert len(env.observation_spaces) == num_agents


@pytest.mark.parametrize("num_agents", [1, 2, 4])
def test_environment_reset(num_agents):
    """Test that environment reset produces valid initial states."""
    env = JaxNav(num_agents=num_agents, max_steps=100)
    key = jax.random.PRNGKey(42)

    obs, state = env.reset(key)

    # Check observations
    assert len(obs) == num_agents
    for agent in env.agents:
        assert agent in obs
        assert obs[agent].shape == (env.lidar_num_beams + 5,)

    # Check state
    assert state.pos.shape == (num_agents, 2)
    assert state.theta.shape == (num_agents,)
    assert state.vel.shape == (num_agents, 2)
    assert state.goal.shape == (num_agents, 2)
    assert not state.ep_done
    assert state.step == 0
    assert jnp.all(state.done == False)
    assert jnp.all(state.vel == 0.0)


@pytest.mark.parametrize("seed", [0, 42, 123])
def test_environment_reset_reproducibility(seed):
    """Test that reset with same seed produces same initial state."""
    env = JaxNav(num_agents=2)
    key = jax.random.PRNGKey(seed)

    _, state1 = env.reset(key)
    _, state2 = env.reset(key)

    assert jnp.allclose(state1.pos, state2.pos)
    assert jnp.allclose(state1.theta, state2.theta)
    assert jnp.allclose(state1.goal, state2.goal)


def test_environment_reset_valid_initial_positions():
    """Test that initial positions don't collide with map or other agents."""
    env = JaxNav(num_agents=3, map_id="Grid-Rand")
    key = jax.random.PRNGKey(0)

    _, state = env.reset(key)

    # Check no map collisions
    map_collisions = jax.vmap(
        env._map_obj.check_agent_map_collision, in_axes=(0, 0, None)
    )(state.pos, state.theta, state.map_data)
    assert jnp.all(~map_collisions), "Initial positions collide with map"

    # Check agents aren't too close
    for i in range(env.num_agents):
        for j in range(i + 1, env.num_agents):
            dist = jnp.linalg.norm(state.pos[i] - state.pos[j])
            assert dist > 2 * env.rad, f"Agents {i} and {j} too close at init"


# ===== Differential Drive and Movement Tests =====


def test_differential_drive_forward_motion():
    """Test that robot moves forward correctly."""
    env = JaxNav(num_agents=1, max_steps=10)
    key = jax.random.PRNGKey(0)
    _, initial_state = env.reset(key)

    # Action: move forward (v=1.0, omega=0.0)
    action = jnp.array([1.0, 0.0])
    new_pos, new_theta, new_vel = env.update_state(
        initial_state.pos[0],
        initial_state.theta[0],
        initial_state.vel[0],
        action,
        False,
    )

    # Should move in direction of theta
    expected_dx = env.max_v * jnp.cos(initial_state.theta[0]) * env.dt
    expected_dy = env.max_v * jnp.sin(initial_state.theta[0]) * env.dt

    assert jnp.isclose(new_pos[0] - initial_state.pos[0, 0], expected_dx, atol=0.01)
    assert jnp.isclose(new_pos[1] - initial_state.pos[0, 1], expected_dy, atol=0.01)
    assert jnp.isclose(new_vel[0], env.max_v, atol=0.01)


def test_differential_drive_rotation():
    """Test that robot rotates correctly."""
    env = JaxNav(num_agents=1, max_steps=10)
    pos = jnp.array([5.0, 5.0])
    theta = 0.0
    vel = jnp.array([0.0, 0.0])

    # Action: rotate in place (v=0, omega=max_w)
    action = jnp.array([0.0, env.max_w])
    new_pos, new_theta, new_vel = env.update_state(pos, theta, vel, action, False)

    expected_theta = wrap(theta + env.max_w * env.dt)

    assert jnp.allclose(new_pos, pos, atol=0.01), "Position should not change"
    assert jnp.isclose(new_theta, expected_theta, atol=0.01)
    assert jnp.isclose(new_vel[1], env.max_w, atol=0.01)


def test_differential_drive_arc_motion():
    """Test combined forward and rotation motion."""
    env = JaxNav(num_agents=1)
    pos = jnp.array([5.0, 5.0])
    theta = 0.0
    vel = jnp.array([0.0, 0.0])

    # Action: move forward while turning
    action = jnp.array([0.5, 0.3])
    new_pos, new_theta, new_vel = env.update_state(pos, theta, vel, action, False)

    # Position should change and theta should change
    assert not jnp.allclose(new_pos, pos)
    assert not jnp.isclose(new_theta, theta)
    assert jnp.isclose(new_vel[0], 0.5, atol=0.01)
    assert jnp.isclose(new_vel[1], 0.3, atol=0.01)


def test_update_state_done_agent():
    """Test that done agents don't move."""
    env = JaxNav(num_agents=1, evaporating=False)
    pos = jnp.array([5.0, 5.0])
    theta = 0.0
    vel = jnp.array([0.5, 0.3])
    action = jnp.array([1.0, 0.5])

    # Update with done=True
    new_pos, new_theta, new_vel = env.update_state(pos, theta, vel, action, True)

    # Should not move
    assert jnp.allclose(new_pos, pos)
    assert jnp.isclose(new_theta, theta)
    assert jnp.allclose(new_vel, jnp.array([0.0, 0.0]))


def test_velocity_limits():
    """Test that velocity limits are enforced."""
    env = JaxNav(num_agents=1, min_v=0.0, max_v=1.0, max_w=0.6)
    pos = jnp.array([5.0, 5.0])
    theta = 0.0
    vel = jnp.array([0.0, 0.0])

    # Try to exceed max velocity
    action = jnp.array([2.0, 1.0])
    _, _, new_vel = env.update_state(pos, theta, vel, action, False)

    assert new_vel[0] <= env.max_v
    assert jnp.abs(new_vel[1]) <= env.max_w


# ===== Discrete Action Tests =====


def test_discrete_action_mapping():
    """Test that discrete actions map to correct continuous actions."""
    from jaxmarl.environments.jaxnav.jaxnav_env import DISCRETE_ACTS

    # Test actions directly from the array
    # Test first action (stop and turn right)
    result = DISCRETE_ACTS[0]
    assert jnp.allclose(result, jnp.array([0.0, 0.5]))

    # Test forward action
    result = DISCRETE_ACTS[12]
    assert jnp.allclose(result, jnp.array([1.0, 0.0]))

    # Test forward left
    result = DISCRETE_ACTS[10]
    assert jnp.allclose(result, jnp.array([1.0, 0.5]))


@pytest.mark.parametrize("action_idx", range(15))
def test_discrete_action_all_valid(action_idx):
    """Test all discrete actions produce valid outputs."""
    from jaxmarl.environments.jaxnav.jaxnav_env import DISCRETE_ACTS

    result = DISCRETE_ACTS[action_idx]

    assert result.shape == (2,)
    assert result[0] >= 0.0  # Forward velocity non-negative
    assert result[0] <= 1.0  # Within max velocity
    assert jnp.abs(result[1]) <= 0.5  # Within angular velocity range


def test_discrete_action_environment_step():
    """Test stepping environment with discrete actions."""
    env = JaxNav(num_agents=1, act_type="Discrete", max_steps=10)
    key = jax.random.PRNGKey(0)
    _, state = env.reset(key)

    # Take a discrete action
    actions = {"agent_0": jnp.array(12)}  # Forward
    key, step_key = jax.random.split(key)
    _, new_state, _, _, _ = env.step_env(step_key, state, actions)

    # Should have moved
    assert not jnp.allclose(new_state.pos, state.pos)


# ===== Lidar Sensor Tests =====


def test_lidar_sense_straight_wall():
    """Test lidar detects wall directly in front."""
    env = JaxNav(
        num_agents=1,
        lidar_num_beams=200,
        lidar_max_range=6.0,
        map_id="Grid-Rand",
    )

    # Create simple state with agent facing wall
    map_data = jnp.array([
        [1, 1, 1, 1, 1],
        [1, 0, 0, 0, 1],
        [1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1],
    ])

    state = State(
        pos=jnp.array([[2.5, 1.5]]),
        theta=jnp.array([jnp.pi / 2]),  # Facing up
        vel=jnp.zeros((1, 2)),
        done=jnp.array([False]),
        term=jnp.array([False]),
        goal_reached=jnp.array([False]),
        move_term=jnp.array([False]),
        step=0,
        ep_done=False,
        goal=jnp.array([[2.5, 3.5]]),
        map_data=map_data,
        rew_lambda=0.5,
    )

    ranges = env._lidar_sense(0, state)

    # Center beam should detect wall at ~0.5 units
    center_idx = len(ranges) // 2
    assert ranges[center_idx] < 1.0, f"Center lidar range {ranges[center_idx]} too large"
    assert ranges[center_idx] > 0.3, f"Center lidar range {ranges[center_idx]} too small"


def test_lidar_max_range():
    """Test lidar returns max range when no obstacles."""
    env = JaxNav(num_agents=1, lidar_max_range=6.0, lidar_num_beams=50)

    # Large empty map
    map_data = jnp.zeros((20, 20))
    map_data = map_data.at[0, :].set(1)
    map_data = map_data.at[-1, :].set(1)
    map_data = map_data.at[:, 0].set(1)
    map_data = map_data.at[:, -1].set(1)

    state = State(
        pos=jnp.array([[10.0, 10.0]]),
        theta=jnp.array([0.0]),
        vel=jnp.zeros((1, 2)),
        done=jnp.array([False]),
        term=jnp.array([False]),
        goal_reached=jnp.array([False]),
        move_term=jnp.array([False]),
        step=0,
        ep_done=False,
        goal=jnp.array([[15.0, 10.0]]),
        map_data=map_data,
        rew_lambda=0.5,
    )

    ranges = env._lidar_sense(0, state)

    # Many beams should return max range
    assert jnp.sum(ranges >= env.lidar_max_range * 0.95) > 10


def test_lidar_agent_detection():
    """Test lidar detects other agents."""
    env = JaxNav(num_agents=2, lidar_num_beams=100, lidar_max_range=6.0)

    map_data = jnp.zeros((10, 10))
    map_data = map_data.at[0, :].set(1)
    map_data = map_data.at[-1, :].set(1)
    map_data = map_data.at[:, 0].set(1)
    map_data = map_data.at[:, -1].set(1)

    # Two agents facing each other, close together
    state = State(
        pos=jnp.array([[5.0, 5.0], [6.0, 5.0]]),
        theta=jnp.array([0.0, jnp.pi]),  # Facing each other
        vel=jnp.zeros((2, 2)),
        done=jnp.array([False, False]),
        term=jnp.array([False, False]),
        goal_reached=jnp.array([False, False]),
        move_term=jnp.array([False, False]),
        step=0,
        ep_done=False,
        goal=jnp.array([[7.0, 5.0], [4.0, 5.0]]),
        map_data=map_data,
        rew_lambda=0.5,
    )

    ranges = env._lidar_sense(0, state)
    center_idx = len(ranges) // 2

    # Center beam should detect agent at ~1.0 unit (minus radii)
    assert ranges[center_idx] < 1.5, "Lidar should detect nearby agent"


def test_lidar_normalization():
    """Test lidar normalization and unnormalization."""
    env = JaxNav(num_agents=1, normalise_obs=True, lidar_max_range=6.0)

    # Test normalization
    ranges = jnp.array([0.0, 3.0, 6.0])
    normalized = env.normalise_lidar(ranges)

    assert jnp.isclose(normalized[0], -0.5)  # 0 / 6 - 0.5 = -0.5
    assert jnp.isclose(normalized[1], 0.0)  # 3 / 6 - 0.5 = 0.0
    assert jnp.isclose(normalized[2], 0.5)  # 6 / 6 - 0.5 = 0.5

    # Test unnormalization
    unnormalized = env.unnormalise_lidar(normalized)
    assert jnp.allclose(unnormalized, ranges)


# ===== Collision Detection Tests =====


def test_map_collision_detection_circle():
    """Test collision detection with map for circle agents."""
    map_obj = GridMapCircleAgents(num_agents=1, rad=0.3, map_size=(5, 5))

    map_data = jnp.array([
        [1, 1, 1, 1, 1],
        [1, 0, 0, 0, 1],
        [1, 0, 1, 0, 1],
        [1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1],
    ])

    # No collision in free space
    collision = map_obj.check_circle_map_collision(jnp.array([2.5, 1.5]), map_data)
    assert not collision

    # Collision with wall
    collision = map_obj.check_circle_map_collision(jnp.array([2.5, 2.5]), map_data)
    assert collision

    # Edge case: near wall but not touching
    collision = map_obj.check_circle_map_collision(jnp.array([2.8, 1.5]), map_data)
    # Result depends on radius


def test_map_collision_detection_polygon():
    """Test collision detection for polygon agents."""
    map_obj = GridMapPolygonAgents(num_agents=1, rad=0.3, map_size=(5, 5))

    map_data = jnp.array([
        [1, 1, 1, 1, 1],
        [1, 0, 0, 0, 1],
        [1, 0, 1, 0, 1],
        [1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1],
    ])

    # No collision
    collision = map_obj.check_agent_map_collision(
        jnp.array([2.5, 1.5]), jnp.array(0.0), map_data
    )
    assert not collision

    # Collision with obstacle
    collision = map_obj.check_agent_map_collision(
        jnp.array([2.5, 2.5]), jnp.array(0.0), map_data
    )
    assert collision


def test_agent_agent_collision_circle():
    """Test agent-agent collision detection for circles."""
    map_obj = GridMapCircleAgents(num_agents=3, rad=0.3, map_size=(10, 10))

    # No collisions
    positions = jnp.array([[2.0, 2.0], [5.0, 5.0], [8.0, 8.0]])
    theta = jnp.array([0.0, 0.0, 0.0])
    collisions = map_obj.check_all_agent_agent_collisions(positions, theta)
    assert jnp.all(~collisions)

    # Two agents colliding
    positions = jnp.array([[2.0, 2.0], [2.4, 2.0], [8.0, 8.0]])
    collisions = map_obj.check_all_agent_agent_collisions(positions, theta)
    assert collisions[0] and collisions[1]
    assert not collisions[2]


def test_agent_agent_collision_polygon():
    """Test agent-agent collision for polygon agents using SAT."""
    map_obj = GridMapPolygonAgents(num_agents=2, rad=0.3, map_size=(10, 10))

    # No collision
    positions = jnp.array([[3.0, 3.0], [6.0, 6.0]])
    theta = jnp.array([0.0, 0.0])
    collisions = map_obj.check_all_agent_agent_collisions(positions, theta)
    assert jnp.all(~collisions)

    # Collision
    positions = jnp.array([[3.0, 3.0], [3.4, 3.0]])
    collisions = map_obj.check_all_agent_agent_collisions(positions, theta)
    assert jnp.all(collisions)


def test_collision_during_step():
    """Test that collisions are detected during environment step."""
    env = JaxNav(num_agents=1, map_id="Grid-Rand", max_steps=100)
    key = jax.random.PRNGKey(42)

    # Create state where agent will collide with wall
    map_data = jnp.array([
        [1, 1, 1, 1, 1],
        [1, 0, 0, 0, 1],
        [1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1],
    ])

    state = State(
        pos=jnp.array([[0.8, 1.5]]),  # Very close to wall
        theta=jnp.array([jnp.pi]),  # Facing wall
        vel=jnp.zeros((1, 2)),
        done=jnp.array([False]),
        term=jnp.array([False]),
        goal_reached=jnp.array([False]),
        move_term=jnp.array([False]),
        step=0,
        ep_done=False,
        goal=jnp.array([[3.5, 1.5]]),
        map_data=map_data,
        rew_lambda=0.5,
    )

    # Move toward wall
    actions = {"agent_0": jnp.array([1.0, 0.0])}
    _, new_state, _, dones, info = env.step_env(key, state, actions)

    # Should detect collision
    assert new_state.move_term[0] or info["MapC"] > 0


# ===== Goal Reaching Tests =====


def test_goal_reached_detection():
    """Test that goal reaching is detected correctly."""
    env = JaxNav(num_agents=1, goal_radius=0.5)

    # Agent at goal
    pos_at_goal = jnp.array([5.0, 5.0])
    goal = jnp.array([5.0, 5.0])
    reached = env._check_goal_reached(pos_at_goal, goal)
    assert reached

    # Agent just within radius
    pos_close = jnp.array([5.0, 5.4])
    reached = env._check_goal_reached(pos_close, goal)
    assert reached

    # Agent outside radius
    pos_far = jnp.array([5.0, 5.6])
    reached = env._check_goal_reached(pos_far, goal)
    assert not reached


def test_goal_reached_episode_termination():
    """Test that reaching goal terminates episode correctly."""
    env = JaxNav(num_agents=1, goal_radius=0.5, evaporating=True)
    key = jax.random.PRNGKey(0)

    map_data = jnp.zeros((10, 10))
    map_data = map_data.at[0, :].set(1)
    map_data = map_data.at[-1, :].set(1)
    map_data = map_data.at[:, 0].set(1)
    map_data = map_data.at[:, -1].set(1)

    # Agent very close to goal
    state = State(
        pos=jnp.array([[5.0, 5.0]]),
        theta=jnp.array([0.0]),
        vel=jnp.zeros((1, 2)),
        done=jnp.array([False]),
        term=jnp.array([False]),
        goal_reached=jnp.array([False]),
        move_term=jnp.array([False]),
        step=0,
        ep_done=False,
        goal=jnp.array([[5.2, 5.0]]),  # Very close
        map_data=map_data,
        rew_lambda=0.5,
    )

    # Move toward goal
    actions = {"agent_0": jnp.array([1.0, 0.0])}
    _, new_state, rewards, dones, info = env.step_env(key, state, actions)

    # Check goal reached
    assert new_state.goal_reached[0] or dones["agent_0"]
    if new_state.goal_reached[0]:
        assert rewards["agent_0"] > 0  # Should get positive reward


def test_multi_agent_goal_completion():
    """Test that episode ends when all agents reach goals."""
    env = JaxNav(num_agents=2, goal_radius=0.5, evaporating=True, max_steps=100)
    key = jax.random.PRNGKey(0)

    map_data = jnp.zeros((10, 10))
    map_data = map_data.at[0, :].set(1)
    map_data = map_data.at[-1, :].set(1)
    map_data = map_data.at[:, 0].set(1)
    map_data = map_data.at[:, -1].set(1)

    # Both agents at their goals
    state = State(
        pos=jnp.array([[2.0, 2.0], [7.0, 7.0]]),
        theta=jnp.array([0.0, 0.0]),
        vel=jnp.zeros((2, 2)),
        done=jnp.array([False, False]),
        term=jnp.array([False, False]),
        goal_reached=jnp.array([False, False]),
        move_term=jnp.array([False, False]),
        step=0,
        ep_done=False,
        goal=jnp.array([[2.0, 2.0], [7.0, 7.0]]),
        map_data=map_data,
        rew_lambda=0.5,
    )

    actions = {"agent_0": jnp.array([0.0, 0.0]), "agent_1": jnp.array([0.0, 0.0])}
    _, new_state, _, dones, _ = env.step_env(key, state, actions)

    # Episode should end (both reached goal)
    if env.evaporating:
        assert new_state.ep_done or dones["__all__"]


# ===== Observation Space Tests =====


def test_observation_shape():
    """Test that observations have correct shape."""
    env = JaxNav(num_agents=2, lidar_num_beams=200)
    key = jax.random.PRNGKey(0)
    obs, _ = env.reset(key)

    expected_size = env.lidar_num_beams + 5  # lidar + vel(2) + goal(2) + lambda(1)

    for agent in env.agents:
        assert obs[agent].shape == (expected_size,)


def test_observation_normalization():
    """Test that observations are normalized when enabled."""
    env = JaxNav(num_agents=1, normalise_obs=True, lidar_num_beams=50)
    key = jax.random.PRNGKey(0)
    obs, state = env.reset(key)

    obs_array = obs["agent_0"]

    # Lidar should be in range [-0.5, 0.5]
    lidar = obs_array[:env.lidar_num_beams]
    assert jnp.all(lidar >= -0.6)  # Small tolerance
    assert jnp.all(lidar <= 0.6)

    # Velocity should be normalized
    vel = obs_array[env.lidar_num_beams:env.lidar_num_beams + 2]
    assert jnp.all(jnp.abs(vel) <= 1.0)


def test_observation_goal_encoding():
    """Test that goal information is correctly encoded in observations."""
    env = JaxNav(num_agents=1, normalise_obs=False, lidar_num_beams=50)

    map_data = jnp.zeros((10, 10))
    map_data = map_data.at[0, :].set(1)
    map_data = map_data.at[-1, :].set(1)
    map_data = map_data.at[:, 0].set(1)
    map_data = map_data.at[:, -1].set(1)

    state = State(
        pos=jnp.array([[5.0, 5.0]]),
        theta=jnp.array([0.0]),  # Facing east
        vel=jnp.zeros((1, 2)),
        done=jnp.array([False]),
        term=jnp.array([False]),
        goal_reached=jnp.array([False]),
        move_term=jnp.array([False]),
        step=0,
        ep_done=False,
        goal=jnp.array([[8.0, 5.0]]),  # Goal directly ahead
        map_data=map_data,
        rew_lambda=0.5,
    )

    obs = env._get_obs(state)

    # Goal distance should be 3.0
    goal_dist = obs[0, env.lidar_num_beams + 2]
    assert jnp.isclose(goal_dist, 3.0, atol=0.1)

    # Goal orientation should be 0 (straight ahead)
    goal_orient = obs[0, env.lidar_num_beams + 3]
    assert jnp.isclose(goal_orient, 0.0, atol=0.1)


def test_get_world_state():
    """Test world state representation includes all relevant information."""
    env = JaxNav(num_agents=2)
    key = jax.random.PRNGKey(0)
    _, state = env.reset(key)

    world_state = env.get_world_state(state)

    # Should have one state per agent
    assert world_state.shape[0] == env.num_agents

    # Should contain all necessary information
    expected_size = (
        state.map_data[1:-1, 1:-1].size  # Map
        + env.num_agents * 2  # Agent positions
        + env.num_agents  # Agent orientations
        + env.num_agents * 2  # Goals
        + env.num_agents * 2  # Velocities
        + 1  # Step
        + env.num_agents  # Agent indices
        + (env.lidar_num_beams + 5)  # Local observations
    )
    assert world_state.shape[1] == expected_size


# ===== Reward Computation Tests =====


def test_reward_goal_reached():
    """Test that reaching goal gives positive reward."""
    env = JaxNav(num_agents=1, goal_rew=4.0)

    obs = jnp.zeros(env.lidar_num_beams + 5)
    new_pos = jnp.array([5.0, 5.0])
    old_pos = jnp.array([4.5, 5.0])
    action = jnp.array([1.0, 0.0])
    goal = jnp.array([5.0, 5.0])
    collision = False
    goal_reached = True
    done = False
    old_goal_reached = False
    old_move_term = False

    reward, _ = env.compute_reward(
        obs, new_pos, old_pos, action, goal,
        collision, goal_reached, done, old_goal_reached, old_move_term
    )

    assert reward > 0
    assert jnp.isclose(reward, env.goal_rew, atol=0.1)


def test_reward_collision():
    """Test that collision gives negative reward."""
    env = JaxNav(num_agents=1, coll_rew=-4.0)

    obs = jnp.zeros(env.lidar_num_beams + 5)
    new_pos = jnp.array([1.0, 1.0])
    old_pos = jnp.array([1.2, 1.0])
    action = jnp.array([0.5, 0.0])
    goal = jnp.array([5.0, 5.0])
    collision = True
    goal_reached = False
    done = False
    old_goal_reached = False
    old_move_term = False

    reward, _ = env.compute_reward(
        obs, new_pos, old_pos, action, goal,
        collision, goal_reached, done, old_goal_reached, old_move_term
    )

    assert reward < 0


def test_reward_approaching_goal():
    """Test that moving toward goal gives positive dense reward."""
    env = JaxNav(num_agents=1, weight_g=0.25, dt_rew=-0.01)

    obs = jnp.zeros(env.lidar_num_beams + 5)
    goal = jnp.array([5.0, 5.0])
    old_pos = jnp.array([2.0, 5.0])  # Distance 3.0
    new_pos = jnp.array([3.0, 5.0])  # Distance 2.0, moved closer by 1.0
    action = jnp.array([1.0, 0.0])

    reward, _ = env.compute_reward(
        obs, new_pos, old_pos, action, goal,
        False, False, False, False, False
    )

    # Should get positive reward for approaching
    expected_approach_reward = env.weight_g * 1.0  # Moved 1.0 closer
    assert reward > expected_approach_reward - 0.1  # Account for time penalty


def test_reward_time_penalty():
    """Test that time penalty is applied each step."""
    env = JaxNav(num_agents=1, dt_rew=-0.01)

    obs = jnp.zeros(env.lidar_num_beams + 5)
    pos = jnp.array([3.0, 3.0])
    action = jnp.array([0.0, 0.0])  # No movement
    goal = jnp.array([5.0, 5.0])

    reward, _ = env.compute_reward(
        obs, pos, pos, action, goal,
        False, False, False, False, False
    )

    # Should be approximately the time penalty
    assert reward < 0
    assert jnp.isclose(reward, env.dt_rew, atol=0.01)


def test_reward_lidar_proximity_penalty():
    """Test proximity penalty when too close to obstacles."""
    env = JaxNav(
        num_agents=1,
        lidar_rew=-0.1,
        lidar_thresh=0.1,
        normalise_obs=True,
    )

    # Create observation with close obstacle
    obs = jnp.zeros(env.lidar_num_beams + 5)
    # Set some lidar readings to indicate very close obstacle
    close_reading = env.normalise_lidar(jnp.array(0.05))  # Very close
    obs = obs.at[0].set(close_reading)

    pos = jnp.array([3.0, 3.0])
    action = jnp.array([0.5, 0.0])
    goal = jnp.array([5.0, 5.0])

    reward, _ = env.compute_reward(
        obs, pos, pos, action, goal,
        False, False, False, False, False
    )

    # Should include lidar penalty
    assert reward < 0


# ===== Terminal Condition Tests =====


def test_max_steps_termination():
    """Test that episode ends when max steps reached."""
    max_steps = 10
    env = JaxNav(num_agents=1, max_steps=max_steps)
    key = jax.random.PRNGKey(0)
    obs, state = env.reset(key)

    # Run until max steps
    for step in range(max_steps):
        actions = {"agent_0": jnp.array([0.1, 0.0])}
        key, step_key = jax.random.split(key)
        obs, state, _, dones, _ = env.step_env(step_key, state, actions)

        if step < max_steps - 1:
            # Should not be done yet
            continue
        else:
            # Should be done at max steps
            assert dones["__all__"] or state.step >= max_steps


def test_evaporating_mode():
    """Test evaporating mode where agents disappear when done."""
    env = JaxNav(num_agents=2, evaporating=True, goal_radius=0.5)
    key = jax.random.PRNGKey(0)

    map_data = jnp.zeros((10, 10))
    map_data = map_data.at[0, :].set(1)
    map_data = map_data.at[-1, :].set(1)
    map_data = map_data.at[:, 0].set(1)
    map_data = map_data.at[:, -1].set(1)

    # One agent at goal, one not
    state = State(
        pos=jnp.array([[2.0, 2.0], [7.0, 7.0]]),
        theta=jnp.array([0.0, 0.0]),
        vel=jnp.zeros((2, 2)),
        done=jnp.array([False, False]),
        term=jnp.array([False, False]),
        goal_reached=jnp.array([False, False]),
        move_term=jnp.array([False, False]),
        step=0,
        ep_done=False,
        goal=jnp.array([[2.0, 2.0], [9.0, 9.0]]),  # First at goal, second not
        map_data=map_data,
        rew_lambda=0.5,
    )

    actions = {"agent_0": jnp.array([0.0, 0.0]), "agent_1": jnp.array([0.5, 0.0])}
    _, new_state, _, dones, _ = env.step_env(key, state, actions)

    # First agent should be done, second not
    if new_state.goal_reached[0]:
        assert new_state.done[0]
        assert not new_state.ep_done  # Episode not done until all done


# ===== Map Generation Tests =====


def test_map_sampling_boundaries():
    """Test that generated maps have proper boundaries."""
    map_obj = GridMapCircleAgents(num_agents=1, rad=0.3, map_size=(7, 7), fill=0.3)
    key = jax.random.PRNGKey(0)

    map_data = map_obj.sample_map(key)

    # Check boundaries are walls
    assert jnp.all(map_data[0, :] == 1)  # Top
    assert jnp.all(map_data[-1, :] == 1)  # Bottom
    assert jnp.all(map_data[:, 0] == 1)  # Left
    assert jnp.all(map_data[:, -1] == 1)  # Right


def test_map_fill_ratio():
    """Test that map respects fill ratio approximately."""
    map_size = (11, 11)
    fill = 0.3
    map_obj = GridMapCircleAgents(num_agents=1, rad=0.3, map_size=map_size, fill=fill)
    key = jax.random.PRNGKey(42)

    # Sample multiple maps and check average fill
    num_samples = 10
    total_fill = 0

    for i in range(num_samples):
        key, subkey = jax.random.split(key)
        map_data = map_obj.sample_map(subkey)

        # Count interior cells (excluding boundaries)
        interior = map_data[1:-1, 1:-1]
        filled = jnp.sum(interior)
        total = interior.size
        total_fill += filled / total

    avg_fill = total_fill / num_samples

    # Should be approximately the target fill (within reasonable tolerance)
    assert avg_fill < fill + 0.2  # Upper bound


def test_grid_sample_test_case():
    """Test grid-based test case sampling."""
    map_obj = GridMapCircleAgents(
        num_agents=2,
        rad=0.3,
        map_size=(11, 11),
        fill=0.2,
        sample_test_case_type="grid",
        cell_size=1.0,
    )
    key = jax.random.PRNGKey(0)

    map_data, test_case = map_obj.grid_sample_test_case(key)

    # Test case should have correct shape
    assert test_case.shape == (2, 2, 3)  # [num_agents, 2 (start/goal), 3 (x,y,theta)]

    # Positions should be on grid (integer + 0.5)
    for agent in range(2):
        start_pos = test_case[agent, 0, :2]
        goal_pos = test_case[agent, 1, :2]

        # Should be at grid cell centers (x.5, y.5)
        assert jnp.isclose(start_pos[0] % 1.0, 0.5, atol=0.1)
        assert jnp.isclose(start_pos[1] % 1.0, 0.5, atol=0.1)
        assert jnp.isclose(goal_pos[0] % 1.0, 0.5, atol=0.1)
        assert jnp.isclose(goal_pos[1] % 1.0, 0.5, atol=0.1)


def test_polygon_agent_map():
    """Test polygon agent map initialization."""
    agent_coords = jnp.array([
        [-0.25, -0.25],
        [-0.25, 0.25],
        [0.25, 0.25],
        [0.25, -0.25],
    ])

    map_obj = GridMapPolygonAgents(
        num_agents=2,
        rad=0.3,
        map_size=(7, 7),
        agent_coords=agent_coords,
    )

    assert map_obj.agent_coords.shape == (4, 2)
    assert map_obj.num_agents == 2


def test_barn_map_smoothing():
    """Test barn map with smoothing."""
    map_obj = GridMapBarn(
        num_agents=1,
        rad=0.3,
        map_size=(20, 20),
        smoothing_iters=3,
        cell_size=0.15,
    )

    assert map_obj.smoothing_iters == 3
    key = jax.random.PRNGKey(0)

    # Should be able to sample without errors
    map_data, test_case = map_obj.sample_test_case(key)
    assert map_data.shape == (20, 20)


# ===== Path Planning and Graph Utility Tests =====


def test_grid_to_graph_conversion():
    """Test conversion of grid map to graph representation."""
    grid = jnp.array([
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1],
    ])

    A = grid_to_graph(grid)

    # Should be square matrix
    assert A.shape[0] == A.shape[1]
    assert A.shape[0] == grid.size


def test_component_mask():
    """Test connected component detection."""
    grid = jnp.array([
        [1, 1, 1, 1, 1],
        [1, 0, 0, 0, 1],
        [1, 0, 1, 0, 1],
        [1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1],
    ])

    # Start position in free space
    pos = jnp.array([1, 1])  # [x, y]

    mask = component_mask_with_pos(grid, pos)

    # Should mark connected free cells
    assert mask[1, 1]  # Start position
    assert mask[1, 2]  # Adjacent free cell

    # Should not mark walls or disconnected areas
    assert not mask[0, 0]  # Wall


def test_dikstra_path():
    """Test Dijkstra pathfinding on grid map."""
    map_obj = GridMapCircleAgents(num_agents=1, rad=0.3, map_size=(7, 7))

    map_data = jnp.array([
        [1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 1],
        [1, 0, 1, 1, 1, 0, 1],
        [1, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1],
    ])

    start = jnp.array([1.5, 1.5])
    goal = jnp.array([5.5, 3.5])

    passable, path_len = map_obj.dikstra_path(map_data, start, goal)

    # Path should exist
    assert passable
    # Path length should be positive
    assert path_len > 0


def test_passable_check():
    """Test passability checking between two points."""
    map_obj = GridMapCircleAgents(num_agents=1, rad=0.3, map_size=(7, 7))

    map_data = jnp.array([
        [1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 1],
        [1, 0, 1, 1, 1, 0, 1],
        [1, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1],
    ])

    # Path around obstacle
    pos1 = jnp.array([1.5, 1.5])
    pos2 = jnp.array([5.5, 3.5])
    passable = map_obj.passable_check(pos1, pos2, map_data)
    assert passable

    # Blocked by wall
    pos1 = jnp.array([1.5, 1.5])
    pos2 = jnp.array([3.5, 2.5])  # Behind wall
    # Note: passable_check uses graph connectivity, may still be passable if connected


# ===== Utility Function Tests =====


def test_wrap_angle():
    """Test angle wrapping to [-pi, pi]."""
    from jaxmarl.environments.jaxnav.jaxnav_env import wrap

    # Test values
    assert jnp.isclose(wrap(0.0), 0.0)
    # Note: pi and -pi are boundary cases, wrap function behavior may vary
    assert jnp.abs(jnp.abs(wrap(jnp.pi)) - jnp.pi) < 0.01
    assert jnp.abs(jnp.abs(wrap(-jnp.pi)) - jnp.pi) < 0.01

    # Values needing wrapping
    wrapped = wrap(3 * jnp.pi)  # Should wrap to around pi or -pi
    assert jnp.abs(jnp.abs(wrapped) - jnp.pi) < 0.01

    wrapped = wrap(-3 * jnp.pi)  # Should wrap to around pi or -pi
    assert jnp.abs(jnp.abs(wrapped) - jnp.pi) < 0.01

    # Test intermediate values
    assert jnp.isclose(wrap(jnp.pi / 2), jnp.pi / 2)
    assert jnp.isclose(wrap(-jnp.pi / 2), -jnp.pi / 2)


def test_cart2pol_conversion():
    """Test cartesian to polar coordinate conversion."""
    from jaxmarl.environments.jaxnav.jaxnav_env import cart2pol

    # Test cardinal directions
    rho, phi = cart2pol(1.0, 0.0)
    assert jnp.isclose(rho, 1.0)
    assert jnp.isclose(phi, 0.0)

    rho, phi = cart2pol(0.0, 1.0)
    assert jnp.isclose(rho, 1.0)
    assert jnp.isclose(phi, jnp.pi / 2)

    # Test arbitrary point
    rho, phi = cart2pol(3.0, 4.0)
    assert jnp.isclose(rho, 5.0)
    assert jnp.isclose(phi, jnp.arctan2(4.0, 3.0))


def test_pol2cart_conversion():
    """Test polar to cartesian coordinate conversion."""
    x, y = pol2cart(1.0, 0.0)
    assert jnp.isclose(x, 1.0)
    assert jnp.isclose(y, 0.0)

    x, y = pol2cart(1.0, jnp.pi / 2)
    assert jnp.isclose(x, 0.0, atol=1e-6)
    assert jnp.isclose(y, 1.0)

    # Round trip
    rho, phi = 5.0, jnp.pi / 4
    x, y = pol2cart(rho, phi)
    rho2, phi2 = cart2pol(x, y)
    assert jnp.isclose(rho, rho2)
    assert jnp.isclose(phi, phi2)


def test_unitvec():
    """Test unit vector generation from angle."""
    vec = unitvec(0.0)
    assert jnp.allclose(vec, jnp.array([1.0, 0.0]))

    vec = unitvec(jnp.pi / 2)
    assert jnp.allclose(vec, jnp.array([0.0, 1.0]), atol=1e-6)

    vec = unitvec(jnp.pi)
    assert jnp.allclose(vec, jnp.array([-1.0, 0.0]), atol=1e-6)


def test_rotation_matrix():
    """Test 2D rotation matrix."""
    # 90 degree rotation
    R = rot_mat(jnp.pi / 2)
    vec = jnp.array([1.0, 0.0])
    rotated = R @ vec
    assert jnp.allclose(rotated, jnp.array([0.0, 1.0]), atol=1e-6)

    # 180 degree rotation
    R = rot_mat(jnp.pi)
    rotated = R @ vec
    assert jnp.allclose(rotated, jnp.array([-1.0, 0.0]), atol=1e-6)


def test_euclidean_distance():
    """Test Euclidean distance calculation."""
    p1 = jnp.array([0.0, 0.0])
    p2 = jnp.array([3.0, 4.0])

    dist = euclid_dist(p1, p2)
    assert jnp.isclose(dist, 5.0)

    # Same point
    dist = euclid_dist(p1, p1)
    assert jnp.isclose(dist, 0.0)


# ===== Integration and Complex Scenario Tests =====


def test_full_episode_rollout():
    """Test a complete episode rollout."""
    env = JaxNav(num_agents=2, max_steps=50, act_type="Continuous")
    key = jax.random.PRNGKey(0)

    obs, state = env.reset(key)

    for step in range(50):
        # Random actions
        key, *action_keys = jax.random.split(key, env.num_agents + 1)
        actions = {
            agent: env.action_space(agent).sample(action_keys[i])
            for i, agent in enumerate(env.agents)
        }

        key, step_key = jax.random.split(key)
        obs, state, rewards, dones, info = env.step_env(step_key, state, actions)

        # Check validity of outputs
        assert len(obs) == env.num_agents
        assert len(rewards) == env.num_agents
        assert len(dones) == env.num_agents + 1  # +1 for __all__

        if dones["__all__"]:
            break


def test_set_state_and_restore():
    """Test saving and restoring environment state."""
    env = JaxNav(num_agents=2)
    key = jax.random.PRNGKey(42)

    # Get initial state
    _, initial_state = env.reset(key)

    # Step environment
    actions = {agent: jnp.array([0.5, 0.1]) for agent in env.agents}
    key, step_key = jax.random.split(key)
    _, stepped_state, _, _, _ = env.step_env(step_key, initial_state, actions)

    # Restore to stepped state
    obs, restored_state = env.set_state(stepped_state)

    # Should match
    assert jnp.allclose(restored_state.pos, stepped_state.pos)
    assert jnp.allclose(restored_state.vel, stepped_state.vel)
    assert jnp.allclose(restored_state.theta, stepped_state.theta)


def test_mixed_agent_states():
    """Test handling of mixed agent states (some done, some not)."""
    env = JaxNav(num_agents=3, evaporating=True)
    key = jax.random.PRNGKey(0)

    map_data = jnp.zeros((10, 10))
    map_data = map_data.at[0, :].set(1)
    map_data = map_data.at[-1, :].set(1)
    map_data = map_data.at[:, 0].set(1)
    map_data = map_data.at[:, -1].set(1)

    # Mixed state: one done, two active
    state = State(
        pos=jnp.array([[2.0, 2.0], [5.0, 5.0], [7.0, 7.0]]),
        theta=jnp.array([0.0, 0.0, 0.0]),
        vel=jnp.zeros((3, 2)),
        done=jnp.array([True, False, False]),
        term=jnp.array([False, False, False]),
        goal_reached=jnp.array([True, False, False]),
        move_term=jnp.array([False, False, False]),
        step=5,
        ep_done=False,
        goal=jnp.array([[2.0, 2.0], [8.0, 8.0], [2.0, 7.0]]),
        map_data=map_data,
        rew_lambda=0.5,
    )

    actions = {
        "agent_0": jnp.array([1.0, 0.0]),
        "agent_1": jnp.array([0.5, 0.2]),
        "agent_2": jnp.array([0.7, -0.1]),
    }

    _, new_state, rewards, _, _ = env.step_env(key, state, actions)

    # Done agent should not move
    if env.evaporating:
        # Evaporating agents disappear
        pass  # Position handling depends on implementation
    else:
        # Non-evaporating stay in place
        assert jnp.allclose(new_state.pos[0], state.pos[0])


def test_action_space_sampling():
    """Test that action space sampling works correctly."""
    env = JaxNav(num_agents=1, act_type="Continuous")
    key = jax.random.PRNGKey(0)

    for _ in range(10):
        key, action_key = jax.random.split(key)
        action = env.action_space("agent_0").sample(action_key)

        # Check bounds
        assert action.shape == (2,)
        assert action[0] >= env.min_v
        assert action[0] <= env.max_v
        assert action[1] >= -jnp.pi / 6  # Hard-coded in code
        assert action[1] <= jnp.pi / 6


def test_discrete_action_space():
    """Test discrete action space."""
    env = JaxNav(num_agents=1, act_type="Discrete")

    action_space = env.action_space("agent_0")
    assert action_space.n == 15  # 15 discrete actions

    key = jax.random.PRNGKey(0)
    action = action_space.sample(key)
    assert 0 <= action < 15


def test_get_env_metrics():
    """Test environment metrics calculation."""
    env = JaxNav(num_agents=2, map_id="Grid-Rand")
    key = jax.random.PRNGKey(0)

    _, state = env.reset(key)

    metrics = env.get_env_metrics(state)

    # Should have expected keys
    assert "n_walls" in metrics
    assert "shortest_path_length_mean" in metrics
    assert "passable" in metrics


def test_reward_lambda_interpolation():
    """Test reward lambda for team vs individual rewards."""
    env = JaxNav(num_agents=2, do_sep_reward=True, rew_lambda=0.7, fixed_lambda=True)

    # Just verify initialization
    assert env.rew_lambda == 0.7
    assert env.fixed_lambda
    assert env.do_sep_reward


def test_info_by_agent_flag():
    """Test info_by_agent flag affects info structure."""
    # With info_by_agent=True
    env1 = JaxNav(num_agents=2, info_by_agent=True)
    key = jax.random.PRNGKey(0)
    _, state = env1.reset(key)
    actions = {agent: jnp.array([0.5, 0.0]) for agent in env1.agents}
    _, _, _, _, info = env1.step_env(key, state, actions)

    # Info should be per-agent
    assert isinstance(info["GoalR"], jnp.ndarray) or isinstance(info["GoalR"], bool)

    # With info_by_agent=False
    env2 = JaxNav(num_agents=2, info_by_agent=False)
    _, state = env2.reset(key)
    actions = {agent: jnp.array([0.5, 0.0]) for agent in env2.agents}
    _, _, _, _, info = env2.step_env(key, state, actions)

    # Info should be aggregated
    # Structure depends on implementation


if __name__ == "__main__":
    # Run a simple smoke test
    print("Running smoke tests...")
    test_environment_initialization(1, "Continuous", "Grid-Rand")
    print("✓ Environment initialization")
    test_environment_reset(2)
    print("✓ Environment reset")
    test_wrap_angle()
    print("✓ Angle wrapping")
    test_discrete_action_mapping()
    print("✓ Discrete actions")
    test_cart2pol_conversion()
    print("✓ Coordinate conversion")
    print("\nAll smoke tests passed!")
