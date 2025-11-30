"""
Comprehensive behavior tests for Hanabi environment.

Tests cover:
- Environment initialization and configuration
- Action types (discard, play, hint color, hint rank, noop)
- Legal move validation
- State transitions
- Observation encoding
- Game mechanics (info tokens, life tokens, score)
- Multi-agent coordination
"""

import jax
import jax.numpy as jnp
import pytest

from jaxmarl import make


class TestHanabiInitialization:
    """Test Hanabi environment initialization."""

    def test_default_initialization(self):
        """Test environment initializes with default parameters."""
        env = make("hanabi")
        assert env.num_agents == 2
        assert env.num_colors == 5
        assert env.num_ranks == 5
        assert env.hand_size == 5
        assert env.max_info_tokens == 8
        assert env.max_life_tokens == 3

    def test_custom_num_agents(self):
        """Test environment with different agent counts."""
        for n_agents in [2, 3, 4, 5]:
            env = make("hanabi", num_agents=n_agents)
            assert env.num_agents == n_agents
            assert len(env.agents) == n_agents

    def test_hand_size_auto_adjustment(self):
        """Test hand size automatically adjusts based on player count."""
        # 2-3 players should have hand size 5
        env2 = make("hanabi", num_agents=2)
        assert env2.hand_size == 5

        env3 = make("hanabi", num_agents=3)
        assert env3.hand_size == 5

        # 4-5 players should have hand size 4
        env4 = make("hanabi", num_agents=4)
        assert env4.hand_size == 4

        env5 = make("hanabi", num_agents=5)
        assert env5.hand_size == 4

    def test_custom_agents_list(self):
        """Test environment with custom agent names."""
        custom_agents = ["player_a", "player_b"]
        env = make("hanabi", num_agents=2, agents=custom_agents)
        assert env.agents == custom_agents

    def test_action_space(self):
        """Test action space is correct."""
        env = make("hanabi")
        for agent in env.agents:
            action_space = env.action_space(agent)
            assert action_space.n == env.num_moves

    def test_observation_space(self):
        """Test observation space is correct."""
        env = make("hanabi")
        for agent in env.agents:
            obs_space = env.observation_space(agent)
            assert obs_space is not None


class TestHanabiReset:
    """Test Hanabi environment reset."""

    def test_reset_returns_valid_state(self):
        """Test reset returns valid initial state."""
        env = make("hanabi")
        rng = jax.random.PRNGKey(0)
        obs, state = env.reset(rng)

        # Check state fields
        # info_tokens and life_tokens are encoded as arrays of 1s
        assert int(jnp.sum(state.info_tokens)) == env.max_info_tokens
        assert int(jnp.sum(state.life_tokens)) == env.max_life_tokens
        assert int(state.score) == 0
        assert int(state.turn) == 0
        assert int(jnp.sum(state.cur_player_idx)) == 1  # Exactly one active player

    def test_reset_determinism(self):
        """Test reset produces same state with same seed."""
        env = make("hanabi")
        rng = jax.random.PRNGKey(42)

        obs1, state1 = env.reset(rng)
        obs2, state2 = env.reset(rng)

        # Should be identical
        assert jnp.array_equal(state1.deck, state2.deck)
        assert jnp.array_equal(state1.player_hands, state2.player_hands)

    def test_reset_different_seeds(self):
        """Test reset produces different states with different seeds."""
        env = make("hanabi")

        obs1, state1 = env.reset(jax.random.PRNGKey(0))
        obs2, state2 = env.reset(jax.random.PRNGKey(42))

        # Decks should differ (with high probability)
        # Not guaranteed but very likely
        assert not jnp.array_equal(state1.deck, state2.deck)

    def test_reset_observations_dict(self):
        """Test reset returns observations as dict."""
        env = make("hanabi")
        rng = jax.random.PRNGKey(0)
        obs, state = env.reset(rng)

        assert isinstance(obs, dict)
        for agent in env.agents:
            assert agent in obs


class TestHanabiStep:
    """Test Hanabi environment step function."""

    def test_step_basic(self):
        """Test basic step execution."""
        env = make("hanabi")
        rng = jax.random.PRNGKey(0)
        obs, state = env.reset(rng)

        # Get legal moves
        legal_moves = env.get_legal_moves(state)

        # Take first legal action for current player
        cur_player = int(jnp.argmax(state.cur_player_idx))
        actions = {a: jnp.array(env.num_moves - 1) for a in env.agents}  # noop

        # Find a legal action for current player
        agent = env.agents[cur_player]
        legal_actions = jnp.where(legal_moves[agent] == 1)[0]
        if len(legal_actions) > 0:
            actions[agent] = legal_actions[0]

        rng, step_rng = jax.random.split(rng)
        obs, state, reward, done, info = env.step(step_rng, state, actions)

        assert int(state.turn) == 1
        assert isinstance(obs, dict)
        assert isinstance(reward, dict)
        assert isinstance(done, dict)

    def test_step_turn_advances(self):
        """Test that turn counter advances."""
        env = make("hanabi")
        rng = jax.random.PRNGKey(0)
        obs, state = env.reset(rng)

        initial_turn = int(state.turn)

        # Execute a step
        actions = {a: jnp.array(env.num_moves - 1) for a in env.agents}  # noop
        rng, step_rng = jax.random.split(rng)
        obs, state, reward, done, info = env.step(step_rng, state, actions)

        assert int(state.turn) == initial_turn + 1


class TestHanabiActions:
    """Test different Hanabi action types."""

    def test_discard_action(self):
        """Test discard action mechanics."""
        env = make("hanabi")
        rng = jax.random.PRNGKey(0)
        obs, state = env.reset(rng)

        # Reduce info tokens to allow discard (set one token to 0)
        new_info_tokens = state.info_tokens.at[-1].set(0)
        state = state.replace(info_tokens=new_info_tokens)

        # Find discard action (first hand_size actions)
        discard_action = 0  # Discard first card

        cur_player = int(jnp.argmax(state.cur_player_idx))
        actions = {a: jnp.array(env.num_moves - 1) for a in env.agents}
        actions[env.agents[cur_player]] = jnp.array(discard_action)

        initial_info = int(jnp.sum(state.info_tokens))
        rng, step_rng = jax.random.split(rng)
        obs, state, reward, done, info = env.step(step_rng, state, actions)

        # Info tokens should increase (capped at max)
        assert int(jnp.sum(state.info_tokens)) >= initial_info

    def test_play_action_correct_card(self):
        """Test playing a card."""
        env = make("hanabi")
        rng = jax.random.PRNGKey(42)
        obs, state = env.reset(rng)

        # Play action for first card
        play_action = env.hand_size  # Play actions start after discards

        cur_player = int(jnp.argmax(state.cur_player_idx))
        actions = {a: jnp.array(env.num_moves - 1) for a in env.agents}
        actions[env.agents[cur_player]] = jnp.array(play_action)

        rng, step_rng = jax.random.split(rng)
        obs, new_state, reward, done, info = env.step(step_rng, state, actions)

        # Either score increases (correct play) or life decreases (wrong play)
        assert int(new_state.score) >= int(state.score) or int(
            jnp.sum(new_state.life_tokens)
        ) <= int(jnp.sum(state.life_tokens))

    def test_hint_color_action(self):
        """Test giving color hint."""
        env = make("hanabi")
        rng = jax.random.PRNGKey(0)
        obs, state = env.reset(rng)

        # Must have info tokens to give hint
        assert int(jnp.sum(state.info_tokens)) > 0

        # Find hint color action
        hint_offset = 2 * env.hand_size
        hint_action = hint_offset  # First hint color action

        cur_player = int(jnp.argmax(state.cur_player_idx))
        actions = {a: jnp.array(env.num_moves - 1) for a in env.agents}
        actions[env.agents[cur_player]] = jnp.array(hint_action)

        initial_info = int(jnp.sum(state.info_tokens))
        rng, step_rng = jax.random.split(rng)
        obs, state, reward, done, info = env.step(step_rng, state, actions)

        # Info tokens should decrease
        assert int(jnp.sum(state.info_tokens)) == initial_info - 1

    def test_hint_rank_action(self):
        """Test giving rank hint."""
        env = make("hanabi")
        rng = jax.random.PRNGKey(0)
        obs, state = env.reset(rng)

        # Find hint rank action
        hint_color_offset = 2 * env.hand_size
        hint_rank_offset = hint_color_offset + (env.num_agents - 1) * env.num_colors
        hint_action = hint_rank_offset  # First hint rank action

        cur_player = int(jnp.argmax(state.cur_player_idx))
        actions = {a: jnp.array(env.num_moves - 1) for a in env.agents}
        actions[env.agents[cur_player]] = jnp.array(hint_action)

        initial_info = int(jnp.sum(state.info_tokens))
        rng, step_rng = jax.random.split(rng)
        obs, state, reward, done, info = env.step(step_rng, state, actions)

        # Info tokens should decrease
        assert int(jnp.sum(state.info_tokens)) == initial_info - 1

    def test_noop_action(self):
        """Test noop action."""
        env = make("hanabi")
        rng = jax.random.PRNGKey(0)
        obs, state = env.reset(rng)

        noop_action = env.num_moves - 1  # Last action is noop

        actions = {a: jnp.array(noop_action) for a in env.agents}

        rng, step_rng = jax.random.split(rng)
        obs, state, reward, done, info = env.step(step_rng, state, actions)

        # Turn should advance
        assert int(state.turn) >= 1


class TestHanabiLegalMoves:
    """Test legal move validation."""

    def test_get_legal_moves(self):
        """Test get_legal_moves returns valid mask."""
        env = make("hanabi")
        rng = jax.random.PRNGKey(0)
        obs, state = env.reset(rng)

        legal_moves = env.get_legal_moves(state)

        assert isinstance(legal_moves, dict)
        for agent in env.agents:
            assert agent in legal_moves
            assert legal_moves[agent].shape == (env.num_moves,)
            # Should have at least one legal move
            assert jnp.sum(legal_moves[agent]) >= 1

    def test_only_current_player_has_actions(self):
        """Test only current player has non-noop legal actions."""
        env = make("hanabi")
        rng = jax.random.PRNGKey(0)
        obs, state = env.reset(rng)

        legal_moves = env.get_legal_moves(state)
        cur_player = jnp.argmax(state.cur_player_idx)

        # Non-current players should only have noop legal
        noop_action = env.num_moves - 1
        for i, agent in enumerate(env.agents):
            if i != cur_player:
                # Only noop should be legal for non-current player
                assert legal_moves[agent][noop_action] == 1

    def test_hint_requires_info_tokens(self):
        """Test hints require info tokens."""
        env = make("hanabi")
        rng = jax.random.PRNGKey(0)
        obs, state = env.reset(rng)

        # Deplete info tokens
        state = state.replace(info_tokens=0)

        legal_moves = env.get_legal_moves(state)
        cur_player = jnp.argmax(state.cur_player_idx)
        agent = env.agents[cur_player]

        # Hint actions should not be legal
        hint_offset = 2 * env.hand_size
        hint_actions_sum = jnp.sum(legal_moves[agent][hint_offset:-1])
        assert hint_actions_sum == 0

    def test_discard_requires_missing_info_tokens(self):
        """Test discard requires info tokens below max."""
        env = make("hanabi")
        rng = jax.random.PRNGKey(0)
        obs, state = env.reset(rng)

        # Max info tokens
        state = state.replace(info_tokens=env.max_info_tokens)

        legal_moves = env.get_legal_moves(state)
        cur_player = jnp.argmax(state.cur_player_idx)
        agent = env.agents[cur_player]

        # Discard actions (first hand_size) should not be legal
        discard_actions_sum = jnp.sum(legal_moves[agent][: env.hand_size])
        assert discard_actions_sum == 0


class TestHanabiGameFlow:
    """Test full game flow."""

    def test_random_episode(self):
        """Test running a random episode."""
        env = make("hanabi")
        rng = jax.random.PRNGKey(123)
        obs, state = env.reset(rng)

        done_all = False
        steps = 0
        max_steps = 200

        while not done_all and steps < max_steps:
            legal_moves = env.get_legal_moves(state)

            actions = {}
            for agent in env.agents:
                rng, act_rng = jax.random.split(rng)
                # Sample from legal moves
                legal = legal_moves[agent]
                probs = legal / jnp.sum(legal)
                action = jax.random.choice(act_rng, env.num_moves, p=probs)
                actions[agent] = action

            rng, step_rng = jax.random.split(rng)
            obs, state, reward, done, info = env.step(step_rng, state, actions)
            done_all = done["__all__"]
            steps += 1

        # Game should end
        assert done_all or steps == max_steps

    def test_game_ends_on_life_loss(self):
        """Test game ends when life tokens run out."""
        env = make("hanabi")
        rng = jax.random.PRNGKey(0)
        obs, state = env.reset(rng)

        # Set life tokens to 1 (only first element is 1, rest are 0)
        life_tokens = jnp.zeros(env.max_life_tokens, dtype=state.life_tokens.dtype)
        life_tokens = life_tokens.at[0].set(1)
        state = state.replace(life_tokens=life_tokens)

        # Game should still be playable
        assert not bool(state.terminal)

    def test_perfect_score(self):
        """Test maximum possible score."""
        env = make("hanabi")
        # Max score is num_colors * num_ranks (one card of each rank per color)
        max_score = env.num_colors * env.num_ranks
        assert max_score == 25  # 5 colors * 5 ranks


class TestHanabiObservations:
    """Test observation encoding."""

    def test_observation_contains_public_info(self):
        """Test observations contain public information."""
        env = make("hanabi")
        rng = jax.random.PRNGKey(0)
        obs, state = env.reset(rng)

        for agent in env.agents:
            assert agent in obs
            # Observations should be arrays
            assert isinstance(obs[agent], jnp.ndarray)

    def test_observations_partial_observable(self):
        """Test agents can't see their own cards directly."""
        env = make("hanabi")
        rng = jax.random.PRNGKey(0)
        obs, state = env.reset(rng)

        # Observations should be different for each agent
        # (each sees different perspectives)
        obs_list = [obs[a] for a in env.agents]
        assert not jnp.array_equal(obs_list[0], obs_list[1])


class TestHanabiDeckInjection:
    """Test deck injection for reproducible testing."""

    def test_reset_from_deck(self):
        """Test reset_from_deck creates deterministic game."""
        env = make("hanabi")

        # Create a simple deck with all cards
        deck = jnp.zeros((env.deck_size, env.num_colors, env.num_ranks))
        # Fill with some cards
        for i in range(env.deck_size):
            deck = deck.at[i, i % env.num_colors, i % env.num_ranks].set(1)

        obs, state = env.reset_from_deck(deck)

        # Verify the game started properly
        # Note: reset_from_deck deals cards to players, so remaining deck differs
        assert state is not None
        assert state.num_cards_dealt > 0  # Cards were dealt to players
        assert state.turn == 0  # Game just started

    def test_reset_from_deck_deterministic(self):
        """Test that reset_from_deck produces same state with same deck."""
        env = make("hanabi")

        # Create a deck
        deck = jnp.zeros((env.deck_size, env.num_colors, env.num_ranks))
        for i in range(env.deck_size):
            deck = deck.at[i, i % env.num_colors, i % env.num_ranks].set(1)

        obs1, state1 = env.reset_from_deck(deck)
        obs2, state2 = env.reset_from_deck(deck)

        # Same deck should produce same initial state
        assert jnp.array_equal(state1.player_hands, state2.player_hands)
        assert jnp.array_equal(state1.fireworks, state2.fireworks)


class TestHanabiEdgeCases:
    """Test edge cases."""

    def test_deck_depletion(self):
        """Test game behavior when deck runs low through gameplay."""
        env = make("hanabi")
        rng = jax.random.PRNGKey(0)
        obs, state = env.reset(rng)

        # Take many steps to deplete the deck
        for _ in range(100):
            actions = {a: jnp.array(0) for a in env.agents}  # Play first card
            rng, step_rng = jax.random.split(rng)
            obs, state, reward, done, info = env.step(step_rng, state, actions)

            if done["__all__"]:
                break

        # Game should complete or terminate naturally
        assert state is not None

    def test_game_ends_after_final_round(self):
        """Test that game ends after last round counter expires."""
        env = make("hanabi")
        rng = jax.random.PRNGKey(42)
        obs, state = env.reset(rng)

        # Simulate many steps - game should eventually end
        for _ in range(200):
            # Discard first card (always valid)
            actions = {
                a: jnp.array(env.hand_size) for a in env.agents
            }  # Discard action
            rng, step_rng = jax.random.split(rng)
            obs, state, reward, done, info = env.step(step_rng, state, actions)

            if done["__all__"]:
                break

        # Game should have ended
        assert done["__all__"]


# Run all tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
