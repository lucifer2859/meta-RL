from gym.envs.registration import register

# Bandit
# ----------------------------------------

for k in [5, 10, 50, 100, 500]:
    register(
        'Bandit-K{0}-v0'.format(k),
        entry_point='helper.envs.bandit:BernoulliBanditEnv',
        kwargs={'k': k}
    )

# TabularMDP
# ----------------------------------------

register(
    'TabularMDP-v0',
    entry_point='helper.envs.mdp:TabularMDPEnv',
    kwargs={'num_states': 10, 'num_actions': 5},
    max_episode_steps=10
)

# 2D Navigation
# ----------------------------------------

register(
    '2DNavigation-v0',
    entry_point='helper.envs.navigation:Navigation2DEnv',
    max_episode_steps=100
)
