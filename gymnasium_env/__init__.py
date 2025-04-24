from gymnasium.envs.registration import register

register(
    id="GridWorld3D-v0",
    entry_point="gymnasium_env.envs:GridWorld3DEnv",
)

register(
    id="GridWorld-v0",
    entry_point="gymnasium_env.envs:GridWorldEnv",
)
