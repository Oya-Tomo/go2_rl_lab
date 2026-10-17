"""Go2 flat-ground velocity locomotion task."""

import gymnasium as gym


gym.register(
    id="go2-flat",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "go2_rl_lab.tasks.flat.env_cfg:FlatVelocityRobotEnvCfg",
        "play_env_cfg_entry_point": "go2_rl_lab.tasks.flat.env_cfg:FlatVelocityRobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "go2_rl_lab.tasks.flat.rsl_rl_cfg:Go2FlatVelocityPPORunnerCfg",
    },
)
