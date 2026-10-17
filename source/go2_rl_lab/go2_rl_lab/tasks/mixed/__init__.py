"""Go2 mixed-terrain environment, MDP, and agent configuration."""

import gymnasium as gym

gym.register(
    id="go2-mixed",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "go2_rl_lab.tasks.mixed.env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": "go2_rl_lab.tasks.mixed.env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "go2_rl_lab.tasks.mixed.rsl_rl_cfg:Go2PPORunnerCfg",
    },
)

gym.register(
    id="go2-mixed-moe-cts",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "go2_rl_lab.tasks.mixed.env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": "go2_rl_lab.tasks.mixed.env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "go2_rl_lab.tasks.mixed.rsl_rl_cfg:Go2MoECTSRunnerCfg",
    },
)
