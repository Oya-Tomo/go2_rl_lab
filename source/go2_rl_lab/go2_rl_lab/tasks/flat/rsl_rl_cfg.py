"""PPO configuration for the flat velocity-command baseline."""

from isaaclab.utils.configclass import configclass

from go2_rl_lab.tasks.mixed.rsl_rl_cfg import Go2PPORunnerCfg


@configclass
class Go2FlatVelocityPPORunnerCfg(Go2PPORunnerCfg):
    """Use the mixed-task PPO hyperparameters on flat ground."""
