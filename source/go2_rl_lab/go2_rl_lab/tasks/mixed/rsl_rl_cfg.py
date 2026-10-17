"""RSL-RL PPO configuration for the Go2 locomotion task."""

from isaaclab.utils.configclass import configclass

from rsl_rl.moe_cts.config import (
    MoeCtsSymmetryCfg,
    RslRlMoeCtsActorCriticCfg,
    RslRlMoeCtsAlgorithmCfg,
    RslRlMoeCtsRunnerCfg,
)
from rsl_rl.ppo.config import RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg, RslRlPpoRunnerCfg


@configclass
class Go2PPORunnerCfg(RslRlPpoRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 100000
    clip_rewards = 10.0
    save_interval = 500
    experiment_name = ""  # same as task name
    obs_groups = {"policy": ["policy"], "critic": ["critic"]}
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class Go2MoECTSRunnerCfg(RslRlMoeCtsRunnerCfg):
    """Reference MoE-CTS hyperparameters for the mixed-terrain Go2 task."""

    num_steps_per_env = 24
    max_iterations = 100000
    clip_rewards = 10.0
    save_interval = 500
    experiment_name = ""  # same as task name
    obs_groups = {"policy": ["policy"], "critic": ["critic"]}
    policy = RslRlMoeCtsActorCriticCfg(
        init_noise_std=1.0,
        noise_std_type="scalar",
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        teacher_encoder_hidden_dims=[512, 256],
        student_encoder_hidden_dims=[512, 256, 256],
        expert_num=8,
        latent_dim=32,
        norm_type="l2norm",
        activation="elu",
    )
    algorithm = RslRlMoeCtsAlgorithmCfg(
        value_loss_coef=1.0,
        load_balance_coef=0.01,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        student_encoder_learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        betas=(0.9, 0.999),
        weight_decay=0.0,
        desired_kl=0.01,
        max_grad_norm=1.0,
        teacher_env_ratio=0.75,
        normalize_advantage_per_mini_batch=False,
        symmetry_cfg=MoeCtsSymmetryCfg(),
    )
