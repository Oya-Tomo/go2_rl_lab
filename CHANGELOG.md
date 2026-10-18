# Changelog

## 0.4.5 - 2026-10-03

1. Replace Neptune with SwanLab logging, including metrics, configs, code snapshots, run naming, CLI options, ignored outputs, and bilingual documentation
2. Fix Newton terrain height scans with `global_world_only` and a `COLLIDE_SHAPES` sensor BVH, excluding robot and visual geometry from height rewards and privileged observations
3. Simplify training/playback runtime bootstrap to direct function calls, preserving Kit import order and adding shutdown and exception cleanup
4. Generate editor settings with Isaac Lab `--editor`, discover Isaac Sim/Kit import paths for Pyright, and ignore generated `pyrightconfig.json`
5. Require an explicit `--task`, set all training tasks to 8,192 environments by default, and simplify README examples to use the default disabled visualization

## 0.4.4 - 2026-10-01

1. Adapt the velocity baseline to Isaac Lab `3.0.0rc1` and Isaac Sim `6.1` on `3.0-ea-vel`
2. Keep velocity PPO and `go2-mixed-moe-cts` with the mixed terrain and flat baseline
3. Use direct body commands `[vx, vy, yaw_rate]` throughout training, export, RoboGauge metadata, and MuJoCo deployment
4. Simplify feed-forward PPO/MoE-CTS batching and preserve sagittal symmetry, observation history, and policy exports
5. Align README usage with `wty-yy/go2_rl_gym` and preserve the complete task framework on `3.0-ea`
6. Keep all validation scripts, reports, and artifacts under ignored local `tests/`
