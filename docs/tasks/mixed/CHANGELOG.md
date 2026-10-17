# CHANGELOG

## v1.3.2 - 2026-10-12

1. Add position-based terrain detection and per-step velocity limits to the shared `RLGymVelCommand` for `go2-flat`, `go2-mixed`, and `go2-mixed-moe-cts`, preserving requested commands for speed recovery and assigned terrain masks for curriculum statistics
2. Fix the `base_height_l2` target to a constant `0.38 m` (`BASE_HEIGHT_TARGET`): remove the command-conditioned target and its `command_speed_range` interpolation, deleting `BASE_HEIGHT_RANGE`/`BASE_HEIGHT_COMMAND_SPEED_RANGE`; `feet_regulation` keeps the same `0.38` reference
3. Cap the flat-terrain `lin_vel_x` command at `(-4.5, 4.2)` (backward `4.5 m/s`, forward `4.2 m/s`) in the final command-range curriculum stage and `terrain_max_command_ranges.flat`, replacing `(-5.0, 5.0)`

## v1.3.1.post1 - 2026-10-05

1. Change `x_lin_scalar: 1.0 -> 0.5`, 
2. Remove `3.0-ea` default ploter in rerun viz.
3. Clip the command after all command sampling.
4. Add a `SolidColor` in flat terrain.
5. Change terrain ratio `stairs_up: 0.20 -> 0.25` and `wheels: 0.10 -> 0.05`

## v1.3.1 - 2026-10-03

1. Change terrain ratio `stairs_up: 0.25 -> 0.20` and `wheels: 0.05 -> 0.10`

## v1.3 - 2026-09-25

1. Replace `limit_vel` in `RLGymVelCommandCfg` with `limit_vel_curriculum`: each stage fully specifies `iter` plus `lin_vel_x`/`lin_vel_y`/`ang_vel_yaw`, the reached stage replaces the active direction-code configuration and rebuilds `limit_vel_comb`, and incomplete stages are rejected. `go2-mixed` keeps `lin_vel_y [-1, 1]` before iteration `25000` and switches to `lin_vel_y [0]` afterwards; `flat` and mixed playback keep a single `iter 0` stage
2. Move the command-range curriculum to `5000/15000/25000/35000/45000` iterations and end it at `lin_vel_x [-5.0, 5.0]`, with the `flat` command cap raised to `(-5.0, 5.0)`
3. Make `base_height_l2` command-conditioned: the target base height scales proportionally with the commanded linear-velocity norm between `BASE_HEIGHT_RANGE (0.20, 0.38)` and `BASE_HEIGHT_COMMAND_SPEED_RANGE (5.0, 2.0)`, i.e. `0.38` at or below `2.0` and `0.20` at or above `5.0`; `feet_regulation` keeps its `0.38` reference

## v1.2.1 - 2026-09-24

1. Add `command_clip_rules` to `RLGymVelCommandCfg`: after sampling, `|lin_vel_x| > 2.0` clips `lin_vel_y` to `+-0.5` and `ang_vel_yaw` to `+-1.0`, then `|ang_vel_yaw| > 2.0` clips `lin_vel_x` to `+-2.0` and `lin_vel_y` to `+-1.0`; rules are evaluated in list order, so an earlier rule wins on the dimensions it clips
2. Add a command-range curriculum stage at iteration `70000` with `lin_vel_x [-4.0, 3.0]`, `lin_vel_y [-1.0, 1.0]`, and `ang_vel_yaw [-3.0, 3.0]`
3. Raise the backward speed caps to `lin_vel_x -3.0 -> -5.0` on `flat` and `-1.5 -> -2.0` on `wheels`

## v1.2 - 2026-09-24

1. Add `flat` max lin_vel_x from `2.0 -> 3.0`
2. Add `wheels` and `small_squares` terrains with ratio `10%, 5%` respectively; keep `num_cols=20` by lowering `stairs_up` to `0.20` and `obstacles` to `0.10`
3. `wheels`: closed-mesh tyre field; `60 cm` diameter tyres (outer radius `0.30 m`) packed on a `0.6 m` pitch grid outside a `1.0 m` radius tyre-free disc around the spawn point, with hole radius `0.05 -> 0.25 m` and per-tyre height `1-3 cm -> 15-17 cm` growing with difficulty
4. `small_squares`: closed-mesh `10 cm` square columns filling the whole `9 m x 9 m` tile (`90 x 90` cells), each height sampled uniformly from `[0, x]` with `x` growing `1 -> 17 cm` with difficulty
5. Register both terrains in `MIXED_TERRAIN_CFG` (mesh functions in `terrain.py`) so the heightfield and mesh backends share the same layout, and add their names to the mixed command ranges

## v1.1 - 2026-09-12

1. Make the closed-mesh terrain backend the default for `go2-mixed` and `go2-mixed-moe-cts`; select the original heightfield backend with `GO2_MIXED_TERRAIN=heightfield`
2. Keep the shared mesh generator, importer, and height raster in `tasks/mixed/terrain_mesh.py`

## v1.0 - 2026-08-26

1. Change the `go2-mixed` and `go2-mixed-moe-cts` terrain mesh settings to `mesh_simplification_stride=1` and `stairs_representation="heightfield"`
2. Set the `undesired_contacts` weight to `-1.0` for PhysX and `-1.0/3.0` for Newton MJWarp
