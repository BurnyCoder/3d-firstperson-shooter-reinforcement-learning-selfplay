# Technical reference

## Shared game and public interfaces

`Simulation.reset(seed,scenario,mask)` resets selected arenas; `step(controls)` advances exactly0.05seconds. `observe()` returns a fresh four-frame observation copy. `snapshot()` supports replay checks; `state_dict()` and `load_state_dict()` preserve simulation/history/random state. Rendering interpolates poses only.

World coordinates are XYZ with Y up. Positive yaw turns from +Z toward +X; positive pitch looks up. The original map occupies30x24metres. Box walls/crates and ground-constrained box-shaped players keep geometry simple. Symmetric spawn baffles block initial cross-team sightlines. The ray/box slab algorithm is adapted from [PBRT](https://www.pbr-book.org/4ed/Geometry_and_Transformations/Bounding_Boxes).

Physical controls have seven fields: forward,strafe,yaw increment,pitch increment,fire,reload,interact. Learned categorical controls decode into the same representation as human input. The actor cannot invoke scripted pathfinding or target locking.

The simulation batches arenas in NumPy. `ArenaEnv` adapts its five learner slots to TorchRL; the opposite team runs a fixed snapshot for each round. The actor sees one player's observations. The critic combines the five learner observations only during optimization. Full trajectories keep shape `[arena,time,player,...]` until GAE; minibatch shuffling preserves the player group.

## Observation schema

Each frame has96float32features; four frames, oldest to newest, produce384features. Positions/distances use documented30m normalization.

|Indices|Information|
|---|---|
|0–15|Own position, sine/cosine yaw and pitch, health, ammunition, reload, alive, team, carrier flag, round clock, interaction progress|
|16–21|Known bomb-relative position, planted flag, bomb clock, information-valid flag|
|22–27|Two public bomb-site relative positions|
|28–35|Eight waist-height horizontal obstacle-distance rays|
|36–95|Ten fixed player slots: relative XYZ, visibility flag, same-team flag, visible teammate health|

Enemy dynamic data are zeroed unless in field of view and unobstructed. Enemy health is never supplied. Public team identity is fixed even when a player is hidden. Unplanted enemy carrier information is hidden; planting announces the bomb position. Dead players retain useful public clock/objective information. Geometry rays include known walls rather than secretly detecting enemies behind cover.

## Combat and objectives

Defaults are explicitly serialized from game configuration:100health,25damage, rifle6rounds/second, magazine30,reserve90,reload2seconds, movement5m/second, round60seconds,bomb25seconds,plant2seconds,defuse3seconds. These are MiniStrike design values, not claims of exact Counter-Strike mechanics.

Shots use actual3D direction and the nearest obstruction; teammates block shots but take no damage. Damage is applied simultaneously to avoid iteration-order duel advantages. Ground movement is collision constrained; players do not collide with one another. A dropped bomb is recovered within1m. Interaction requires uninterrupted eligibility and proximity.

Completed defuse wins over explosion on the same tick. Completed plant takes precedence over pre-plant round timeout. A planted bomb survives attacker elimination. Mutual full elimination before planting is a draw; otherwise elimination and objective rules determine the winner. Ordinary round expiration is terminal game logic, not an artificial training truncation.

Outcome reward is±1per team, draw0. Combat shaping is bounded by0.10for damage and planting adds0.10; opposite teams receive opposite rewards. Shaping changes the training objective and is not claimed policy-invariant. Evaluation uses actual round outcomes, never shaped return.

## Replays, cameras, and checkpoint identity

Finite viewer runs save screenshots and action replays; replaying the same seed/actions headlessly must reproduce the full final snapshot. Free camera, first-person bot camera, and human takeover all read the same world. Taking over or handing back changes only the action provider for one slot; it cannot respawn, heal, refill ammunition, or reset objectives.

Checkpoints include actor/critic/optimizer, environment/observation history, roles, opponent pool and identities, random generators, used budget, source revision, and action/observation schema. Loading uses PyTorch's restricted weights-only format with explicit conversion of NumPy values. Use project-produced checkpoints and retain their configuration/model card.

## Primary implementation sources

- [Ursina FPS example](https://github.com/pokepetter/ursina/blob/master/samples/fps.py): graphics/input patterns; its scripted enemy behavior is not used for learned actors.
- [NumPy broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html): batched world calculations.
- [TorchRL EnvBase](https://docs.pytorch.org/rl/stable/reference/generated/torchrl.envs.EnvBase.html): typed batched environment and partial reset machinery.
- [TorchRL MAPPO tutorial](https://docs.pytorch.org/rl/stable/tutorials/multiagent_ppo.html): library PPO/GAE and centralized training.
- [Python logging](https://docs.python.org/3/library/logging.html): complete timestamped console/file diagnostics.
- [uv PyTorch integration](https://docs.astral.sh/uv/guides/integration/pytorch/): explicit official accelerator wheel source without redirecting other packages.
