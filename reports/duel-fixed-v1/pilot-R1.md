# A-PILOT-R1: stronger damage reward improves random-opponent combat, but fails selection

Author: Libor Burian. Exploratory seed 5201; the [registered R1 configuration](../../configs/duel-pilot-R1.json) changes R0's effective-damage coefficient from 0.25 to 0.50. Architecture, self-play, optimizer settings and arena-step budget are unchanged. The [active protocol](../../docs/duel-protocol.md) defines every numerical engineering choice and selection criterion.

## Identity and completeness

Source: [`367c10828b4eb08a22d5997e47e9906b19efe512`](https://github.com/BurnyCoder/counter-strike-reinforcement-learning-selfplay/tree/367c10828b4eb08a22d5997e47e9906b19efe512). Both actor and critic initial tensor hashes match R0 exactly; full checkpoint files differ because metadata identifies distinct rewards/runs. [Assessment and checkpoint hashes](data/pilot-R1/assessment.json), [raw-file hashes](data/pilot-R1/evidence.json), [resolved config](data/pilot-R1/config.json).

R1 completed **1,245,184 arena steps and 152 updates**, matching R0. Trainer elapsed time was **574.844 seconds**; full launch/supervision was **587.359 seconds**. Self-play produced 4,481 rounds, 1,752 learner wins and 2,183 draws. These outcomes involve evolving opponents and cannot replace fixed benchmark evaluation. [Result](data/pilot-R1/result.json), [all updates](data/pilot-R1/metrics.jsonl), [group and telemetry](data/pilot-R1/group.json).

Observed host-memory headroom fell to **14.32%**, below the 20% selection threshold measured during the earlier hardware probes. The single-worker run nevertheless finished without an OOM or logged training error. This is a change in whole-host memory availability, not evidence that the original calibration measured 20% headroom throughout later work. No faster configuration or repeat calibration was substituted; no cause for other host memory use is asserted.

## Primary validation and decision

Validation root 3,000,000 and both spawn ends match R0. Scores are win 1, draw 0.5, loss 0. Random opponents use 128 paired seeds; each deterministic control has one unique trajectory per end. Baselines and ends receive equal weight. [Initial raw rounds](data/pilot-R1/initial_validation.json), [final raw rounds](data/pilot-R1/final_validation.json).

| Opponent | Initial score | Final score | Final wins / draws / losses | Matches |
|---|---:|---:|---:|---:|
| Random actions | 0.500000 | 0.689453 | 97 / 159 / 0 | 256 |
| Corresponding initial policy | 0.500000 | 0.500000 | 0 / 2 / 0 | 2 |
| Moving quantized scripted benchmark | 0.000000 | 0.000000 | 0 / 0 / 2 | 2 |
| Equal-opponent aggregate | 0.333333 | 0.396484 | — | 260 |

Improvement **0.0631510** is below the registered **0.10** gate. The paired match-bootstrap 95% interval [0.0553385, 0.0703125] is conditional on one initial/final model pair, not uncertainty across independent training runs. Final elimination fraction is **99/260 = 38.08%**. No final-test records were accessed and no confirmation run is authorized by this result. Reproduce the assessment and raw evidence copy with `uv run python scripts/assess_duel_pilot.py R1`.

R1's random-opponent wins exceed R0's 19/256 at the same training budget, but performance is strongly asymmetric: random-opponent score is **0.53125 from end 0 and 0.84765625 from end 1**. This finding motivates later symmetry diagnostics if needed; it does not authorize changing the frozen spawns or observations mid-experiment. Both deterministic matches against the initial policy still timed out.

## Diagnostic interpretation

Against random opponents, damage per round increased from 9.180 to **63.184**, hits per shot from 0.01421 to **0.10250**, and mean duration fell from 20 to **13.3875 seconds**. Final reload occupancy was 0.000438 and aim saturation zero. Against the moving benchmark, final mean damage remained 25 and mean duration was 0.8 seconds. Counter definitions and pooled denominators are reproducible through the assessment script; raw rows remain available above.

The final update recorded finite policy loss −0.003950, critic loss 0.015074, entropy 5.462410, approximate KL 0.010618, clip fraction 0.137939 and gradient norm 0.708728. These diagnostics and existing passing correctness tests identify no new implementation defect in this run; they do not prove the absence of all defects. A single matched exploratory seed does not establish a reliable reward advantage across retraining. Losses, training wins and shooting counters do not override failed benchmark criteria.

## Secondary stochastic behavior

The separately registered stochastic evaluation is substantially stronger than deterministic deployment. Both endpoints use 128 paired seeds per opponent and both spawn ends, 768 matches each. [Initial stochastic rounds](data/pilot-R1/initial_validation_stochastic.json), [final stochastic rounds](data/pilot-R1/final_validation_stochastic.json).

| Opponent | Initial stochastic score | Final stochastic score | Final wins / draws / losses |
|---|---:|---:|---:|
| Random actions | 0.498047 | 0.892578 | 201 / 55 / 0 |
| Corresponding initial policy | 0.500000 | 0.898438 | 204 / 52 / 0 |
| Moving quantized scripted benchmark | 0.000000 | 0.000000 | 0 / 0 / 256 |
| Equal-opponent aggregate | 0.332682 | 0.597005 | — |

This is evidence of useful improvement against the two weak stochastic controls for this trained checkpoint: aggregate improvement is **0.264323** and final elimination fraction **86.07%**. It does not establish competence against the moving benchmark, which still wins every match. Stochastic and deterministic evaluations also differ in the initial opponent's action selection, so their difference is **not** an isolated causal effect of learner stochasticity for that baseline. Against random/scripted controls, opponent behavior definitions remain the same between modes.

The primary deterministic gate still fails. Replacing it with the stronger secondary result after observing both would violate the registration. The gap motivates a prospective action-selection diagnostic if R2 also fails, rather than a claim that no learning occurred. R1 remains exploratory and does not become a deployment recommendation.

Proceed to the already registered **R2** candidate: retain alpha 0.50 and add the bounded, terminal-correct aim potential difference. The hypothesis is that denser aim feedback helps this fixed-spawn task at the same step budget; the observed counter changes do not prove aiming is the sole remaining bottleneck. R2 is not guaranteed to converge. Its formula follows [Ng, Harada and Russell (1999)](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf); that MDP result does not establish convergence for partial-observation self-play. The exact prospective decision is in the [R1 diagnosis amendment](../../docs/amendments/2026-10-23-R1-diagnosis.md). Independent-model uncertainty and limited-seed interpretation follow the [shared bibliography](../../docs/sources.md).
