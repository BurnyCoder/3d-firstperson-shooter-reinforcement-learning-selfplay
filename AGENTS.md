# Engineering instructions

Work only in this project directory. Use `uv`, Python3.12, the local `.venv`, and project-local `.env`; never publish credentials. Keep the implementation small, reusable, and functional.

- Preserve a single simulation for training, evaluation, replay, and rendering. Presentation cannot implement physics or consume simulator randomness.
- Keep entry points as readable orchestration wrappers over modular phases. Reuse library implementations and remove unused code.
- Search primary documentation before adapting algorithms or APIs. Include source links and handwritten local/global explanations in code comments and docstrings.
- Use test-driven changes for behavioral rules and learning correctness. Exercise the complete user workflow and inspect logs/screenshots after substantive fixes.
- Preserve full timestamped diagnostic logs. Any future LLM integration must record complete prompts/outputs without truncation; the current application uses none.
- Never leak hidden enemy state into actor observations or scripted controls. Keep training-only critic information out of gameplay inference.
- Keep death masks separate from round termination, and preserve full temporal trajectories through advantage estimation.
- Before acting, read `docs/experiment-roadmap.md`, the active `docs/duel-protocol.md`, and the latest `runs/duel-fixed-v1/manifest.json` if present. Stage A has a separate 12-hour absolute budget; future roadmap entries have no budget until separately authorized. Preserve `docs/protocol.md` and the original 5v5 suite as deferred evidence; do not restart it or extend its original deadlines. Mark changes after results as exploratory; never tune on final-test data.
- Before any optimizer-bearing gameplay experiment, pilot or throughput probe, publish implementation, roadmap, protocol and resolved configuration. Use the launch preflight. Preserve original registrations and append dated amendments. Check `docs/sources.md` and `docs/evidence-index.md` for claim-level provenance; label numerical defaults as engineering decisions.
- Push implementation and protocol before training. Use meaningful commits and feature PRs, attach PRs to the Codex chat, and merge after checks. One substantive review suffices unless serious findings remain.
- Keep README as the concise installation/use overview, technical details in `docs`, and measured evidence in `reports`. Do not duplicate whole explanations.
- Reports must distinguish measured observations, assumptions, and limitations; include failed experiments. Name Libor Burian as paper author. Publish model artifacts and model cards in a Hugging Face collection.
- Save architecture, per-run loss/reward/activity graphs and evaluation comparisons with raw inputs, hashes and reproducible plotting commands. Difference cumulative counters before plotting interval rates. Keep missing metrics visible and match uncertainty separate from independent-model uncertainty; loss/reward curves alone do not establish learning quality.
- During registered duration/entropy diagnostics, also read `docs/amendments/2026-10-23-duration-entropy-diagnostic.md`. Preserve the source revision through both serial workers and their automatic validation; completed mode controls and original pilots must not be repeated. Milestone checkpoints are diagnostic evidence, not deployment recommendations.
- Verify installation, headless training, checkpoint resume, evaluation, both spectator camera modes, live takeover/hand-back, and first-person human play before claiming completion.
