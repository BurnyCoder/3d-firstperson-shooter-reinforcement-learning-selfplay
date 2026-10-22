# Engineering instructions

Work only in this project directory. Use `uv`, Python3.12, the local `.venv`, and project-local `.env`; never publish credentials. Keep the implementation small, reusable, and functional.

- Preserve a single simulation for training, evaluation, replay, and rendering. Presentation cannot implement physics or consume simulator randomness.
- Keep entry points as readable orchestration wrappers over modular phases. Reuse library implementations and remove unused code.
- Search primary documentation before adapting algorithms or APIs. Include source links and handwritten local/global explanations in code comments and docstrings.
- Use test-driven changes for behavioral rules and learning correctness. Exercise the complete user workflow and inspect logs/screenshots after substantive fixes.
- Preserve full timestamped diagnostic logs. Any future LLM integration must record complete prompts/outputs without truncation; the current application uses none.
- Never leak hidden enemy state into actor observations or scripted controls. Keep training-only critic information out of gameplay inference.
- Keep death masks separate from round termination, and preserve full temporal trajectories through advantage estimation.
- Follow the frozen experiment protocol in `docs/protocol.md`. Mark changes made after results as exploratory. Never select models or hyperparameters using final-test data.
- Push implementation and protocol before training. Use meaningful commits and feature PRs, attach PRs to the Codex chat, and merge after checks. One substantive review suffices unless serious findings remain.
- Keep README as the concise installation/use overview, technical details in `docs`, and measured evidence in `reports`. Do not duplicate whole explanations.
- Reports must distinguish measured observations, assumptions, and limitations; include failed experiments. Name Libor Burian as paper author. Publish model artifacts and model cards in a Hugging Face collection.
- Verify installation, headless training, checkpoint resume, evaluation, both spectator camera modes, live takeover/hand-back, and first-person human play before claiming completion.
