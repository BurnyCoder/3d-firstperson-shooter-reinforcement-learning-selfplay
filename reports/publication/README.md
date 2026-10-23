# Public checkpoint archive

The [MiniStrike staged shooter self-play collection](https://huggingface.co/collections/BurnyCoder/ministrike-staged-shooter-self-play-6ada65970d163e293d48f3ad) retains the original collection ID `6ada65970d163e293d48f3ad`; renaming its display title changed its readable slug. It contains **17 repositories**: two previous CPU/CUDA smoke artifacts plus 15 newly published model archives.

The new archives contain three confirmation pairs, three reward-pilot pairs, three duration/entropy/representation diagnostic pairs, and six deferred 5v5 pairs. Each includes its config, canonical metrics and model card; completed duel evaluations and release context are included where available. Diagnostic cards prohibit deployment; deferred cards identify partial budgets and absent final evaluation. Confirmation cards identify the failed moving-script gate and the limited fixed-spawn domain.

[Verification](verification.json) checks all **150 uploaded files** across the 15 new repositories against local bytes: SHA256 for LFS checkpoints and Git blob SHA1 for ordinary files, with local SHA256 retained for all files. All repositories were public and members of the collection. This checks byte identity and access metadata, not a claim that asynchronous Hub security scans have completed. [Raw remote listings and upload receipts](data/), [exact published cards](cards/) and timestamped full CLI logs are retained.

The [selected experimental model](https://huggingface.co/BurnyCoder/ministrike-duel-fixed-v1-seed5101) was selected by validation before final testing. All [5102](https://huggingface.co/BurnyCoder/ministrike-duel-fixed-v1-seed5102) and [5103](https://huggingface.co/BurnyCoder/ministrike-duel-fixed-v1-seed5103) outcomes remain public; selecting one for inspection does not remove their failures.

```powershell
uv run hf download BurnyCoder/ministrike-duel-fixed-v1-seed5101 --local-dir artifacts/models/duel-seed5101
uv run csrl watch --experiment duel-fixed-v1 --checkpoint artifacts/models/duel-seed5101/model.pt --first-person
uv run csrl play --experiment duel-fixed-v1 --checkpoint artifacts/models/duel-seed5101/model.pt
```

Publication used the [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/en/guides/cli) through a strict filename allowlist. Credentials, `.env`, environments and caches were excluded. Restricted checkpoint loading and explicit match-interface checks are implemented in the source; the model card provides training-source and checkpoint identities. The [confirmation report](../duel-fixed-v1/confirmation/README.md) and [all-experiment summary](../summary.md) explain the evidence and limitations.
