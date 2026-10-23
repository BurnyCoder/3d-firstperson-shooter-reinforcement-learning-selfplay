# Releasing measured checkpoints

The [public collection](https://huggingface.co/collections/BurnyCoder/ministrike-staged-shooter-self-play-6ada65970d163e293d48f3ad) groups confirmation, pilots, diagnostics, deferred 5v5 and verification runs. Its collection ID is unchanged. Cards disclose exact frame count, completion status, source revision and measured score. Confirmation is experimental because the moving-script criterion failed; diagnostics cannot deploy. [All publication receipts and byte checks](../reports/publication/README.md).

After authenticating with `uv run hf auth login`, publish the direct child run directories of a suite:

```powershell
uv run python -m ministrike.experiments.publishing --input runs/duel-fixed-v1/confirmation --collection BurnyCoder/ministrike-staged-shooter-self-play-6ada65970d163e293d48f3ad
# Explicit archival publication only; this does not complete or restart deferred training:
uv run python -m ministrike.experiments.publishing --input runs/suite --include-deferred --collection BurnyCoder/ministrike-staged-shooter-self-play-6ada65970d163e293d48f3ad
```

The command stages only `model.pt`, `initial.pt`, configuration, result, evaluation, metric and release-context files plus a generated card. Deferred archives get a distinct derived `archive-status.json`, never a fabricated completed trainer result. It reapplies this allowlist at upload. `.env`, credentials, caches and unrelated files are excluded. Pilot/diagnostic IDs receive distinct repository names, preventing seed5201 variants from overwriting each other. `publication.json` retains receipts; remote commit/file metadata supports byte verification. The authenticated [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/en/guides/cli) streams complete output to timestamped logs and adds each model to the existing collection.

Download one model with `uv run hf download REPOSITORY_ID --local-dir artifacts/models/NAME`. Use its `model.pt` with either `csrl watch --first-person --checkpoint ...` or `csrl play --checkpoint ...`. The initial checkpoint remains available for direct before/after inspection.
