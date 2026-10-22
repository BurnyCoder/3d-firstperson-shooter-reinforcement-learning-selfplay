# Releasing measured checkpoints

The [public collection](https://huggingface.co/collections/BurnyCoder/ministrike-self-play-tactical-bots-6ada65970d163e293d48f3ad) groups both experimental conditions and verification runs. Model cards disclose the exact frame count, completion status, source revision, and held-out score. A smoke checkpoint demonstrates an executable pipeline; it does not establish useful tactical behavior.

After authenticating with `uv run hf auth login`, publish the direct child run directories of a suite:

```powershell
uv run python -m ministrike.experiments.publishing --input runs/suite --collection BurnyCoder/ministrike-self-play-tactical-bots-6ada65970d163e293d48f3ad
```

The command stages only `model.pt`, `initial.pt`, configuration, result, evaluation, and metric files plus a generated model card. It reapplies this allowlist at the upload boundary. `.env`, credentials, caches, unrelated files, and upload staging are excluded from scientific report discovery. Existing repository names make retries idempotent; `publication.json` records completed receipts. The command uses the authenticated [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/en/guides/cli), streams its output to timestamped logs, and adds each model to the collection.

Download one model with `uv run hf download REPOSITORY_ID --local-dir artifacts/models/NAME`. Use its `model.pt` with either `csrl watch --first-person --checkpoint ...` or `csrl play --checkpoint ...`. The initial checkpoint remains available for direct before/after inspection.
