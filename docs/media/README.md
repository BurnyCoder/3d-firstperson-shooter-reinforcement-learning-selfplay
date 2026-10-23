# Gameplay recording

[duel-gameplay.mp4](duel-gameplay.mp4) is an uninterrupted 24-second recording of the native MiniStrike window, captured on 23 October 2026 at 12:29:52 UTC. It shows first-person bot watching in `duel-fixed-v1`; both players use the same frozen confirmation seed 5101 checkpoint. The spectator automatically follows a living bot when the followed player dies. There was no human takeover, policy update, scripted opponent, time compression or montage.

The checkpoint was selected by validation before final testing, as recorded in the [confirmation results](../../reports/duel-fixed-v1/confirmation/README.md). It remains **experimental** because the moving-script acceptance criterion failed. A short fixed-spawn self-play clip is not an evaluation of generalization or human-opponent skill. This presentation recording is separate from the completed research budget; it introduces no training or new benchmark results.

## Provenance and reproduction

- Source: [`d2bc7e677e208012c9b4368b228d3c9194bd1669`](https://github.com/BurnyCoder/3d-firstperson-shooter-reinforcement-learning-selfplay/tree/d2bc7e677e208012c9b4368b228d3c9194bd1669).
- Model: [BurnyCoder/ministrike-duel-fixed-v1-seed5101](https://huggingface.co/BurnyCoder/ministrike-duel-fixed-v1-seed5101), `model.pt` SHA256 `08555d745249be9e49bb4b320f04737c5c5730bb53c39473d5fb4cf16a481ecb`.
- Video SHA256: `82a6dff950a7dd4c543a5b6b00cc8c5f467fc7364b068cbf3481c2c1f6e893e5`.
- Format: H.264, 1280 × 720, 24 seconds, no audio; 1,065,525 bytes. Capture targeted 30 frames/s; the file contains 716 frames (29.833 frames/s on average). Resolution and encoding settings are presentation choices, not learning parameters.
- Match seed: `111`. Recording starts during an already running session; reproducing the commands does not reproduce the exact wall-clock capture boundary.

After the [README setup](../../README.md#methodology-and-quick-start), download and launch the checkpoint:

```powershell
uv run hf download BurnyCoder/ministrike-duel-fixed-v1-seed5101 --local-dir artifacts/models/duel-seed5101
uv run csrl watch --experiment duel-fixed-v1 --checkpoint artifacts/models/duel-seed5101/model.pt --first-person --seed 111 --seconds 90 --screenshot artifacts/readme-gameplay-video/final.png
```

In another terminal, with [FFmpeg](https://ffmpeg.org/download.html) installed and the game visible, record its client area only. Exit the viewer with F10 after capture. FFmpeg's [Windows `gdigrab` device](https://ffmpeg.org/ffmpeg-devices.html#gdigrab) supports window-title capture; this avoids recording unrelated desktop content.

```powershell
ffmpeg -hide_banner -f gdigrab -framerate 30 -draw_mouse 0 -i "title=MiniStrike | Self-play Lab" -t 24 -vf "scale=1280:720" -c:v libx264 -preset fast -crf 23 -threads 2 -pix_fmt yuv420p -movflags +faststart -an artifacts/readme-gameplay-video/duel-gameplay.mp4
```

The encoder was FFmpeg `N-124716-g054dffd133-20260531`. H.264 follows [GitHub's compatibility recommendation](https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/attaching-files). The README player uses a public GitHub video attachment; the identical MP4 remains in this directory for cloning and download. [GitHub CLI's attachment support](https://github.blog/changelog/2026-09-01-github-cli-media-in-issues-pull-requests-and-comments/) provides the upload without an additional hosting service.

## Verification

The full video decoded without FFmpeg errors. Six evenly spaced frames were visually inspected for gameplay and capture boundaries. GitHub's Markdown renderer produced a video player with controls, and an unauthenticated download of the public attachment matched the MP4 hash above. The [recorded session](evidence/final.replay.json), its [final screenshot](evidence/final.png) and [parity receipt](evidence/parity.json) preserve the session containing the clip: all 348 recorded simulation ticks across 20 started rounds reproduced the final canonical state exactly through the existing headless replay verifier. The session is longer than the 24-second video excerpt. This checks rendering/simulation consistency, not learning quality.

```powershell
uv run python scripts/verify_native_replays.py --checkpoint artifacts/models/duel-seed5101/model.pt --output artifacts/readme-video-reverification docs/media/evidence/final.replay.json
ffprobe -v error -count_frames -show_entries stream=nb_read_frames,codec_name,avg_frame_rate:format=duration,size -of json docs/media/duel-gameplay.mp4
ffmpeg -v error -i docs/media/duel-gameplay.mp4 -f null -
```
