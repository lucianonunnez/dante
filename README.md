# Attention Lens

> Turns a video into an **engagement curve**: it reads TRIBE's per-second
> prediction of cortical activation and tells you which seconds *land* and which
> are **dead air** — with the cut points and the strongest hook called out.

Built on top of **TRIBE** — a model that predicts brain activity, second by
second, from what a person is watching and hearing (see the original Colab
notebook under [`notebooks/`](notebooks/)). Attention Lens is the layer that
makes that signal *useful to a human editor*: an engagement score, ranked hooks,
dead-air segments to trim, and a plain recommendation list. Runs **fully
offline** with a synthetic clip, so the whole thing works without a GPU or model
weights.

```bash
pip install -e .
attention-lens demo                 # analyse a synthetic clip, print the report
attention-lens analyze run.npz      # analyse a real TRIBE export
attention-lens demo --json out.json # machine-readable output
pytest -q                            # 6 tests, all offline
```

Example output ([full report](examples/sample_report.md)):

```
- Duration: 3:00
- Overall engagement: 42/100

▃▃▃▄▄▄▄▇██▇▄▄▄▄▄▄▄▃▄▂▁▁▁▁▁▁▁▂▂▂▂▂▂▂▂▂▂▃▆▇▇▇▇▅▄▄▄▄▄▂▁▁▁▂▄▄▄▄▄

Hooks     0:21–0:33 (96) · 1:57–2:13 (93)
Dead air  1:01–1:46 (45s, 15) · 2:31–2:45 (14s, 15)
→ Lead with 0:21–0:33 — your strongest hook. Trim 1:01–1:46 — dead air.
```

## Why this exists

TRIBE outputs a rich, per-second neural signal — but a raw fMRI stream is not
something a video editor, a marketer or a course designer can act on. Attention
Lens answers the operational question instead: **where does this video win or
lose attention, and what should I do about it?** Cut the dead air, open with the
hook.

## How it works

```
video ──▶ TRIBE ──▶ per-second cortical activation  (the .npz export)
                          │
                          ▼  smooth        rolling mean, de-jitter
                          ▼  normalise     robust 0..100 engagement index (p5–p95)
                          ▼  segment       hooks (sustained high) · dead air (sustained low)
                          ▼  align         transcript words → each segment
                          ▼
                    EngagementReport  (score · hooks · dead air · recommendations)
```

Every step is **transparent arithmetic on top of the model** — no opaque
post-processing. An editor can see exactly why a second scored low (it's the
normalised activation), which is the whole point: the recommendation has to be
defensible.

| Piece | Where it lives | Responsibility |
|---|---|---|
| Data contract | `schema.py` | `ActivationTimeline`, `Segment`, `EngagementReport` — one shared shape. |
| Model seam | `source.py` | Load a real TRIBE `.npz` **or** a synthetic clip, same contract out. |
| Metrics | `engagement.py` | Smooth → normalise → segment into hooks / dead air. |
| Reporting | `report.py` | Markdown report + ASCII engagement sparkline. |
| CLI | `cli.py` | `demo` / `analyze`, text or JSON. |
| Research origin | `notebooks/` | The original TRIBE Colab notebook this is built on. |

## Honest scope

- **Real:** the whole engagement pipeline — smoothing, normalisation,
  segmentation, transcript alignment, scoring, reporting and the CLI — plus the
  `.npz` loader for genuine TRIBE exports.
- **Mocked (behind a seam):** the neural signal itself. Running TRIBE needs a
  GPU and model weights; `source.from_mock()` fabricates a plausible, seeded
  timeline so the package is testable and demoable anywhere. The seam lives in
  `source.py` and nowhere else — plug the real export in and nothing downstream
  changes.
- **Deliberately not claimed:** this is an engagement **index**, not a
  neuroscience ground truth. It ranks and explains attention within one video;
  it is not a validated cross-subject measurement.

## The `.npz` contract

`attention-lens analyze` expects a TRIBE export with a `fmri` array
(`seconds × voxels`), optional `fmri_left` / `fmri_right`, and an optional
`words` / `attention_json` array of `[text, start, end]` triples. Missing
hemispheres or transcript degrade gracefully.

## Project layout

```
attention-lens/
├── src/attention_lens/
│   ├── schema.py         # typed contracts
│   ├── source.py         # TRIBE .npz loader + synthetic seam
│   ├── synth.py          # seeded synthetic activation generator
│   ├── engagement.py     # smooth · normalise · segment
│   ├── report.py         # markdown + sparkline
│   └── cli.py            # demo / analyze
├── notebooks/            # original TRIBE Colab notebook (research origin)
├── examples/             # a rendered sample report
├── tests/                # offline, deterministic
└── pyproject.toml · requirements.txt · LICENSE
```

## License

MIT — see [LICENSE](LICENSE).
