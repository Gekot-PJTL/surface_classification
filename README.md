# surface_classification

Just run the python files, they will automatically upload the corresponding `.cfg` in the same folder.

## Range-profile viewers
- [surface_classifier.py](surface_classifier.py) — plots the range profile.
- [surface_classifier_v2.py](surface_classifier_v2.py) — same thing, plus a few extra statistic parameters. (not very useful anyways)

## Grass vs. not-grass classifiers
All three follow the same flow: point the radar at grass, press `c` to record a reference, then move around and the dashboard shows GRASS / NOT GRASS. `r` re-records, `q` quits.

- [grass_no_grass.py](grass_no_grass.py) — cosine similarity against the reference. Simplest version.
- [grass_no_grass_v2.py](grass_no_grass_v2.py) — cosine similarity + bin shifting. Slides the reference window by up to `HEIGHT_TOLERANCE_CM` to find the best match.
- [grass_no_grass_v3.py](grass_no_grass_v3.py) — L1 difference (sum of absolute per-bin dB differences) + bin shifting.

## ML classifiers
Under [ML/](ML/). Workflow is: collect labeled data → train a model → run live inference.

1. **Collect data** — run [ML/grass_logger.py](ML/grass_logger.py). Press `g` to log grass frames, `n` to log not-grass frames, `p` to pause, `q` to quit. Rows get appended to a CSV (`label, bin0, bin1, ...`) across sessions.
2. **Train** — pick one:
   - [ML/cnn/cnn_train.py](ML/cnn/cnn_train.py) — supervised 1D CNN on grass + not-grass data. Outputs `grass_model.pth`.
   - [ML/autoencoder/autoencoder_train.py](ML/autoencoder/autoencoder_train.py) — trains only on grass and flags high reconstruction error as not-grass. Useful if you don't have much not-grass data. Outputs `autoencoder_model.pth`.
3. **Classify live** — run the matching inference script:
   - [ML/cnn/cnn_classify.py](ML/cnn/cnn_classify.py)
   - [ML/autoencoder/autoencoder_classify.py](ML/autoencoder/autoencoder_classify.py)

Both training scripts accept `--csv`, `--epochs`, and `--lr`; both classify scripts accept `--model` and `--threshold`.
