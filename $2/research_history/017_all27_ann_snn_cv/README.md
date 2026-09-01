# 017 — All-27 ANN/SNN cross-validation

The compact dataset had 1,082 30-s windows (947 RR targets). Five subject-level test folds, three validation subjects per fold, seeds 42/314/2718, max 120 epochs and early stopping were used. Subject-macro RR MAE: ANN direct 1.528, SNN direct 1.508, signed-rate 1.449, delta-event 1.426. The DSP baseline was better at 0.992. The whole comparison took about 858 s because the SNN had only 2,690 parameters. It is preserved in [`../../legacy_compact_model`](../../legacy_compact_model/README.md) as a baseline, not the final model.
