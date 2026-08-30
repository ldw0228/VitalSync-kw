# Commercial-goal audit

Candidate: `validation-locked structured two-SNN ensemble`

This is a retrospective engineering audit, not a commercial or medical claim.

| Metric | Achieved | Target | Result |
|---|---:|---:|:---:|
| Overall MAE | 1.291 bpm | ≤ 1.0 bpm | FAIL |
| Identity-macro MAE | 1.220 bpm | ≤ 1.0 bpm | FAIL |
| RMSE | 2.410 bpm | ≤ 1.8 bpm | FAIL |
| Within ±2 bpm | 80.791% | ≥ 90% | FAIL |
| Error >5 bpm | 6.231% | ≤ 3% | FAIL |
| 25–35 bpm MAE | 4.216 bpm | ≤ 2.0 bpm | FAIL |

## Required evidence gates

| Evidence | Result | Status |
|---|:---:|---|
| Complete radar-mask robustness | PASS | complete |
| Non-overlap evaluation | PASS | complete |
| Validation-locked interval calibration | FAIL | ranking_only_not_interval_calibrated |
| Deployment-faithful E2E p95 within stride | FAIL | timing_complete_not_feature_bit_exact |

## Dependence-aware view

The greedy non-overlap subset has n=444, MAE=1.570 bpm and RMSE=2.860 bpm.

## Conclusion

The declared full-coverage commercial goal was not met.
