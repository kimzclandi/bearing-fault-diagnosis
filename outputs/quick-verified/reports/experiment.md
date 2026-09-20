# Measured experiment report

All values below come from saved predictions. This experiment is a held-out-load evaluation, not an independent-device validation.

## Classification

| model | seed | accuracy | precision_macro | recall_macro | f1_macro |
| --- | --- | --- | --- | --- | --- |
| rf | 42 | 0.8958 | 0.9313 | 0.9132 | 0.9119 |
| rf_without_physics | 42 | 0.8583 | 0.9021 | 0.8819 | 0.8809 |
| cnn | 42 | 0.7875 | 0.7819 | 0.8229 | 0.7810 |

Physics feature ablation, seed 42: macro F1 difference (with minus without) = +0.0310. This measures this fixed protocol only.

## Robustness

| model | perturbation | level | f1_mean | f1_std | f1_change_mean | accuracy_mean |
| --- | --- | --- | --- | --- | --- | --- |
| cnn | missing | 0.0500 | 0.7576 | 0.0071 | -0.0233 | 0.7681 |
| cnn | missing | 0.1000 | 0.6592 | 0.0039 | -0.1218 | 0.6708 |
| cnn | missing | 0.2000 | 0.4906 | 0.0038 | -0.2903 | 0.4903 |
| cnn | noise | 0.0000 | 0.7261 | 0.0069 | -0.0549 | 0.6875 |
| cnn | noise | 10.0000 | 0.8029 | 0.0061 | 0.0219 | 0.7958 |
| cnn | noise | 20.0000 | 0.7845 | 0.0062 | 0.0036 | 0.7903 |
| rf | missing | 0.0500 | 0.9096 | 0.0053 | -0.0023 | 0.8931 |
| rf | missing | 0.1000 | 0.9243 | 0.0019 | 0.0125 | 0.9097 |
| rf | missing | 0.2000 | 0.9351 | 0.0051 | 0.0232 | 0.9222 |
| rf | noise | 0.0000 | 0.8333 | 0.0019 | -0.0786 | 0.7944 |
| rf | noise | 10.0000 | 0.9431 | 0.0055 | 0.0312 | 0.9319 |
| rf | noise | 20.0000 | 0.9191 | 0.0062 | 0.0073 | 0.9042 |
| rf_without_physics | missing | 0.0500 | 0.8994 | 0.0057 | 0.0185 | 0.8792 |
| rf_without_physics | missing | 0.1000 | 0.9301 | 0.0038 | 0.0492 | 0.9153 |
| rf_without_physics | missing | 0.2000 | 0.9387 | 0.0088 | 0.0578 | 0.9264 |
| rf_without_physics | noise | 0.0000 | 0.8031 | 0.0305 | -0.0777 | 0.7569 |
| rf_without_physics | noise | 10.0000 | 0.9164 | 0.0037 | 0.0356 | 0.8986 |
| rf_without_physics | noise | 20.0000 | 0.8835 | 0.0020 | 0.0026 | 0.8611 |

Largest observed mean change: cnn, missing, level 0.2: -0.2903 macro F1. Noise can mask impulses and dropped intervals can remove evidence; these are plausible mechanisms, not established causal explanations.

## Cases and interpretation

- success: 217:77824, true inner, predicted inner, confidence 0.9701.
- failure: 204:69632, true outer, predicted ball, confidence 0.6412.

Cases are chosen deterministically as the highest-confidence correct/incorrect clean CNN prediction. Feature importance favors correlated and high-cardinality features. Input gradients measure local sensitivity, not causation. The marked frequencies are ideal kinematic estimates.

## Health deviation

| label | median_index | above_reference_fraction | class |
| --- | --- | --- | --- |
| 0 | 0.8144 | 0.7500 | normal |
| 1 | 2.5127 | 1.0000 | inner |
| 2 | 2.4637 | 1.0000 | outer |
| 3 | 2.3140 | 1.0000 | ball |

The threshold uses training normal windows only. Above-reference fractions for normal data measure observed false alarms on this condition. Fault and normal distributions may overlap. Within-record fluctuations are not natural degradation histories; no RUL or supervised regression scores are available.

## Limits

Normal sampling rate is a declared 48 kHz convention, not an embedded per-file measurement. Labels are manifest-derived. Shared bearing identity across loads cannot be excluded. Window metrics contain dependent observations and do not establish field reliability. Training and validation loss use different weighting, as recorded in training metadata.
