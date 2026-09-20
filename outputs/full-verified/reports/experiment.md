# Measured experiment report

All values below come from saved predictions. This experiment is a held-out-load evaluation, not an independent-device validation.

## Classification

| model | seed | accuracy | precision_macro | recall_macro | f1_macro |
| --- | --- | --- | --- | --- | --- |
| rf | 42 | 0.9276 | 0.9500 | 0.9397 | 0.9393 |
| rf_without_physics | 42 | 0.9655 | 0.9732 | 0.9713 | 0.9712 |
| cnn | 42 | 0.8069 | 0.8985 | 0.8391 | 0.8433 |
| rf | 43 | 0.8966 | 0.9311 | 0.9138 | 0.9123 |
| rf_without_physics | 43 | 0.9448 | 0.9555 | 0.9540 | 0.9542 |
| cnn | 43 | 0.8103 | 0.8959 | 0.8420 | 0.8427 |
| rf | 44 | 0.9448 | 0.9595 | 0.9540 | 0.9540 |
| rf_without_physics | 44 | 0.9172 | 0.9360 | 0.9310 | 0.9311 |
| cnn | 44 | 0.8034 | 0.9010 | 0.8362 | 0.8404 |

Physics feature ablation, seed 42: macro F1 difference (with minus without) = -0.0319. This measures this fixed protocol only.

Physics feature ablation, seed 43: macro F1 difference (with minus without) = -0.0418. This measures this fixed protocol only.

Physics feature ablation, seed 44: macro F1 difference (with minus without) = +0.0229. This measures this fixed protocol only.

## Robustness

| model | perturbation | level | f1_mean | f1_std | f1_change_mean | accuracy_mean |
| --- | --- | --- | --- | --- | --- | --- |
| cnn | missing | 0.0500 | 0.8487 | 0.0057 | 0.0066 | 0.8149 |
| cnn | missing | 0.1000 | 0.8646 | 0.0102 | 0.0225 | 0.8349 |
| cnn | missing | 0.2000 | 0.8999 | 0.0094 | 0.0578 | 0.8805 |
| cnn | noise | 0.0000 | 0.7076 | 0.1010 | -0.1346 | 0.7046 |
| cnn | noise | 10.0000 | 0.8064 | 0.0248 | -0.0357 | 0.7751 |
| cnn | noise | 20.0000 | 0.8396 | 0.0063 | -0.0025 | 0.8038 |
| rf | missing | 0.0500 | 0.9402 | 0.0158 | 0.0050 | 0.9287 |
| rf | missing | 0.1000 | 0.9385 | 0.0104 | 0.0033 | 0.9264 |
| rf | missing | 0.2000 | 0.9433 | 0.0096 | 0.0081 | 0.9318 |
| rf | noise | 0.0000 | 0.7906 | 0.0103 | -0.1446 | 0.7448 |
| rf | noise | 10.0000 | 0.9542 | 0.0121 | 0.0190 | 0.9452 |
| rf | noise | 20.0000 | 0.9365 | 0.0193 | 0.0013 | 0.9245 |
| rf_without_physics | missing | 0.0500 | 0.9620 | 0.0045 | 0.0099 | 0.9548 |
| rf_without_physics | missing | 0.1000 | 0.9572 | 0.0044 | 0.0051 | 0.9490 |
| rf_without_physics | missing | 0.2000 | 0.9368 | 0.0059 | -0.0154 | 0.9245 |
| rf_without_physics | noise | 0.0000 | 0.8199 | 0.0063 | -0.1322 | 0.7789 |
| rf_without_physics | noise | 10.0000 | 0.9398 | 0.0100 | -0.0123 | 0.9276 |
| rf_without_physics | noise | 20.0000 | 0.9573 | 0.0093 | 0.0051 | 0.9487 |

Largest observed mean change: rf, noise, level 0.0: -0.1446 macro F1. Noise can mask impulses and dropped intervals can remove evidence; these are plausible mechanisms, not established causal explanations.

## Cases and interpretation

- success: 217:45056, true inner, predicted inner, confidence 1.0000.
- failure: 204:20480, true outer, predicted ball, confidence 0.9543.

Cases are chosen deterministically as the highest-confidence correct/incorrect clean CNN prediction. Feature importance favors correlated and high-cardinality features. Input gradients measure local sensitivity, not causation. The marked frequencies are ideal kinematic estimates.

## Health deviation

| label | median_index | above_reference_fraction | class |
| --- | --- | --- | --- |
| 0 | 0.8029 | 0.7931 | normal |
| 1 | 2.4914 | 1.0000 | inner |
| 2 | 2.4769 | 1.0000 | outer |
| 3 | 2.3061 | 1.0000 | ball |

The threshold uses training normal windows only. Above-reference fractions for normal data measure observed false alarms on this condition. Fault and normal distributions may overlap. Within-record fluctuations are not natural degradation histories; no RUL or supervised regression scores are available.

## Limits

Normal sampling rate is a declared 48 kHz convention, not an embedded per-file measurement. Labels are manifest-derived. Shared bearing identity across loads cannot be excluded. Window metrics contain dependent observations and do not establish field reliability. Training and validation loss use different weighting, as recorded in training metadata.
