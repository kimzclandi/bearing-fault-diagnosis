# Fixed experiment protocol

Protocol v1. No hyperparameter search against test outcomes.

- CWRU 48 kHz drive-end subset: three fault sizes, inner/ball/outer at 6:00, four loads; four normal files. Label mapping: 0 normal, 1 inner, 2 outer, 3 ball.
- Whole recordings: train loads 0/1 HP; validation 2 HP; test 3 HP. Split precedes all window extraction. Same bearing may occur across loads.
- Native normal-file rate: 48 kHz convention, not independently encoded in MAT metadata; this uncertainty must accompany results.
- Polyphase anti-alias resampling to 12 kHz, 4096 samples, zero overlap. Subtract per-window DC. Train-only scalar normalization; no per-window amplitude normalization.
- Quick: at most 24 evenly indexed windows/file; 8 epochs, patience 3, 100 trees, seed 42.
- Full: all nonoverlapping windows; 50 epochs, patience 8, 300 trees, seeds 42/43/44. Both configurations frozen before first evaluation.
- CNN selection: validation macro F1, ties resolved by unweighted validation cross entropy. Training uses inverse-frequency weighted cross entropy; curves label their differing objectives.
- RF: balanced class weights, leaf size 2. Physics ablation changes only feature columns. All models share sample lists.
- Explanations: seed 42; highest-confidence correct and incorrect clean CNN predictions; report absence if no such case exists.
- Robustness: SNR 20/10/0 dB and missing fraction .05/.10/.20, perturbation seeds 101/102/103. Report every training seed. Identical perturbations for each model; no retraining.
- Health reference: training normal median/MAD, log-compressed average absolute deviation. Causal EWMA alpha .2; training-normal q99 threshold. No natural degradation labels, no RUL scores.
- Report clean and perturbed predictions, per-class and per-file metrics. Standard deviations across seeds are not population confidence intervals.
- Preserve run directories. Software corrections are documented and affected runs repeated under a new identifier. Test results do not direct architecture or hyperparameter changes.

Implementation clarification: forests fit with configured worker count and switch to one inference worker before validation and serialization. This fixes accumulation order at near ties. Discrete outputs must match exactly in reproducibility checks; probability tolerance is atol 1e-7 / rtol 1e-6, suitable for stored float32 CNN probabilities, and is not used to waive class differences.
