# Scope and limits

CWRU uses seeded defects and controlled laboratory loads. A held-out load is not an independent machine. Normal and fault acquisition differences may create shortcuts even after resampling. Windows from one record are dependent and effective sample size is closer to the recording count than the window count.

The geometry uses published drive-end ball diameter 0.3126 and pitch diameter 1.537 inches. Nine rolling elements and zero contact angle are configured idealizations consistent with the published ring-frequency multipliers. The standard BSF formula yields about 2.3567 times shaft frequency; the official rolling-element value 4.7135 is approximately 2*BSF, so both are shown without equating them.

Frequency bands use ±max(8 Hz, one FFT bin). The envelope band 2–5 kHz is a fixed candidate resonance band, not a proven optimum. Slip, varying speed, uncertain geometry and transmission paths shift or obscure peaks. Spectral energy near a frequency does not alone establish a fault.

RF impurity importance can favor correlated/high-cardinality predictors. Input-gradient explanations are local, potentially noisy sensitivities. Probabilities are uncalibrated and should not be used as safety probabilities.

Health deviation measures distance from a small training-normal reference. Cross-load shifts can cause false alarms. Record time is not equipment age. No lifespan, degradation ordering between files, or true remaining life is inferred.

Further work: verify acquisition rates; use bearing-identity holdouts when available; validate on naturally evolving faults; tune resonance bands using training/validation only; calibrate probabilities; test independent devices. Such changes require a new protocol and untouched evaluation data.
