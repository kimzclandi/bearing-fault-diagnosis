# Execution notes

Before any model evaluation, actual downloads and MAT inspection revealed:

1. Whole-file transfer of source 123 repeatedly terminated early. The downloader was changed to validated 1 MiB byte ranges with resumable partial files and atomic completion. A mock-transfer regression test covers resumption.
2. The official endpoint returned HTTP 503 during metadata requests. HEAD and range requests now use bounded retries and exponential backoff. Initial failed download run directories are retained.
3. Source 99 embeds source 98 channels, and source 175 embeds a source 217 channel. Channel selection uses exact source prefixes, including RPM selection, before allowing a unique-channel fallback. This prevents unrelated embedded recordings from entering the selected split.
4. Source 174 has a single mismatched channel name X173 and a shorter signal. The anomaly is retained and documented; the official table label is used without padding or synthesizing samples.

No test metrics were used to make these corrections. The fixed feature definitions, split, models and training settings remained unchanged.

The first quick preparation run rejected manifest metadata serialization because empty outer-position cells were parsed as NaN. The CSV loader now preserves optional empty strings, with a strict JSON regression test. No model was trained or evaluated in this failed run. The rerun uses a new directory.

Visual inspection found that theoretical frequency markers shared one color. The rendering now assigns distinct colors to FTF/BPFO/BPFI/BSF. Final quick/full runs repeat the same frozen numerical protocol after this presentation correction. Their predictions are compared with the initial successful runs to check deterministic reproducibility; no settings are changed in response to accuracy.

Reproduction exposed one perturbed-window class difference in each initial quick/full repeat, despite probability differences no larger than 4.45e-16. Parallel forest probability accumulation can reorder floating-point addition at near ties. The saved forest now uses single-thread inference after parallel fitting; features, trees, seeds and training protocol are unchanged. The evidence is retained in `outputs/qa/parallel_inference_audit.json`. Final verified runs supersede these diagnostic runs.
