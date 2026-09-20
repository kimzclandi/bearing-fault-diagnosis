# Data provenance

Official sources:
- https://engineering.case.edu/bearingdatacenter/download-data-file
- https://engineering.case.edu/bearingdatacenter/48k-drive-end-bearing-fault-data
- https://engineering.case.edu/bearingdatacenter/normal-baseline-data
- https://engineering.case.edu/bearingdatacenter/bearing-information

Download with `python scripts/download_data.py --config configs/quick.yaml --run-dir outputs/download-check` or use `run_all.py --download`. Raw files stay in `data/raw`; a local directory can be configured with `data_dir`. Downloads validate MAT decoding and record SHA-256. TLS verification remains enabled. Repeated calls reuse existing files; processing revalidates contents and records hashes.

`manifests/cwru.csv` specifies URLs, numeric labels, nominal RPM, load, source sampling rate, defect diameter and split. The loader selects the `X<source_id>_DE_time` channel when present, otherwise requires exactly one drive-end channel; labels are taken from the official table, not inferred from waveform content. If `RPM` is missing, nominal RPM is used and explicitly recorded.

The 36 fault recordings belong to the official 48 kHz collection. The four normal records use the commonly used 48 kHz interpretation; the linked normal table does not explicitly state a per-file rate, and MAT files do not guarantee embedded rate metadata. This is an unresolved acquisition-metadata limitation, not a measured verification. Rate cannot be proven from array length alone.

Public download availability does not establish a redistribution license. No explicit dataset license was identified in the linked pages. Raw data are not bundled or relicensed by this project. Code dependency licenses remain separate.

Synthetic input can be generated with `python scripts/generate_example_data.py`. It only exercises signal loading and the interface. It is never mixed into CWRU training or real-data metrics. If the official host is unavailable, retry the downloader or supply the exact original MAT files locally; no silent data replacement occurs.

Observed source issue: `99.mat` embeds both `X098_DE_time` and `X099_DE_time`. Only `X099_DE_time` is selected for source 99, and the ignored channel is recorded. RPM lookup follows the selected channel prefix. This prevents an embedded training-condition signal from entering validation.

Additional observed metadata anomalies: source `174.mat` contains a single drive-end field `X173_DE_time` with 63,788 samples; the loader preserves its official row label and records the sole-channel fallback. Source `175.mat` contains `X175_DE_time` and an additional `X217_DE_time`; only the exact `X175` field is selected. Short records produce fewer windows rather than being padded or duplicated. These anomalies limit confidence in unverified acquisition metadata.
