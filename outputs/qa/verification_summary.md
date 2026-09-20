# Local verification summary

- Automated tests: 21 passed in each final run.
- Quick run: 40 source files, 850 windows, 57 prediction tables independently checked.
- Full run: 40 source files, 995 windows, 171 prediction tables independently checked.
- Re-inference: all 228 clean/perturbed prediction tables have exactly matching predicted classes. Float32 probability serialization is checked with atol 1e-7 and rtol 1e-6.
- Training repetition: 200 quick and 1800 full forest trees are identical to their previous fits; all four CNN state dictionaries match their corresponding previous training runs exactly. The RF inference worker count is now one to remove floating summation ordering variability.
- Streamlit AppTest: local example, empty file state, numeric CSV, numeric NPY and nonfinite CSV all passed.
- Browser: local example rendered; synthetic CSV produced predictions; invalid CSV produced a friendly error. Only localhost was used.
- Figures: 20 PNG files decode successfully; layout checked for curves, labels, legends and confusion matrices.
- Local Markdown links and technical-language checks passed.
- Docker executable unavailable: image build was not tested. GPU and independent hardware were not tested.

See the adjacent JSON files and each verified run's reports directory for machine-readable checks. No remote repository was created or changed.

Clean-environment archive check: a fresh virtual environment and extracted archive completed the entire quick pipeline and 21 tests. All 57 prediction tables have matching classes and metrics. Original checked data and cached dependency wheels were reused on the same machine.
