# Synthetic samples

No screenplay files are distributed in this directory. Generate harmless synthetic UTF-8 `.txt` samples locally with:

```powershell
.\.venv\Scripts\python.exe scripts\generate_synthetic_samples.py --count 1
```

Generated files go to `samples/generated/`, which is ignored by Git. These generated samples can be used to exercise the local offline pipeline and, with explicit consent and provider credentials, a bounded live evaluation.

Streamlit Cloud must set `PROMOFINDER_HOSTED = "1"` in its secrets. Hosted mode forces the synthetic demo and disables provider setup and live evaluation.

For dashboard upload instructions and the executable CLI stages, see the [main README](../README.md).
