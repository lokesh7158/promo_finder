# PromoFinder AI

PromoFinder AI is an offline-first prototype for turning authorized screenplay text into traceable promotional-video proposals. It combines deterministic parsing, candidate generation, proposal construction, optional LLM evaluation, evidence validation, Python scoring, and review-oriented ranking in a Streamlit dashboard.

The project is a research prototype. It does not inspect video footage, approve final edits, or replace editorial review. Users are responsible for ensuring that uploaded material may be processed and, when selected, shared with their chosen LLM provider.

## What it does

- Parses UTF-8 screenplay text, scene structure, event labels, and source timestamps.
- Finds continuous 20–60 second candidate windows and constructs single- or multi-segment proposals.
- Preserves source-event references and maps source timestamps to assembled playback positions.
- Optionally asks one of four LLM providers for structured evaluation output.
- Validates model responses against the existing schema, source evidence, and timing rules.
- Calculates weighted scores and classifications in Python, then ranks validated proposals.
- Provides a local dashboard for uploads, review, analytics, and exports.
- Offers a synthetic-only hosted/demo mode with live provider access disabled.

## Architecture

```text
UTF-8 TXT uploads
      |
      v
Screenplay parser --> Parsed episode JSON
      |
      v
Candidate generator --> Continuous candidate JSON
      |                           |
      +--> LLM input preparation  |
      |                           v
      +--------------------> Proposal construction
                                  |
                      +-----------+-----------+
                      |                       |
                 Offline checks         Explicit LLM action
                                              |
                                  Gemini / OpenAI / Claude / Grok
                                              |
                                              v
                              JSON parsing and evidence validation
                                              |
                              Python scoring and classification
                                              |
                              Ranking, review, analytics, exports
```

`app.py` is the dashboard entry point. `services/pipeline.py` validates uploads and runs the existing command-line stages in a temporary workspace. `src/` contains the parser, candidate and proposal generators, provider adapters, evaluation prompts/schema, validator, scoring, and ranking. `services/llm_credentials.py` keeps dashboard credentials in process memory. The application does not need confidential project data to start or run the synthetic demo.

## Technology

- Python 3.13
- Streamlit and Plotly
- Python standard-library HTTP client for provider APIs
- `unittest` for offline tests

No separate provider SDK is required by this release.

## Install and start locally

From the repository root in Windows PowerShell:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
$env:PROMOFINDER_MODE = 'local'
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open the local URL printed by Streamlit, usually `http://localhost:8501`.

Local mode accepts up to 50 UTF-8 `.txt` files (10 MB per file and 100 MB total). Uploaded inputs and generated run artifacts are held in a temporary workspace and cleared when the run is cleared, replaced, or the process ends. Do not upload material unless you have permission to process it.

## Synthetic demo

The default mode is `demo`. In this mode, the dashboard offers a generated synthetic sample pack, accepts only the exact synthetic files it generates, and disables provider configuration, API-key input, live LLM requests, and local historical results.

To make a small synthetic sample for a local-mode workflow, run:

```powershell
.\.venv\Scripts\python.exe scripts\generate_synthetic_samples.py --count 1
```

Upload the resulting file from `samples/generated/` in local mode. It is generated demonstration text, not source screenplay content. The generated folder is ignored by Git.

## LLM providers

The dashboard offers Google Gemini, OpenAI, Anthropic Claude, and xAI Grok. It provides presets and custom model IDs where allowed by the adapter. Model availability and structured-output support depend on the provider account and selected model; verify current model documentation before use.

| Provider | Example preset | Key variable for local process configuration |
|---|---|---|
| Google Gemini | `gemini-3.5-flash-lite` | `GEMINI_API_KEY` |
| OpenAI | `gpt-4o-mini` | `OPENAI_API_KEY` |
| Anthropic Claude | `claude-sonnet-5-5` | `ANTHROPIC_API_KEY` |
| xAI Grok | `grok-4.7` | `XAI_API_KEY` |

These are example defaults, not a guarantee of current account access. Gemini is restricted in this project to its configured model allowlist and requires explicit free-tier confirmation. The code does not silently switch providers or models.

### Configure and run an evaluation

1. Select the provider and model on **Upload & Run**.
2. Enter the key in the password-masked field. Never paste it into chat, source code, `.env`, or generated files.
3. Use **Test Connection** to check key/model access. A connection test is an explicit provider request and does not prove that evaluation structured output will be accepted.
4. Upload files and click **Run pipeline**. This performs offline parsing and proposal generation; the pipeline's Evaluation stage is dry-run only.
5. Inspect proposals in **Promo Discovery**.
6. On **Upload & Run**, select a small maximum (default 1), explicitly authorize sending selected proposal text and excerpts to the provider, check Gemini's free-tier confirmation when applicable, and click **Run LLM Evaluation**.
7. Review result counts, provider-returned token usage, validated evidence, and rankings. Check the provider's current pricing, quota, and data-processing terms first. The dashboard does not invent cost estimates.

A successful connection does not automatically preserve an unsaved key for later actions. **Save for this session** stores it in process memory for up to 20 minutes idle; otherwise enter it again for the next action. Process restart clears in-memory keys. Streamlit temporarily holds unsubmitted password-widget content in server-side widget state until form submission. Do not enable external session persistence or share a local process with untrusted users.

## Verification status

The offline mock suite exercises all four provider adapters' request/response normalization and error paths. A single Gemini synthetic evaluation was separately reported successful during development. OpenAI, Claude, and Grok have **not** been live-verified for this release. Tests do not make real provider calls.

## Tests

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Tests use synthetic fixtures, mocked provider responses, and temporary directories. They do not read user screenplay directories or call external LLM APIs.

## Hosted demo deployment

For Streamlit Community Cloud, select this repository, branch `main`, and app file `app.py`. Explicitly set this secret:

```toml
PROMOFINDER_HOSTED = "1"
```

This hosted lock forces synthetic demo mode regardless of `PROMOFINDER_MODE`, and disables provider configuration, API-key entry, live LLM evaluation, and local historical results. Do not configure provider API keys in the hosted deployment. Local development continues to support provider configuration and explicitly requested live evaluation.

## Security and confidentiality

- Do not commit keys, `.env` files, uploaded screenplays, private reports, historical outputs, or generated screenplay-derived artifacts.
- `.gitignore` excludes local data, reports, secrets, generated samples, caches, virtual environments, and databases.
- Dashboard keys are session/provider scoped and held in process memory; they are not intentionally written to run outputs or logs.
- Only an explicit evaluation action sends proposal content to a selected provider. That content can contain screenplay excerpts, so treat provider requests and exports as sensitive.
- The export archive excludes the original upload directory, but generated outputs may still contain source excerpts and provider responses. Do not publish those archives.
- Hosted demo is intended for synthetic content only. Streamlit Cloud must have `PROMOFINDER_HOSTED = "1"` configured.

## Limitations

- Screenplay timestamps are not footage-verified edit points; every final promo requires video and editorial review.
- Semantic evidence support is heuristic and cannot guarantee that a model's interpretation is correct.
- Provider presets can become stale, and connection success does not guarantee structured-output compatibility.
- Provider mock tests do not establish live service behavior.
- Historical assignment results and screenplay materials are not part of this public-safe package.

## License

No license is provided in this package. Obtain permission from the project owner before redistributing or reusing it.
