# PromoFinder AI

PromoFinder AI is an offline-first research prototype for turning authorized screenplay text into traceable promotional-video proposals. It includes a Streamlit dashboard and independently executable Python CLI stages. The project does not inspect video footage or replace editorial review.

The public repository intentionally excludes Kuku's confidential screenplay input pack, screenplay-derived results, and historical results. Hosted demo mode accepts synthetic content only.

## Quick navigation

- [Streamlit UI / Dashboard](#streamlit-ui--dashboard)
- [CLI execution](#cli-execution)
- [Troubleshooting](#troubleshooting)
- [Detailed setup](QUICKSTART.md)
- [Synthetic sample instructions](samples/README.md)

## Prerequisites

- Python 3.13
- Windows PowerShell examples below; use the equivalent activation command for your shell.
- Network access only when installing dependencies or intentionally making a live provider request.

Install dependencies from the repository root:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Runtime dependencies are pinned in `requirements.txt` (`streamlit` and `plotly`); the pipeline and provider HTTP clients otherwise use the Python standard library.

## A. Streamlit UI / Dashboard

The dashboard validates uploads, runs selected parsing and proposal stages, presents promo discovery and analytics, optionally requests provider evaluation after a separate explicit action, and offers downloads. Start it from the repository root:

```powershell
python -m streamlit run app.py
```

Open the local address Streamlit prints (normally `http://localhost:8501`). `app.py` is the dashboard entry point.

By default, the dashboard is in synthetic demo mode. On **Upload & Run**, download its sample pack, upload unchanged files from that pack, select pipeline stages, and click **Run pipeline**. To generate a smaller synthetic text sample locally:

```powershell
python scripts\generate_synthetic_samples.py --count 1
```

For permitted local input files, set local mode before launch:

```powershell
Remove-Item Env:PROMOFINDER_HOSTED -ErrorAction SilentlyContinue
$env:PROMOFINDER_MODE = 'local'
python -m streamlit run app.py
```

For local provider controls, also ensure a local Streamlit secrets file does not set `PROMOFINDER_HOSTED = "1"`.

Upload only files you are authorized to process. The pipeline button's Evaluation stage performs offline evaluation-input validation; it does not call an LLM or produce model scores. Inspect proposals in **Promo Discovery** and run summary charts in **Analytics**. **Historical Results** displays read-only data only when the expected local historical files are present; none are included in this repository. **Export** offers reports and a generated-output ZIP; outputs may contain screenplay-derived material, so do not publish them.

### Optional local provider evaluation

Local mode supports Gemini, OpenAI, Claude, and Grok. On **Upload & Run**, choose a provider/model and enter its key in the masked field, or configure the corresponding process environment variable (`GEMINI_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, or `XAI_API_KEY`). **Test Connection** makes an explicit provider request. To evaluate, first run the pipeline and inspect the proposals, then set a small maximum, explicitly authorize sharing selected proposal text and excerpts, and click **Run LLM Evaluation**. Gemini additionally requires the displayed free-tier confirmation. Check provider terms, quotas, and charges before sending content. Dashboard-entered keys are held in server memory; do not save keys in the repository.

### Hosted demo mode

In Streamlit Cloud, add this value under the app's **Secrets** settings:

```toml
PROMOFINDER_HOSTED = "1"
```

The application reads this setting from `st.secrets` (and the process environment). It forces demo mode even when `PROMOFINDER_MODE=local` is also present. Hosted mode accepts only unchanged generated synthetic samples and disables provider configuration, API-key entry, live LLM evaluation, and local historical results. Do not configure provider API keys in the hosted deployment.

## B. CLI execution

Use the CLI when you want to run individual stages, inspect intermediate files, or automate an offline workflow. These are executable scripts; `src/evaluation/ranking.py` is an internal module, not a standalone CLI. Each command below uses the same Python environment installed above.

Place authorized top-level `.txt` inputs in `data\input\episodes` (or set `$inputDir` to another directory). The parser writes JSON under `data\output`; keep inputs and outputs private. The following PowerShell sequence runs each available stage and then performs a safe proposal-evaluation dry run:

```powershell
$inputDir = 'data\input\episodes'
$outputRoot = 'data\output'
$parsedDir = Join-Path $outputRoot 'parsed'
$candidateDir = Join-Path $outputRoot 'candidates'
$llmInputDir = Join-Path $outputRoot 'llm_inputs'
$proposalDir = Join-Path $outputRoot 'promo_proposals'
$evaluationDir = Join-Path $outputRoot 'proposal_evaluations'
New-Item -ItemType Directory -Force -Path $inputDir | Out-Null

python src\ingestion\screenplay_parser.py --input-dir $inputDir --output-dir $parsedDir
python src\candidate_generation\candidate_generator.py --input-dir $parsedDir --output-dir $candidateDir
python src\evaluation\llm_input_preparer.py --candidate-dir $candidateDir --parsed-dir $parsedDir --screenplay-dir $inputDir --output-dir $llmInputDir
python src\promo_construction\promo_proposal_generator.py --parsed-dir $parsedDir --candidate-dir $candidateDir --output-dir $proposalDir
python src\evaluation\promo_evaluator.py --proposal-input (Join-Path $proposalDir 'shortlist.json') --parsed-dir $parsedDir --candidate-dir $candidateDir --output-dir $evaluationDir --dry-run
```

The parser also supports one file with `--file <path>` instead of `--input-dir <directory>`. All scripts accept `--help` for their actual options. The evaluation preparer verifies the original screenplay files against metadata in the parsed/candidate data, so keep those files available at `$inputDir`.

The `--dry-run` command validates proposal structure, source references, timestamps, offsets, and schemas without API calls or model scores. It writes `proposal_dry_run_summary.json`. To dry-run individual prepared candidates instead, run:

```powershell
python src\evaluation\promo_evaluator.py --input-dir $llmInputDir --output-dir (Join-Path $outputRoot 'evaluations') --dry-run
```

### Optional live CLI evaluation

The CLI reads provider configuration from the current process environment. Set `PROMO_LLM_PROVIDER` (`gemini`, `openai`, `claude`, or `grok`), optionally set `PROMO_LLM_MODEL`, and set the matching API-key variable listed above using a secure local method. For Gemini, use the project's supported model and set `PROMO_GEMINI_FREE_TIER_CONFIRMED=YES` only after verifying the account's free-tier and billing configuration. Never put a real key in source control or a published file.

For example, the non-secret Gemini settings are:

```powershell
$env:PROMO_LLM_PROVIDER = 'gemini'
$env:PROMO_LLM_MODEL = 'gemini-3.5-flash-lite'
$env:PROMO_GEMINI_FREE_TIER_CONFIRMED = 'YES'
```

Set `GEMINI_API_KEY` separately from a secure local source; do not enter a real key in a script or README.

After the proposal dry run and review, a bounded live run uses the same command without `--dry-run`, with a small `--limit`, for example:

```powershell
python src\evaluation\promo_evaluator.py --proposal-input (Join-Path $proposalDir 'shortlist.json') --parsed-dir $parsedDir --candidate-dir $candidateDir --output-dir $evaluationDir --limit 1
```

For individual prepared candidates instead of assembled proposals, run:

```powershell
python src\evaluation\promo_evaluator.py --input-dir $llmInputDir --output-dir (Join-Path $outputRoot 'evaluations') --limit 1
```

Use the same evaluation output directory to resume: valid completed records matching the input, provider, and model are reused. `--force` requests re-evaluation of completed entries. Evaluation outputs can include validated results, raw provider responses, failure records, summaries, and rankings; these can contain sensitive source material and must not be published.

The proposal evaluator writes `ranked_proposals.json` and `proposal_evaluation_summary.json` as part of live evaluation. Live candidate evaluation writes `ranked_candidates.json` and `evaluation_summary.json`. There is no separate ranking command. Candidate preparation writes per-episode LLM-input JSON; proposal generation writes `all_proposals.json`, `shortlist.json`, rejection/not-shortlisted files, and `summary.json`. The repository `.gitignore` excludes `data/` and generated samples.

## Tests

Run the offline suite (mock provider APIs; no live requests):

```powershell
python -m unittest discover -s tests -v
python -m compileall -q app.py services src tests scripts
```

## Troubleshooting

- If `py -3.13` is unavailable, install Python 3.13 and ensure the Python Launcher can find it.
- If PowerShell blocks environment activation, run commands with `.\.venv\Scripts\python.exe` instead of activating the environment.
- If Streamlit reports a missing dependency, activate `.venv` and rerun `python -m pip install -r requirements.txt`.
- If provider evaluation fails, verify the selected provider/model, the matching local key variable or dashboard key, and any required Gemini confirmation. A connection test is an explicit API request.
- For Streamlit Cloud, verify `PROMOFINDER_HOSTED = "1"` is present in app Secrets. No screenplay pack or local historical result directory is required to start the synthetic demo.

## Security and research limitations

Do not commit API keys, `.env` files, source screenplays, private reports, raw responses, or generated screenplay-derived results. The hosted demo uses synthetic content only. Model outputs and timestamp-based proposals require human and video/editorial review; the tool does not approve final edits.

No license is provided in this package. Obtain permission from the project owner before redistribution or reuse.
