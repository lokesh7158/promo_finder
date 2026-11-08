# Quickstart

## 1. Install

Use Python 3.13 and Windows PowerShell from the repository root:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 2. Run the synthetic demo

Demo mode is the default and does not require an API key:

```powershell
Remove-Item Env:PROMOFINDER_MODE -ErrorAction SilentlyContinue
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open the local URL printed by Streamlit. On **Upload & Run**, download the synthetic sample pack, upload unchanged `.txt` files from that pack, and click **Run pipeline**. Demo mode is synthetic-only and disables provider controls and live LLM evaluation.

## 3. Run locally on a synthetic sample

Generate one synthetic file, then launch local mode:

```powershell
.\.venv\Scripts\python.exe scripts\generate_synthetic_samples.py --count 1
$env:PROMOFINDER_MODE = 'local'
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Upload the generated file from `samples/generated/` and click **Run pipeline**. This creates proposal inputs in a temporary workspace and makes no live LLM request.

## 4. Configure an LLM

On **Upload & Run**:

1. Choose Gemini, OpenAI, Claude, or Grok and select a supported model.
2. Enter your own API key only in the masked dashboard field.
3. Click **Test Connection**.
4. Upload authorized text and click **Run pipeline**.
5. Inspect the proposal shortlist.
6. Set the evaluation maximum (default 1), authorize sharing proposal excerpts, confirm Gemini free-tier use if selected, and click **Run LLM Evaluation**.

Review current provider charges, quotas, and data-handling rules first. Do not paste keys in chat, commit them, or put them in `.env`. Keys entered in the dashboard are held server-side in memory and cleared on process restart; saved session keys expire after 20 minutes idle. For full provider setup and limitations, see [README.md](README.md).

## 5. Run offline tests

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The tests are synthetic and mock provider APIs; they do not make live requests.

## Hosted deployment safety

For Streamlit Cloud, configure this secret:

```toml
PROMOFINDER_HOSTED = "1"
```

This forces demo mode even if `PROMOFINDER_MODE=local` is also set. Hosted mode disables provider configuration, API-key entry, live LLM evaluation, and local historical results. Do not configure provider API keys in the hosted deployment. Local development can still use `PROMOFINDER_MODE=local` for provider configuration and explicit live evaluation.
