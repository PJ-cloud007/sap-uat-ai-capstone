# Synthetic SAP S/4HANA UAT Intelligence — V2-05

An interactive capstone prototype integrating synthetic requirements, live Gemini UAT candidate generation, human review, supervised defect triage, and explainable risk-based regression selection.

## Run locally

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

The app works without an API key for saved evidence, ML and regression. To enable *new* Gemini generation and the contextual assistant, add `GEMINI_API_KEY = "..."` in your **Streamlit app secrets**. The default model is `gemini-3.8-flash`, matching the previously tested V2-03 run; change the model ID in the sidebar if needed. Never commit secrets.

## Deploy

1. Create a GitHub repository and upload the **contents** of this folder, preserving `data/` and `models/`.
2. On Streamlit Community Cloud, create a new app from the repository and set the main file to `app.py`.
3. In app settings → Secrets, add your Gemini key as `GEMINI_API_KEY = "your-key"`.
4. Deploy and test each navigation page. Keep the repository private if your deployment workflow supports it; all included content is fictional.

## Scope and accuracy

- All inputs and outputs are synthetic; never paste real SAP or Defence client data.
- `models/defect_pipeline.joblib` was trained with scikit-learn 1.8.0. Package versions should be aligned when deployed.
- Existing V2-03 generated examples are prior live Gemini output; the app only makes a new API call when the user clicks Generate/Ask.
- Review edits are session-only; download the reviewed CSV to retain changes.
- Regression outputs are precomputed by V2-04 and are **not dynamically reoptimised** when filtering in the app.
- 30.02% effort saving and 82.13% historical exposure are retrospective synthetic metrics, not field-validated outcomes.
