# ml-model-selection-Streamlit

Upload a CSV, pick a target column, and this app will:
- profile your dataset (target type, missingness, cardinality, imbalance)
- recommend a shortlist of ML models
- optionally run a quick baseline pipeline (fast eval)

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
