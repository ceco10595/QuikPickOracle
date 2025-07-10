# 1. ingest the csv
python src/ingest.py data/error_codes.csv data/sample_qa.csv

# 2. launch the app
streamlit run src/app.py