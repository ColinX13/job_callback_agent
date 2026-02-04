Instructions:

Requires Git and Python installation prior:

Git: https://git-scm.com/install/

Python: https://www.python.org/downloads/

Root Folder:
1. Clone project with SSH
2. Navigate to your project root
3. Run `pip3 install -r requirements.txt`

Start Backend:
1. Run `python3 -m backend`

Start Frontend:
1. `cd frontend`
2. Run `streamlit run streamlit_app.py`

Testing:
1. Run `pytest`
2. For coverage run: 
    - `pip3 install pytest-cov`
    - `pytest --cov=backend --cov-report=html`


Data Ingestion:
New data is ingested into the Database via a background task with FastAPI
Run this for manual trigger while backend is running:
    - `curl -X POST http://127.0.0.1:8000/admin/ingest_jobs/`