import os
import requests
from sqlalchemy.orm import Session
from backend.db import SessionLocal
from backend.models import Jobs
from backend.ingestion.normalize import normalize_adzuna_job, normalize_remotive_job

def fetch_remotive_jobs():
    try:
        response = requests.get("https://remotive.com/api/remote-jobs?category=software-dev", timeout=10)
        response.raise_for_status()
        data = response.json()
        print(f"Fetched {len(data.get('jobs', []))} jobs from Remotive API")
        return data["jobs"]
    except requests.RequestException as e:
        raise ValueError(f"Scraping Error - Error fetching jobs from API: {str(e)}")
    except KeyError:
        raise ValueError(f"Scraping Error - Unexpected API response structure: {data}")

def fetch_adzuna_jobs(query="software developer", country="gb", results_per_page=50):
    app_id = os.getenv("ADZUNA_APP_ID")
    app_key = os.getenv("ADZUNA_APP_KEY")
    url = (
        f"http://api.adzuna.com/v1/api/jobs/{country}/search/1"
        f"?app_id={app_id}&app_key={app_key}"
        f"&results_per_page={results_per_page}&what={query}&content-type=application/json"
    )
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        print(f"Fetched {len(data.get('results', []))} jobs from Adzuna API")
        return data["results"]
    except requests.RequestException as e:
        raise ValueError(f"Scraping Error - Error fetching Adzuna jobs: {str(e)}")
    except KeyError:
        raise ValueError(f"Scraping Error - Unexpected Adzuna API response structure: {data}")

# Collect and ingest jobs into the database
def ingest_jobs():
    try:
        db: Session = SessionLocal()
        all_jobs = []

        remotive_jobs = fetch_remotive_jobs()
        all_jobs.extend((job, normalize_remotive_job) for job in remotive_jobs)

        adzuna_jobs = fetch_adzuna_jobs()
        all_jobs.extend((job, normalize_adzuna_job) for job in adzuna_jobs)

        inserted_count = 0

        for job, normalizer in all_jobs:
            normalized = normalizer(job)

            exists = db.query(Jobs).filter(
                Jobs.title == normalized["title"],
                Jobs.company == normalized["company"]
            ).first()

            if exists:
                continue

            db_job = Jobs(
                title=normalized["title"],
                company=normalized["company"],
                description=normalized["description"],
                remote=normalized["remote"],
                skills=normalized["skills"],
                embedding=normalized["embedding"],
            )
            db.add(db_job)
            inserted_count += 1
            # print("inserted_count: ", inserted_count)

        db.commit()
        print(f"Inserted {inserted_count} new jobs into the database.")
    except Exception as e:
        db.rollback()
        raise ValueError(f"Scraping Error - Ingestion error: {str(e)}")
    finally:
        db.close()


if __name__ == "__main__":
    ingest_jobs()
