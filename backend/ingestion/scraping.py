import requests
from sqlalchemy.orm import Session
from backend.db import SessionLocal
from backend.models import Jobs
from backend.embedding import embed_text

# using remotive API for job listings
remotive_url = "https://remotive.com/api/remote-jobs?category=software-dev"
# print("This is the remotive api url: ",remotive_url)

def fetch_jobs():
    try:
        response = requests.get(remotive_url, timeout=10)
        response.raise_for_status()
        data = response.json()
        jobs = data["jobs"]
        return jobs
    except requests.RequestException as e:
        raise ValueError(f"Scraping Error - Error fetching jobs from API: {str(e)}")
    except KeyError:
        raise ValueError(f"Scraping Error - Unexpected API response structure: {data}")

# Normalize job data to match our DB schema
def normalize_job(job):
    text_for_embedding = (
        f"{job['title']} {job['description']} {job.get('job_type', '')}"
    )

    normalized = {
        "title": job["title"],
        "company": job["company_name"],
        "description": job["description"],
        "remote": job["candidate_required_location"] == "Worldwide",
        "skills": job.get("tags", []),
        "embedding": embed_text(text_for_embedding),
    }

    return normalized

# Collect and ingest jobs into the database
def ingest_jobs():
    try:
        db: Session = SessionLocal()
        jobs = fetch_jobs()

        inserted_count = 0

        for job in jobs:
            normalized = normalize_job(job)

            exists = db.query(Jobs).filter(
                Jobs.title == normalized["title"],
                Jobs.company == normalized["company"]
            ).first()

            # print("exists result: ", exists)

            # avoid duplicates
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
            print("inserted_count: ", inserted_count)

        db.commit()

        print(f"Inserted {inserted_count} new jobs into the database.")
    except Exception as e:
        db.rollback()
        raise ValueError(f"Scraping Error - Ingestion error: {str(e)}")
    finally:
        db.close()


if __name__ == "__main__":
    ingest_jobs()
