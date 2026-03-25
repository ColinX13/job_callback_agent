from fastapi import HTTPException
from dotenv import load_dotenv
import asyncio
import os
from contextlib import asynccontextmanager

load_dotenv()

from fastapi import FastAPI, UploadFile, File, Body, Depends, BackgroundTasks
from sqlalchemy.orm import Session
from backend.db import SessionLocal
from backend.models import Jobs
from fastapi.middleware.cors import CORSMiddleware
from backend.parser import parse_resume
from backend.scoring import rank_jobs
from backend.explanation import explain_match
from backend.ingestion.scraping import ingest_jobs


### Scheduled scraping background task
async def scheduled_scraping():
    while True:
        try:
            ingest_jobs()
        except Exception as e:
            print(f"Error during scheduled scraping: {str(e)}")
        await asyncio.sleep(60 * 60 * 24) # every 24 hours

@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup code
    asyncio.create_task(scheduled_scraping())
    yield
    # shutdown code (clean up resources if needed)
    pass

app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.post("/admin/ingest_jobs/")
def trigger_scraping(background_tasks: BackgroundTasks):
    background_tasks.add_task(ingest_jobs)
    return {"status": "Scraping started in background"}

# API endpoint to upload resume and get embedding
@app.post("/upload_resume/")
async def upload_resume(file: UploadFile = File(...)):
    print("Received file:", file.filename)
    try:
        file_bytes = await file.read()
        text, skills, years_experience, seniority_level = parse_resume(file_bytes)
        return {"resume_text": text, "skills": skills, "years_experience": years_experience, "seniority_level": seniority_level}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing resume data: {str(e)}")

# API endpoint to list all jobs
@app.get("/jobs/")
def list_jobs(db: Session = Depends(get_db)):
    try:
        jobs = db.query(Jobs).all()
        return {"jobs": jobs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching jobs: {str(e)}")

# API endpoint to rank jobs based on resume keywords and embedding
@app.post("/rank_jobs/")
def rank_jobs_endpoint(payload: dict = Body(...), db: Session = Depends(get_db)):
    try:
        resume_text = payload.get("resume_text")
        resume_skills = payload.get("skills") or payload.get("resume_skills")
        resume_years = float(payload.get("years_experience") or 0)
        resume_seniority = payload.get("seniority_level") or "entry"
        if not resume_text or not resume_skills:
            raise ValueError("Missing required fields: resume_text or skills")
        ranked = rank_jobs(db, resume_text, resume_skills, resume_years, resume_seniority)
        return {"ranked_jobs": ranked}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error ranking jobs: {str(e)}")

# API endpoint to explain job match score with LLM prompt
@app.post("/explain_match/")
def explain_endpoint(payload: dict = Body(...)):
    try:
        resume_text = payload.get("resume_text")
        job_title = payload.get("job_title")
        job_desc = payload.get("job_desc")
        score = payload.get("score")
        experience = payload.get("experience")
        if not all([resume_text, job_title, job_desc, score is not None]):
            raise HTTPException(status_code=400, detail="Missing required fields: resume_text, job_title, job_desc, or score")
        explanation = explain_match(resume_text, job_title, job_desc, score, experience)
        return {"explanation": explanation}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating score explanation: {str(e)}")