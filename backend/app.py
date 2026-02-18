from fastapi import HTTPException
from dotenv import load_dotenv
import asyncio
import os
from contextlib import asynccontextmanager

load_dotenv()

from fastapi import FastAPI, UploadFile, File, Body, Depends, BackgroundTasks
from sqlalchemy.orm import Session
from backend.db import SessionLocal
from fastapi.middleware.cors import CORSMiddleware
from backend.parser import parse_resume
from backend.embedding import embed_text
from backend.scoring import rank_jobs
from backend.explanation import explain_match
from backend.ingestion.scraping import ingest_jobs


app = FastAPI()
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
        text, skills = parse_resume(file_bytes)
        embedding = embed_text(text)
        return {"resume_text": text, "skills": skills, "embedding": embedding}
    except Exception as e:
        return HTTPException(status_code=500, detail=f"Error processing resume data: {str(e)}")

# API endpoint to list all jobs
@app.get("/jobs/")
def list_jobs(db: Session = Depends(get_db)):
    try:
        jobs = db.query(Jobs).all()
        return {"jobs": jobs}
    except Exception as e:
        return HTTPException(status_code=500, detail=f"Error fetching jobs: {str(e)}")

# API endpoint to rank jobs based on resume keywords and embedding
@app.post("/rank_jobs/")
def rank_jobs_endpoint(payload: dict = Body(...), db: Session = Depends(get_db)):
    try:
        resume_text = payload.get("resume_text")
        resume_emb = payload.get("embedding") or payload.get("resume_emb")
        resume_skills = payload.get("skills") or payload.get("resume_skills")
        if not resume_text or not resume_emb or not resume_skills:
            raise ValueError("Missing required fields: resume_text, embedding, or skills")
        ranked = rank_jobs(db, resume_text, resume_emb, resume_skills)
        return {"ranked_jobs": ranked}
    except Exception as e:
        return HTTPException(status_code=500, detail=f"Error ranking jobs: {str(e)}")

# API endpoint to explain job match score with LLM prompt
@app.post("/explain_match/")
def explain_endpoint(payload: dict = Body(...)):
    try:
        resume_text = payload["resume_text"]
        job_title = payload["job_title"]
        job_desc = payload["job_desc"]
        score = payload["score"]
        explanation = explain_match(resume_text, job_title, job_desc, score)
        return {"explanation": explanation}
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missing required fields in request: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating score explanation: {str(e)}")