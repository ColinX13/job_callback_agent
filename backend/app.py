from dotenv import load_dotenv
import os

load_dotenv()

from fastapi import FastAPI, UploadFile, File, Body, Depends
from sqlalchemy.orm import Session
from db import SessionLocal
from fastapi.middleware.cors import CORSMiddleware
from parser import parse_resume
from embedding import embed_text
from scoring import rank_jobs
from explanation import explain_match


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


@app.post("/upload_resume/")
async def upload_resume(file: UploadFile = File(...)):
    print("Received file:", file.filename)
    try:
        file_bytes = await file.read()
        text, skills = parse_resume(file_bytes)
        embedding = embed_text(text)
        return {"resume_text": text, "skills": skills, "embedding": embedding}
    except Exception as e:
        return {"error": str(e)}

@app.get("/jobs/")
def list_jobs(db: Session = Depends(get_db)):
    jobs = db.query(Jobs).all()
    return {"jobs": jobs}


@app.post("/rank_jobs/")
def rank_jobs_endpoint(payload: dict = Body(...), db: Session = Depends(get_db)):
    resume_text = payload.get("resume_text")
    resume_emb = payload.get("embedding") or payload.get("resume_emb")
    resume_skills = payload.get("skills") or payload.get("resume_skills")
    ranked = rank_jobs(db, resume_text, resume_emb, resume_skills)
    return {"ranked_jobs": ranked}

@app.post("/explain_match/")
def explain_endpoint(payload: dict = Body(...)):
    resume_text = payload["resume_text"]
    job_title = payload["job_title"]
    job_desc = payload["job_desc"]
    score = payload["score"]
    explanation = explain_match(resume_text, job_title, job_desc, score)
    return {"explanation": explanation}