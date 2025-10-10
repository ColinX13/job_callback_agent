from dotenv import load_dotenv
import os

load_dotenv()

from fastapi import FastAPI, UploadFile, File, Body
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


@app.post("/rank_jobs/")
def rank_jobs_endpoint(payload: dict = Body(...)):
    resume_text = payload["resume_text"]
    embedding = payload["embedding"]
    skills = payload["skills"]
    ranked = rank_jobs(resume_text, embedding, skills)
    return {"ranked_jobs": ranked}


@app.post("/explain_match/")
def explain_endpoint(payload: dict = Body(...)):
    resume_text = payload["resume_text"]
    job_title = payload["job_title"]
    job_desc = payload["job_desc"]
    score = payload["score"]
    explanation = explain_match(resume_text, job_title, job_desc, score)
    return {"explanation": explanation}