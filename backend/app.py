from fastapi import FastAPI, UploadFile, File
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
    text, skills = parse_resume(await file.read())
    embedding = embed_text(text)
    return {"resume_text": text, "skills": skills, "embedding": embedding}


@app.post("/rank_jobs/")
def rank_jobs_endpoint(resume_text: str, embedding: list, skills: list):
    ranked = rank_jobs(resume_text, embedding, skills)
    return {"ranked_jobs": ranked}


@app.post("/explain_match/")
def explain_endpoint(resume_text: str, job_title: str, job_desc: str, score: float):
    explanation = explain_match(resume_text, job_title, job_desc, score)
    return {"explanation": explanation}