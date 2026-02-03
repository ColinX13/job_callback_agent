import json
from sqlalchemy.orm import Session
from backend.models import Jobs
from backend.embedding import cosine_sim, embed_text


# def load_jobs():
#     with open("../jobs_data/sample_jobs.json", "r") as f:
#         return json.load(f)


def skill_overlap(resume_skills, job_skills):
    if not job_skills:
        return 0
    overlap = len(set(resume_skills) & set(job_skills))
    return overlap / len(job_skills)


def rank_jobs(db: Session, resume_text, resume_emb, resume_skills):
    try:
        jobs = db.query(Jobs).all()
        # print("All jobs: ",jobs[0].title)
        results = []
        for job in jobs:
            job_emb = job.embedding
            if isinstance(job_emb, str):
                try:
                    job_emb = json.loads(job_emb)
                except Exception:
                    job_emb = []
            # print("resume_emb:", resume_emb, "job_emb:", job_emb)
            sim = cosine_sim(resume_emb, job_emb)

            job_skills = job.skills or []
            if isinstance(job_skills, str):
                try:
                    job_skills = json.loads(job_skills)
                except Exception:
                    job_skills = []
            overlap = skill_overlap(resume_skills, job_skills)
            score = round(0.7 * sim + 0.3 * overlap, 3)
            results.append({
                "id": job.id,
                "title": job.title, 
                "company": job.company,
                "description": job.description,
                "skills": job.skills,
                "score": score
            })
        return sorted(results, key=lambda x: x["score"], reverse=True)[:10]
    except Exception as e:
        raise ValueError(f"Scoring error - Rank jobs failed: {str(e)}")