import json
from embedding import cosine_sim, embed_text


def load_jobs():
    with open("../jobs_data/sample_jobs.json", "r") as f:
        return json.load(f)


def skill_overlap(resume_skills, job_skills):
    if not job_skills:
        return 0
    overlap = len(set(resume_skills) & set(job_skills))
    return overlap / len(job_skills)


def rank_jobs(resume_text, resume_emb, resume_skills):
    jobs = load_jobs()
    results = []
    for job in jobs:
        job_emb = embed_text(job["description"])
        sim = cosine_sim(resume_emb, job_emb)
        overlap = skill_overlap(resume_skills, job["skills"])
        score = round(0.7 * sim + 0.3 * overlap, 3)
        results.append({"title": job["title"], "company": job["company"], "score": score})
    return sorted(results, key=lambda x: x["score"], reverse=True)[:5]