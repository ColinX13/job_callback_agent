from backend.embedding import embed_text

def normalize_remotive_job(job):
    text_for_embedding = (
        f"{job['title']} {job['description']} {job.get('job_type', '')}"
    )

    return {
        "title": job["title"],
        "company": job["company_name"],
        "description": job["description"],
        "remote": job["candidate_required_location"] == "Worldwide",
        "skills": job.get("tags", []),
        "embedding": embed_text(text_for_embedding),
    }

def normalize_adzuna_job(job):
    title = job.get("title", "")
    description = job.get("description", "")
    contract_type = job.get("contract_type", "")
    text_for_embedding = f"{title} {description} {contract_type}"

    return {
        "title": title,
        "company": job.get("company", {}).get("display_name", ""),
        "description": description,
        "remote": "remote" in job.get("location", {}).get("display_name", "").lower(),
        "skills": [],
        "embedding": embed_text(text_for_embedding),
    }