from backend.parser import parse_job_requirements


def normalize_remotive_job(job):
    requirements = parse_job_requirements(job["description"])
    return {
        "title": job["title"],
        "company": job["company_name"],
        "description": job["description"],
        "remote": job["candidate_required_location"] == "Worldwide",
        "skills": job.get("tags", []),
        "embedding": None,
        "min_years_required": requirements["min_years_required"],
        "seniority_level": requirements["seniority_level"],
    }


def normalize_adzuna_job(job):
    description = job.get("description", "")
    requirements = parse_job_requirements(description)
    return {
        "title": job.get("title", ""),
        "company": job.get("company", {}).get("display_name", ""),
        "description": description,
        "remote": "remote" in job.get("location", {}).get("display_name", "").lower(),
        "skills": [],
        "embedding": None,
        "min_years_required": requirements["min_years_required"],
        "seniority_level": requirements["seniority_level"],
    }
