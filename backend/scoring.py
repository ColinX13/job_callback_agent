import json
from sqlalchemy.orm import Session
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity
from backend.models import Jobs

# Resume and job description boilerplate that adds no discriminative signal
JOB_RESUME_BOILERPLATE = frozenset([
    # Job posting boilerplate
    "responsibilities", "responsibility", "requirements", "required",
    "qualifications", "qualification", "preferred", "desired",
    "benefits", "compensation", "salary", "perks",
    "description", "overview", "summary", "position",
    "opportunity", "opportunities", "role", "duties",
    "apply", "application", "applicant", "applicants",
    "employer", "employment", "employee",
    "company", "organization",
    "equal", "diversity", "eeo",
    # Resume boilerplate
    "resume", "curriculum", "vitae", "objective",
    "education", "university", "college", "degree",
    "references", "available", "request",
    "experience", "experienced", "professional",
    # Filler adjectives / phrases
    "excellent", "strong", "proven", "solid",
    "outstanding", "exceptional", "passionate",
    "dynamic", "motivated", "self", "starter",
    "detail", "oriented", "driven",
    "fast", "paced", "environment",
    "ability", "able", "capable",
    "including", "includes", "etc",
    "looking", "seeking", "join",
    # Generic soft skill words
    "communication", "interpersonal", "organizational",
    "collaborate", "collaboration", "collaborative",
    "team", "player", "teamwork",
    "work", "working", "worked",
    # Non-discriminative verbs
    "manage", "managed", "management",
    "develop", "developed", "developing",
    "ensure", "ensuring",
    "support", "supporting", "supported",
    "provide", "providing", "provided",
    "maintain", "maintaining", "maintained",
    "responsible", "lead", "leading",
    "implement", "implemented", "implementing",
    "utilize", "utilizing", "utilized",
    "leverage", "leveraging", "leveraged",
])

CUSTOM_STOP_WORDS = list(ENGLISH_STOP_WORDS.union(JOB_RESUME_BOILERPLATE))


def skill_overlap(resume_skills, job_skills):
    if not job_skills:
        return 0
    overlap = len(set(resume_skills) & set(job_skills))
    return overlap / len(job_skills)


def apply_experience_penalty(
    base_score: float,
    candidate_years: float,
    job_min_years: float | None,
    job_seniority: str,
    candidate_seniority: str,
) -> float:
    if job_min_years is None:
        return base_score

    # Hard filter: job requires 2+ more years than candidate has
    if job_min_years > candidate_years + 2:
        return base_score * 0.1

    # Soft penalty: job requires more experience but within 2 years
    if job_min_years > candidate_years:
        gap = job_min_years - candidate_years
        penalty = 1.0 - (gap * 0.10)  # 10% per year gap
        return base_score * max(penalty, 0.5)

    # Seniority mismatch even when years are satisfied
    seniority_order = ["entry", "mid", "senior", "lead"]
    candidate_level = seniority_order.index(candidate_seniority) if candidate_seniority in seniority_order else 0
    job_level = seniority_order.index(job_seniority) if job_seniority in seniority_order else candidate_level

    if job_level - candidate_level > 1:
        return base_score * 0.3

    return base_score

### --- Scoring functionality for Web App with Jobs DB, using TF-IDF Vectorization
def _build_vectorizer():
    return TfidfVectorizer(
        stop_words=CUSTOM_STOP_WORDS,
        ngram_range=(1, 2),
        max_df=0.85,
        sublinear_tf=True,
    )

def rank_jobs(db: Session, resume_text, resume_skills, resume_years=0.0, resume_seniority="entry"):
    try:
        jobs = db.query(Jobs).all()

        # Build corpus: resume first, then all job texts
        job_texts = [f"{job.title} {job.description or ''}" for job in jobs]
        corpus = [resume_text] + job_texts

        vectorizer = _build_vectorizer()
        tfidf_matrix = vectorizer.fit_transform(corpus)

        resume_vec = tfidf_matrix[0]
        job_vecs = tfidf_matrix[1:]
        sims = cosine_similarity(resume_vec, job_vecs)[0]

        results = []
        for i, job in enumerate(jobs):
            job_skills = job.skills or []
            if isinstance(job_skills, str):
                try:
                    job_skills = json.loads(job_skills)
                except Exception:
                    job_skills = []
            overlap = skill_overlap(resume_skills, job_skills)
            base_score = round(0.7 * float(sims[i]) + 0.3 * overlap, 3)
            score = round(apply_experience_penalty(
                base_score,
                resume_years,
                job.min_years_required,
                job.seniority_level or "any",
                resume_seniority,
            ), 3)
            results.append({
                "id": job.id,
                "title": job.title,
                "company": job.company,
                "description": job.description,
                "skills": job.skills,
                "min_years_required": job.min_years_required,
                "seniority_level": job.seniority_level,
                "score": score,
            })
        return sorted(results, key=lambda x: x["score"], reverse=True)[:10]
    except Exception as e:
        raise ValueError(f"Scoring error - Rank jobs failed: {str(e)}")
    
### --- Scoring functionality for Chrome Extension, only TF Vectorization
def _build_single_job_vectorizer():
    return TfidfVectorizer(
        stop_words=CUSTOM_STOP_WORDS,
        ngram_range=(1, 2),
        sublinear_tf=True,
        use_idf=False,  # Pure TF — IDF meaningless with only 2 docs
    )


def extract_skills_from_text(text, reference_skills):
    """Find which reference skills appear in the text (case-insensitive)."""
    text_lower = text.lower()
    return [s for s in reference_skills if s.lower() in text_lower]


def score_single_job(resume_text, job_description, resume_skills=None, resume_years=0.0, resume_seniority="entry"):
    """Score a single job description against a resume. No DB required."""
    try:
        from backend.parser import parse_job_requirements

        corpus = [resume_text, job_description]
        vectorizer = _build_single_job_vectorizer()
        tf_matrix = vectorizer.fit_transform(corpus)
        tf_sim = float(cosine_similarity(tf_matrix[0], tf_matrix[1])[0][0])

        # Parse job requirements for experience/seniority penalties
        requirements = parse_job_requirements(job_description)
        job_min_years = requirements["min_years_required"]
        job_seniority = requirements["seniority_level"] or "any"

        # Extract job skills from JD text by matching against resume skills
        overlap = 0.0
        if resume_skills:
            matched_skills = extract_skills_from_text(job_description, resume_skills)
            overlap = skill_overlap(resume_skills, matched_skills)

        base_score = round(0.7 * tf_sim + 0.3 * overlap, 3)
        score = round(apply_experience_penalty(
            base_score, resume_years, job_min_years, job_seniority, resume_seniority,
        ), 3)

        return {
            "score": score,
            "tf_similarity": round(tf_sim, 3),
            "min_years_required": job_min_years,
            "seniority_level": job_seniority,
        }
    except Exception as e:
        raise ValueError(f"Scoring error - score_single_job failed: {str(e)}")
