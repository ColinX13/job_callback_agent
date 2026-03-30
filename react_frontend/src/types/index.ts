export type ResumeData = {
    resume_text: string,
    skills: string[],
    years_experience: number,
    seniority_level: string
}

export type JobsData = {
    id: number,
    title: string,
    company: string,
    description: string,
    skills: string[],
    min_years_required: number | null,
    seniority_level: string | null,
    score: number,
}

export type ExplanationData = {
    resume_text: string,
    job_title: string,
    job_desc: string,
    score: number,
    experience: {
        candidate_years: number,
        candidate_seniority: string,
        job_min_years: number | null,
        job_seniority: string | null
    } | null
}