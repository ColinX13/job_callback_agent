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