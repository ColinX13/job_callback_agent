import type { ResumeData, JobsData, ExplanationData } from '../types/index'

const backend_url = import.meta.env.VITE_API_URL

export async function uploadResume(file: File): Promise<ResumeData> {
    const formData = new FormData()
    formData.append('file', file)
    console.log('client.ts - formData: ', formData)
    
    const res = await fetch(`${backend_url}/upload_resume/`, {
        method: 'POST',
        body: formData,
    })
    const data = await res.json()
    return data
}

export async function getJobsData(): Promise<JobsData[]> {
    const res = await fetch(`${backend_url}/jobs/`)
    const data = await res.json()
    return data.jobs
}

export async function rankJobData(resume: ResumeData): Promise<JobsData[]> {
    const resume_payload = JSON.stringify({
        resume_text: resume.resume_text,
        skills: resume.skills,
        years_experience: resume.years_experience,
        seniority_level: resume.seniority_level,
    })
    const res = await fetch(`${backend_url}/rank_jobs/`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: resume_payload,
    })
    const data = await res.json()
    return data.ranked_jobs
}

export async function explainMatch(explanation: ExplanationData): Promise<Record<string, unknown>> {
    const explanation_payload = JSON.stringify({
        resume_text: explanation.resume_text,
        job_title: explanation.job_title,
        job_desc: explanation.job_desc,
        score: explanation.score,
        experience: explanation.experience
    })
    const res = await fetch(`${backend_url}/explain_match/`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: explanation_payload
    })
    const data = await res.json()
    return data.explanation
}