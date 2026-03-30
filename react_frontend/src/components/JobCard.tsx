import { useState } from 'react'
import type { JobsData, ResumeData, ExplanationData } from '../types/index'
import styles from './css/JobCard.module.css'
import { explainMatch } from '../api/client'

type Props = {
    job: JobsData,
    resumeData: ResumeData | null
}

export function JobCard({ job, resumeData }: Props) {
    const [loading, setLoading] = useState(false)
    const [explanation, setExplanation] = useState<Record<string, unknown>>()

    async function handleClickExplanation() {
        if (!resumeData) return
        setLoading(true)

        const explain_payload: ExplanationData = {
            resume_text : resumeData.resume_text,
            job_title : job.title,
            job_desc : job.description,
            score : job.score,
            experience : {
                candidate_years : resumeData.years_experience,
                candidate_seniority : resumeData.seniority_level,
                job_min_years : job.min_years_required,
                job_seniority : job.seniority_level
            }
        }

        const res = await explainMatch(explain_payload)
        setLoading(false)
        setExplanation(res)
    }

    return (
        <div className={styles.card}>
            <h3 className={styles.cardTitle}>{job.title} @ {job.company}</h3>
            <p className={styles.cardMeta}>
                Fit score: <strong>{Math.round(job.score * 100)}%</strong>
                {job.seniority_level && ` · ${job.seniority_level}`}
                {job.min_years_required != null && ` · ${job.min_years_required}+ yrs required`}
            </p>
            {job.skills?.length > 0 && (
                <p className={styles.cardSkills}>Skills: {job.skills.join(', ')}</p>
            )}
            <p className={styles.cardDescription}>{job.description}</p>
            <button onClick={handleClickExplanation} disabled={loading || !resumeData}>
                {loading ? 'Generating...' : 'Explain Match'}
            </button>
            {loading && (
                <div>
                    <progress />
                    <p className={styles.processingText}>Generating an Explanation ...</p>
                </div>
            )}
            {explanation && (
                <div className={styles.explanationSection}>
                    <p className={styles.explanationSummary}>{explanation.summary as string}</p>
                    <div className={styles.explanationCategory}>
                        <strong>Strengths:</strong>
                        <ul>
                            {(explanation.strengths as string[]).map((s, i) => (
                                <li key={i}>{s}</li>
                            ))}
                        </ul>
                    </div>
                    <div className={styles.explanationCategory}>
                        <strong>Gaps:</strong>
                        <ul>
                            {(explanation.gaps as Array<Record<string, unknown>>).map((gap, i) => (
                                <li key={i}>
                                    <strong>{gap.skill as string}</strong> — Required: {gap.required as string}, You have: {gap.user_has as string}
                                    {gap.closeable ? ' (closeable)' : ' (not closeable)'}
                                </li>
                            ))}
                        </ul>
                    </div>
                    <div className={styles.explanationCategory}>
                        <strong>Quick Wins:</strong>
                        <ul>
                            {(explanation.quick_wins as string[]).map((qw, i) => (
                                <li key={i}>{qw}</li>
                            ))}
                        </ul>
                    </div>
                </div>
            )}
        </div>
    )
}
