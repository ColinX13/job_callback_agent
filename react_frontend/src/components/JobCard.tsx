import type { JobsData } from '../types/index'
import styles from './css/JobCard.module.css'

// Dumb display component — receives one job, renders its data
export function JobCard({ job }: { job: JobsData }) {
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
        </div>
    )
}
