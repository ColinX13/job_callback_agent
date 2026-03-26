import type { ResumeData, JobsData } from '../types/index'
import { rankJobData } from '../api/client'
import { useState } from 'react'
import { JobCard } from './JobCard'
import styles from './css/JobList.module.css'

type Props = {
    resumeData: ResumeData | null
}

// Smart component — owns jobs state, triggers fetch, maps results to JobCards
export function RankedJobsList({ resumeData }: Props) {
    const [jobs, setJobs] = useState<JobsData[]>([])
    const [loading, setLoading] = useState(false)

    async function handleFindJobs() {
        if (!resumeData) return
        setLoading(true)
        const ranked: JobsData[] = await rankJobData(resumeData)
        setJobs(ranked)
        setLoading(false)
    }

    return (
        <div>
            {resumeData && (<button className={styles.findButton} onClick={handleFindJobs} disabled={!resumeData || loading}>
                {loading? 'Finding jobs...' : 'Find Best Jobs'}
            </button>
            )}

            {jobs.map(job => (
                <JobCard key={job.id} job={job} />
            ))}
        </div>
    )
}
