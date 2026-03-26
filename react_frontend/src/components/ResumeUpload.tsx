import type { ResumeData } from '../types/index'
import { uploadResume } from '../api/client'
import { useState } from 'react'
import styles from './css/ResumeUpload.module.css'

type Props = {
    onUpload: (data: ResumeData) => void
}

export function ResumeUpload({ onUpload }: Props) {
    const [loading, setLoading] = useState(false)
    const [done, setDone] = useState(false)
    async function handleFileUpload(e: React.ChangeEvent<HTMLInputElement>) {
        const file = e.target.files?.[0]
        if (!file) return
        setLoading(true)
        setDone(false)
        const data = await uploadResume(file)
        setLoading(false)
        setDone(true)
        onUpload(data)
    }
    return (
        <div>
            {loading && (
                <div>
                    <progress />
                    <p className={styles.processingText}>Processing resume ...</p>
                </div>
            )}
            {done && <p className={styles.uploadComplete}>Upload Complete!</p>}
            <label className={styles.fileButton}>
                Upload File
                <input type="file" accept=".pdf" onChange={handleFileUpload} hidden />
            </label>
        </div>
    )
}