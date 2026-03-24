import type { ResumeData } from '../types/index'
import { uploadResume } from '../api/client'
import { useState } from 'react'

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
                    <p style={{ fontSize: "0.9rem" }}>Processing resume ...</p>
                </div>
            )}
            {done && <p style={{ color: 'green', fontSize: "0.9rem" }}>Upload Complete!</p>}
            <input type="file" accept=".pdf" onChange={handleFileUpload} />
        </div>
    )
}