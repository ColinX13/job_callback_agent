import type { ResumeData } from '../types/index'

export async function uploadResume(file: File): Promise<ResumeData> {
    const formData = new FormData()
    formData.append('file', file)
    console.log('client.ts - formData: ', formData)
    const res = await fetch('http://localhost:8000/upload_resume/', {
        method: 'POST',
        body: formData,
    })
    const data = await res.json()
    return data
}