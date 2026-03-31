const BACKEND_URL = CONFIG.BACKEND_URL

// check if file is already stored
chrome.storage.local.get('resumeName', ({ resumeName }) => {
    if (resumeName) {
        showStatus(`Resume Uploaded: ${resumeName}`, 'Success')
    }
})

document.getElementById('resume-upload').addEventListener('change', async (e) => {
    const file = e.target.files[0]
    if (!file) return
    console.log("something here")

    // PDF call backend, txt for local
    if (file.name.endsWith('.txt')) {
        const text = await file.text
        await chrome.storage.local.set({
            resumeText: text,
            resumeName: file.name
        })
        showStatus(`Resume Saved: ${file.name}`, 'success')
    } else {
        // send PDF to backend
        const formData = new FormData()
        formData.append('file', file)

        try {
            const response = await fetch(`${BACKEND_URL}/upload_resume/`, {
                method: 'POST',
                body: formData
            })
            const data = await response.json()
            chrome.storage.local.set({
                resumeText: data.resume_text,
                resumeSkills: data.skills,
                resumeYears: data.years_experience,
                resumeSeniority: data.seniority_level,
                resumeName: file.name
            })
            console.log("Resume Uploaded Successfully: ", data)
            showStatus(`Resume Saved: ${file.name}`, 'success')
        } catch {
            showStatus('Upload failed. Is backend running?', 'error')
        }
    }
})

// Manual JD fallback — paste and analyze from popup
document.getElementById('manual-analyze').addEventListener('click', async () => {
    const jdText = document.getElementById('manual-jd').value.trim()
    if (!jdText) {
        showManualStatus('Please paste a job description first.', 'error')
        return
    }

    const { resumeText, resumeSkills, resumeYears, resumeSeniority } = await chrome.storage.local.get([
        'resumeText', 'resumeSkills', 'resumeYears', 'resumeSeniority'
    ])

    if (!resumeText) {
        showManualStatus('Upload a resume first.', 'error')
        return
    }

    showManualStatus('Analyzing...', 'success')

    try {
        const response = await fetch(`${BACKEND_URL}/extension/analyze_one_job/`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                job_description: jdText,
                job_title: '',
                resume_text: resumeText,
                skills: resumeSkills || [],
                years_experience: resumeYears || 0,
                seniority_level: resumeSeniority || 'entry'
            })
        })
        const data = await response.json()
        const score = Math.round(data.score * 100)
        showManualStatus(`Score: ${score}% — Check the page sidebar for full analysis.`, 'success')

        // Store result so content script can pick it up
        await chrome.storage.local.set({ manualAnalysis: data })
    } catch {
        showManualStatus('Analysis failed. Is backend running?', 'error')
    }
})

function showStatus(message, type) {
    const el = document.getElementById('resume-status')
    el.textContent = message
    el.className = `status ${type}`
}

function showManualStatus(message, type) {
    const el = document.getElementById('manual-status')
    el.textContent = message
    el.className = `status ${type}`
}