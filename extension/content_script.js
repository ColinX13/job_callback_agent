// content_script.js

const BACKEND_URL = CONFIG.BACKEND_URL

// scrape job details based on what platform you are on (i.e. Greenhouse, Lever, Ashby)

function scrapeJobDetails() {
    const url = window.location.href

    if (url.includes('greenhouse.io')) {
        return scrapeGreenHouse()
    } else if (url.includes('lever.co')) {
        return scrapeLever()
    } else if (url.includes('ashbyhq.com')) {
        return scrapeAshby()
    }

    return null
}

function scrapeGreenHouse() {
    const title = document.querySelector('.app-title')?.innerText?.trim()
        || document.querySelector('h1')?.innerText?.trim()
        || 'Unknown Role'

    const company = document.querySelector('.company-name')?.innerText?.trim()
        || document.title.split(' at ')?.[1]?.trim()
        || 'Unknown Company'

    // Greenhouse puts job content in #content or .job-post
    const descriptionEl = document.querySelector('#content')
        || document.querySelector('.job-post')
        || document.querySelector('main')

    const description = descriptionEl?.innerText?.trim() || ''

    return { title, company, description, url: window.location.href, source: 'greenhouse' }
}

function scrapeLever() {
    const title = document.querySelector('.posting-headline h2')?.innerText?.trim()
        || document.querySelector('h2')?.innerText?.trim()
        || 'Unknown Company'

    const company = document.querySelector('.main-header-logo img')?.alt?.trim()
        || window.location.pathname.split('/')?.[1]?.trim()
        || 'Unknown Company'

    // Lever puts job content in .posting-description
    const descriptionEl = document.querySelector('.posting-description')
        || document.querySelector('.section-wrapper')
        || document.querySelector('main')

    const description = descriptionEl?.innerText?.trim() || ''

    return { title, company, description, url: window.location.href, source: 'lever' }
}

function scrapeAshby() {
    const title = document.querySelector('h1')?.innerText?.trim()
        || 'Unknown Role'
        
    const company = document.querySelector(
        '[data-testid="company-name"]'
    )?.innerText?.trim()
        || window.location.pathname.split('/')?.[1]?.trim()
        || 'Unknown Company'

    // Ashby puts content in main or article
    const descriptionEl = document.querySelector('article')
        || document.querySelector('main')
        
    const description = descriptionEl?.innerText?.trim() || ''
    
    return { 
        title, 
        company, 
        description, 
        url: window.location.href, 
        source: 'ashby' 
    }
}

// -- rendering the HTML sidebar
function injectSideBar() {
    // Don't inject twice
    if (document.getElementById('jca-sidebar')) return

    const sidebar = document.createElement('div')
    sidebar.id = 'jca-sidebar'
    sidebar.innerHTML = `
        <div id="jca-header">
        <span id="jca-title">Job Callback Agent</span>
        <button id="jca-close">✕</button>
        </div>
        <div id="jca-content">
        <div id="jca-loading">
            <div class="jca-spinner"></div>
            <p>Analyzing job fit...</p>
        </div>
        <div id="jca-result" style="display:none"></div>
        <div id="jca-error" style="display:none"></div>
        </div>
    `
    document.body.appendChild(sidebar)
    document.getElementById('jca-close').addEventListener('click', () => {
        sidebar.style.display = 'none'
    })

    // Push page content left so sidebar doesn't overlap
    document.body.style.marginRight = '380px'
}

// -- render analysis result

function renderResult(data) {
    const resultEl = document.getElementById('jca-result')
    const loadingEl = document.getElementById('jca-loading')

    const displayScore = Math.round(data.score * 100)
    const scoreColor = displayScore >= 60 ? '#22c55e'
        : displayScore >= 30 ? '#f59e0b'
        : '#ef4444'
        
    const gapsHtml = data.explanation.gaps?.map(gap => `
        <li class="jca-gap ${gap.closeable ? 'closeable' : 'uncloseable'}">
        <strong>${gap.skill}</strong>
        <span class="jca-badge ${gap.closeable ? 'badge-yellow' : 'badge-red'}">
            ${gap.closeable ? 'Closeable' : 'Not closeable'}
        </span>
        <p>${gap.required} — you have: ${gap.user_has}</p>
        </li>
    `).join('') || '<li>No significant gaps identified</li>'

    const strengthsHtml = data.explanation.strengths?.map(s => 
        `<li>✓ ${s}</li>`
    ).join('') || ''

    const quickWinsHtml = data.explanation.quick_wins?.map(w => 
        `<li>→ ${w}</li>`
    ).join('') || ''

    resultEl.innerHTML = `
        <div class="jca-score-block">
        <div class="jca-score" style="color: ${scoreColor}">
            ${displayScore}%
        </div>
        <div class="jca-seniority">${data.candidate_experience?.seniority_level || ''} level</div>
        <p class="jca-summary">${data.explanation.summary}</p>
        </div>

        ${strengthsHtml ? `
        <div class="jca-section">
            <h3>Strengths</h3>
            <ul>${strengthsHtml}</ul>
        </div>` : ''}

        <div class="jca-section">
        <h3>Gaps</h3>
        <ul class="jca-gaps">${gapsHtml}</ul>
        </div>

        ${quickWinsHtml ? `
        <div class="jca-section">
            <h3>Quick Wins</h3>
            <ul>${quickWinsHtml}</ul>
        </div>` : ''}

        <button class="jca-btn" id="jca-experience-builder">
        Build My Plan →
        </button>
    `
    loadingEl.style.display = 'none'
    resultEl.style.display = 'block'
}

// -- Error Handling

function renderError(message) {
  document.getElementById('jca-loading').style.display = 'none'
  const errorEl = document.getElementById('jca-error')
  errorEl.innerHTML = `<p class="jca-error-msg">${message}</p>`
  errorEl.style.display = 'block'
}

// -- Main flow

async function main() {
    const { resumeText, resumeSkills, resumeYears, resumeSeniority } = await chrome.storage.local.get([
        'resumeText', 'resumeSkills', 'resumeYears', 'resumeSeniority'
    ])

    if (!resumeText) {
        injectSideBar()
        renderError('No resume found. Please upload a new resume in the extension popup')
        return
    }

    const jobDetails = scrapeJobDetails()
    console.log("Job details scraped: ", jobDetails)

    if (!jobDetails || !jobDetails.description) {
        // Not a job page or could not be scraped
        return
    }

    injectSideBar()

    try {
        console.log("Running the extension analysis endpoint")
        const response = await fetch(`${BACKEND_URL}/extension/analyze_one_job/`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                job_description: jobDetails.description,
                job_title: jobDetails.title,
                resume_text: resumeText,
                skills: resumeSkills || [],
                years_experience: resumeYears || 0,
                seniority_level: resumeSeniority || "entry"
            })
        })

        if (!response.ok) {
            throw new Error('Something went wrong with the response generation')
        }

        const data = await response.json()
        renderResult(data)

    } catch (error) {
        renderError(`Analysis failed. Check if your backend is properly running. Error: ${error}`)
    }
}

main()