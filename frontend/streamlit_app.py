import streamlit as st
import requests


API_URL = "http://localhost:8000"


def main():
    st.title("Job Callback Agent")
    uploaded = st.file_uploader("Upload your resume (PDF)")
    if uploaded and "resume_data" not in st.session_state:
        files = {"file": uploaded.getvalue()}
        response = requests.post(f"{API_URL}/upload_resume/", files=files)
        res = response.json()
        if "resume_text" not in res:
            st.error(f"Upload failed: {res.get('detail', res)}")
            st.stop()
        st.session_state.resume_data = res

    if "resume_data" in st.session_state:
        resume_text = st.session_state.resume_data["resume_text"]
        skills = st.session_state.resume_data["skills"]
        years_experience = st.session_state.resume_data.get("years_experience", 0)
        seniority_level = st.session_state.resume_data.get("seniority_level", "entry")

        if st.button("Find Best Jobs"):
            payload = {
                "resume_text": resume_text,
                "skills": skills,
                "years_experience": years_experience,
                "seniority_level": seniority_level,
            }
            ranked = requests.post(f"{API_URL}/rank_jobs/", json=payload).json()
            print("Ranked jobs response: ", ranked)
            if "ranked_jobs" not in ranked:
                st.error(f"Ranking failed: {ranked.get('detail', ranked)}")
                st.stop()
            st.session_state.ranked_jobs = ranked["ranked_jobs"]

    if "ranked_jobs" in st.session_state:
        for idx, job in enumerate(st.session_state.ranked_jobs):
            st.subheader(f"{job['title']} @ {job['company']}")
            st.write(f"Fit score: {round(job['score'] * 100, 1)}%")
            if st.button(f"Explain match for {job['title']}", key=f"explain_{idx}"):
                st.write(f"Requesting explanation for job: {job['title']}")
                explain = requests.post(f"{API_URL}/explain_match/", json={
                    "resume_text": resume_text,
                    "job_title": job["title"],
                    "job_desc": job.get("description", ""),
                    "score": job["score"],
                    "experience": {
                        "candidate_years": years_experience,
                        "candidate_seniority": seniority_level,
                        "job_min_years": job.get("min_years_required"),
                        "job_seniority": job.get("seniority_level"),
                    }
                }).json()
                st.info(explain["explanation"])


if __name__ == "__main__":
    main()
