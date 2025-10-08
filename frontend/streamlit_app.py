import streamlit as st
import requests


API_URL = "http://localhost:8000"


def main():
    st.title("AI Job Callback Agent")
    uploaded = st.file_uploader("Upload your resume (PDF)")
    if uploaded:
        files = {"file": uploaded.getvalue()}
        res = requests.post(f"{API_URL}/upload_resume/", files=files).json()
        resume_text, emb, skills = res["resume_text"], res["embedding"], res["skills"]
        if st.button("Find Best Jobs"):
            payload = {"resume_text": resume_text, "embedding": emb, "skills": skills}
            ranked = requests.post(f"{API_URL}/rank_jobs/", json=payload).json()
            for job in ranked["ranked_jobs"]:
                st.subheader(f"{job['title']} @ {job['company']}")
                st.write(f"Fit score: {job['score']}")
                if st.button(f"Explain match for {job['title']}"):
                    explain = requests.post(f"{API_URL}/explain_match/", json={
                        "resume_text": resume_text,
                        "job_title": job["title"],
                        "job_desc": job.get("description", ""),
                        "score": job["score"]
                    }).json()
                    st.info(explain["explanation"])


if __name__ == "__main__":
    main()