import streamlit as st
import requests


API_URL = "http://localhost:8000"


def main():
    st.title("Job Callback Agent")
    uploaded = st.file_uploader("Upload your resume (PDF)")
    if uploaded and "resume_data" not in st.session_state:
        files = {"file": uploaded.getvalue()}
        print("API_URL: ", API_URL)
        response = requests.post(f"{API_URL}/upload_resume/", files=files)
        # st.write(response.content)  # Show raw response in Streamlit
        res = response.json()
        st.session_state.resume_data = res
    
    if "resume_data" in st.session_state:
        resume_text = st.session_state.resume_data["resume_text"]
        emb = st.session_state.resume_data["embedding"]
        skills = st.session_state.resume_data["skills"]
        
        if st.button("Find Best Jobs"):
            payload = {"resume_text": resume_text, "embedding": emb, "skills": skills}
            ranked = requests.post(f"{API_URL}/rank_jobs/", json=payload).json()
            print("Ranked jobs response: ", ranked)  # Debug print
            st.session_state.ranked_jobs = ranked["ranked_jobs"]
        
    if "ranked_jobs" in st.session_state:
        for idx, job in enumerate(st.session_state.ranked_jobs):
            st.subheader(f"{job['title']} @ {job['company']}")
            st.write(f"Fit score: {job['score']}")
            if st.button(f"Explain match for {job['title']}", key=f"explain_{idx}"):
                st.write(f"Requesting explanation for job: {job['title']}")
                explain = requests.post(f"{API_URL}/explain_match/", json={
                    "resume_text": resume_text,
                    "job_title": job["title"],
                    "job_desc": job.get("description", ""),
                    "score": job["score"]
                }).json()
                st.info(explain["explanation"])


if __name__ == "__main__":
    main()