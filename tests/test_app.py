import pytest
import asyncio
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock
from backend.app import app, startup_event, scheduled_scraping
from backend.db import SessionLocal
from backend.models import Jobs

client = TestClient(app)

@pytest.fixture()
def db_session():
    db = SessionLocal()
    yield db
    db.close()

@pytest.mark.asyncio
async def test_startup_event():
    # Mock ingest_jobs and asyncio.create_task
    with patch('backend.app.ingest_jobs') as mock_ingest, \
         patch('backend.app.asyncio.create_task') as mock_create_task:
        await startup_event()  # Call the function
        
        # Check that create_task was called (starts the loop)
        mock_create_task.assert_called_once()
        # The task's coro should be the scheduled_scraping function
        task_coro = mock_create_task.call_args[0][0]
        assert asyncio.iscoroutine(task_coro)  # It's a coroutine

@pytest.mark.asyncio
async def test_scheduled_scraping():
    with patch('backend.app.ingest_jobs') as mock_ingest, \
         patch('backend.app.asyncio.sleep', new_callable=AsyncMock) as mock_sleep:
        
        mock_sleep.side_effect = [None, Exception("Stop loop")]  # Stop after first iteration
        async def run_once():
            try:
                await scheduled_scraping()
            except Exception as e:
                if str(e) != "Stop loop":
                    raise

        await run_once()
        
        assert mock_ingest.call_count >= 1
        mock_sleep.assert_called_with(60 * 60 * 24)

def test_upload_resume(db_session):
    # Mock parse_resume to return dummy data
    with patch('backend.app.parse_resume', return_value=("dummy text", ["Python"])), \
         patch('backend.app.embed_text', return_value=[0.1] * 384):
        files = {"file": ("test_resume.pdf", b"fake resume content", "application/pdf")}
        response = client.post("/upload_resume/", files=files)
        assert response.status_code == 200
        data = response.json()
        assert "resume_text" in data
        assert data["resume_text"] == "dummy text"
        assert "skills" in data
        assert "embedding" in data

def test_trigger_scraping():
    with patch('backend.app.ingest_jobs') as mock_ingest:
        response = client.post("/admin/ingest_jobs/")
        assert response.status_code == 200
        data = response.json()
        assert data == {"status": "Scraping started in background"}


def test_rank_jobs(db_session):
    # Clear existing jobs
    db_session.query(Jobs).delete()
    db_session.commit()

    # Add test jobs to the database
    job = Jobs(
        title="Test Job",
        company="Test Company",
        description="Test Description",
        skills=["Python"],
        embedding=[0.1] * 384
    )
    db_session.add(job)
    db_session.commit()

    payload = {
        "resume_text": "Test Text",
        "embedding": [0.1] * 384,
        "skills": ["Python"]
    }
    response = client.post("/rank_jobs/", json=payload)
    assert response.status_code == 200
    data = response.json()
    print("Ranked jobs data: ", data)
    assert "ranked_jobs" in data

def test_explain_match():
    payload = {
        "resume_text": "Test Text",
        "job_title": "Test Job",
        "job_desc": "Test Description",
        "score": 0.9
    }
    response = client.post("/explain_match/", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "explanation" in data