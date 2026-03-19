import pytest
import requests
from unittest.mock import patch, MagicMock
from backend.ingestion.scraping import (
    fetch_remotive_jobs,
    normalize_remotive_job,
    fetch_adzuna_jobs,
    normalize_adzuna_job,
    ingest_jobs,
)
from backend.models import Jobs


# --- fetch_remotive_jobs ---

def test_fetch_remotive_jobs_success():
    mock_response = MagicMock()
    mock_response.json.return_value = {"jobs": [{"title": "Test Job"}]}

    with patch('backend.ingestion.scraping.requests.get', return_value=mock_response) as mock_get:
        url = "https://remotive.com/api/remote-jobs?category=software-dev"
        jobs = fetch_remotive_jobs(url)
        assert jobs == [{"title": "Test Job"}]
        mock_get.assert_called_once_with(url, timeout=10)


def test_fetch_remotive_jobs_api_error():
    with patch('backend.ingestion.scraping.requests.get', side_effect=requests.RequestException("Network error")):
        with pytest.raises(ValueError, match="Scraping Error - Error fetching jobs from API: Network error"):
            fetch_remotive_jobs("https://remotive.com/api/remote-jobs?category=software-dev")


def test_fetch_remotive_jobs_invalid_response():
    mock_response = MagicMock()
    mock_response.json.return_value = {"invalid": "data"}

    with patch('backend.ingestion.scraping.requests.get', return_value=mock_response):
        with pytest.raises(ValueError, match="Scraping Error - Unexpected API response structure"):
            fetch_remotive_jobs("https://remotive.com/api/remote-jobs?category=software-dev")


# --- normalize_remotive_job ---

def test_normalize_remotive_job():
    job_data = {
        "title": "Software Engineer",
        "company_name": "Test Co",
        "description": "Build apps",
        "candidate_required_location": "Worldwide",
        "tags": ["Python", "SQL"],
        "job_type": "Full-time"
    }

    with patch('backend.ingestion.scraping.embed_text', return_value=[0.1] * 384) as mock_embed:
        normalized = normalize_remotive_job(job_data)

        assert normalized["title"] == "Software Engineer"
        assert normalized["company"] == "Test Co"
        assert normalized["description"] == "Build apps"
        assert normalized["remote"] is True
        assert normalized["skills"] == ["Python", "SQL"]
        assert normalized["embedding"] == [0.1] * 384
        mock_embed.assert_called_once_with("Software Engineer Build apps Full-time")


# --- fetch_adzuna_jobs ---

def test_fetch_adzuna_jobs_success():
    mock_response = MagicMock()
    mock_response.json.return_value = {"results": [{"title": "Adzuna Job"}]}

    with patch('backend.ingestion.scraping.requests.get', return_value=mock_response) as mock_get, \
         patch.dict('os.environ', {"ADZUNA_APP_ID": "test_id", "ADZUNA_APP_KEY": "test_key"}):
        jobs = fetch_adzuna_jobs(query="python developer", country="gb", results_per_page=20)
        assert jobs == [{"title": "Adzuna Job"}]
        called_url = mock_get.call_args.args[0]
        assert "test_id" in called_url
        assert "test_key" in called_url
        assert "python+developer" in called_url or "python%20developer" in called_url or "python developer" in called_url
        assert "/gb/" in called_url
        assert "results_per_page=20" in called_url


def test_fetch_adzuna_jobs_api_error():
    with patch('backend.ingestion.scraping.requests.get', side_effect=requests.RequestException("Timeout")), \
         patch.dict('os.environ', {"ADZUNA_APP_ID": "id", "ADZUNA_APP_KEY": "key"}):
        with pytest.raises(ValueError, match="Scraping Error - Error fetching Adzuna jobs: Timeout"):
            fetch_adzuna_jobs()


def test_fetch_adzuna_jobs_invalid_response():
    mock_response = MagicMock()
    mock_response.json.return_value = {"unexpected": "data"}

    with patch('backend.ingestion.scraping.requests.get', return_value=mock_response), \
         patch.dict('os.environ', {"ADZUNA_APP_ID": "id", "ADZUNA_APP_KEY": "key"}):
        with pytest.raises(ValueError, match="Scraping Error - Unexpected Adzuna API response structure"):
            fetch_adzuna_jobs()


# --- normalize_adzuna_job ---

def test_normalize_adzuna_job():
    job_data = {
        "title": "Backend Developer",
        "company": {"display_name": "Acme Corp"},
        "description": "Build APIs",
        "location": {"display_name": "Remote"},
        "contract_type": "permanent",
    }

    with patch('backend.ingestion.scraping.embed_text', return_value=[0.2] * 384) as mock_embed:
        normalized = normalize_adzuna_job(job_data)

        assert normalized["title"] == "Backend Developer"
        assert normalized["company"] == "Acme Corp"
        assert normalized["description"] == "Build APIs"
        assert normalized["remote"] is True
        assert normalized["skills"] == []
        assert normalized["embedding"] == [0.2] * 384
        mock_embed.assert_called_once_with("Backend Developer Build APIs permanent")


def test_normalize_adzuna_job_non_remote():
    job_data = {
        "title": "Frontend Dev",
        "company": {"display_name": "Corp"},
        "description": "Build UI",
        "location": {"display_name": "London, UK"},
        "contract_type": "full_time",
    }

    with patch('backend.ingestion.scraping.embed_text', return_value=[0.1] * 384):
        normalized = normalize_adzuna_job(job_data)
        assert normalized["remote"] is False


# --- ingest_jobs ---

def test_ingest_jobs_success():
    remotive_jobs = [
        {"title": "Job 1", "company_name": "Co 1", "description": "Desc 1", "candidate_required_location": "Worldwide", "tags": ["Skill1"]},
    ]
    adzuna_jobs = [
        {"title": "Job 2", "company": {"display_name": "Co 2"}, "description": "Desc 2", "location": {"display_name": "Remote"}, "contract_type": "permanent"},
    ]

    with patch('backend.ingestion.scraping.fetch_remotive_jobs', return_value=remotive_jobs), \
         patch('backend.ingestion.scraping.fetch_adzuna_jobs', return_value=adzuna_jobs), \
         patch('backend.ingestion.scraping.normalize_remotive_job') as mock_norm_r, \
         patch('backend.ingestion.scraping.normalize_adzuna_job') as mock_norm_a, \
         patch('backend.ingestion.scraping.SessionLocal') as mock_session_class, \
         patch('backend.ingestion.scraping.embed_text', return_value=[0.1] * 384):

        mock_session = MagicMock()
        mock_session_class.return_value = mock_session
        mock_session.query.return_value.filter.return_value.first.return_value = None

        mock_norm_r.return_value = {"title": "Job 1", "company": "Co 1", "description": "Desc 1", "remote": True, "skills": ["Skill1"], "embedding": [0.1]*384}
        mock_norm_a.return_value = {"title": "Job 2", "company": "Co 2", "description": "Desc 2", "remote": True, "skills": [], "embedding": [0.1]*384}

        ingest_jobs()

        assert mock_session.add.call_count == 2
        mock_session.commit.assert_called_once()
        mock_session.close.assert_called_once()


def test_ingest_jobs_with_duplicates():
    remotive_jobs = [{"title": "Job 1", "company_name": "Co 1", "description": "Desc 1", "candidate_required_location": "Worldwide", "tags": []}]

    with patch('backend.ingestion.scraping.fetch_remotive_jobs', return_value=remotive_jobs), \
         patch('backend.ingestion.scraping.fetch_adzuna_jobs', return_value=[]), \
         patch('backend.ingestion.scraping.normalize_remotive_job', return_value={"title": "Job 1", "company": "Co 1", "description": "Desc 1", "remote": True, "skills": [], "embedding": [0.1]*384}), \
         patch('backend.ingestion.scraping.SessionLocal') as mock_session_class:

        mock_session = MagicMock()
        mock_session_class.return_value = mock_session
        mock_session.query.return_value.filter.return_value.first.return_value = MagicMock()  # Duplicate exists

        ingest_jobs()

        mock_session.add.assert_not_called()
        mock_session.commit.assert_called_once()


def test_ingest_jobs_error():
    with patch('backend.ingestion.scraping.fetch_remotive_jobs', side_effect=Exception("Fetch error")), \
         patch('backend.ingestion.scraping.SessionLocal') as mock_session_class:

        mock_session = MagicMock()
        mock_session_class.return_value = mock_session

        with pytest.raises(ValueError, match="Scraping Error - Ingestion error: Fetch error"):
            ingest_jobs()

        mock_session.rollback.assert_called_once()
        mock_session.close.assert_called_once()
