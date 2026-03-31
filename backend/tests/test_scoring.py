import pytest
from unittest.mock import patch, MagicMock
from backend.db import SessionLocal
from backend.scoring import rank_jobs, skill_overlap, apply_experience_penalty
from backend.models import Jobs


@pytest.fixture()
def db_session():
    db = SessionLocal()
    yield db
    db.close()


def make_mock_job(title="Test Job", company="Test Co", description="Desc", skills=None, job_id=1, min_years_required=None, seniority_level="any"):
    job = MagicMock()
    job.id = job_id
    job.title = title
    job.company = company
    job.description = description
    job.skills = skills if skills is not None else ["python"]
    job.min_years_required = min_years_required
    job.seniority_level = seniority_level
    return job


# --- skill_overlap ---

def test_skill_overlap():
    assert skill_overlap(["python"], ["python", "sql"]) == 0.5


def test_skill_overlap_full_match():
    assert skill_overlap(["python", "sql"], ["python", "sql"]) == 1.0


def test_skill_overlap_no_match():
    assert skill_overlap(["python"], ["java"]) == 0.0


def test_skill_overlap_no_job_skills():
    assert skill_overlap(["python"], []) == 0


# --- apply_experience_penalty ---

def test_penalty_no_min_years_returns_base():
    # job_min_years=None → no penalty, return base unchanged
    assert apply_experience_penalty(0.8, 3.0, None, "any", "mid") == 0.8


def test_penalty_hard_filter_large_gap():
    # job requires 8yrs, candidate has 3 → gap of 5 > 2 → 10% of base
    result = apply_experience_penalty(0.8, 3.0, 8.0, "senior", "mid")
    assert result == pytest.approx(0.8 * 0.1)


def test_penalty_soft_penalty_within_2_years():
    # job requires 4yrs, candidate has 3 → gap of 1 → penalty = 1 - (1 * 0.10) = 0.90
    result = apply_experience_penalty(0.8, 3.0, 4.0, "mid", "mid")
    assert result == pytest.approx(0.8 * 0.90)


def test_penalty_soft_penalty_floored_at_50_pct():
    # job requires 5yrs, candidate has 3 → gap of 2 → penalty = 1 - (2 * 0.10) = 0.80
    # but gap must be > candidate (5 > 3) and <= candidate + 2 (5 <= 5), so soft path
    result = apply_experience_penalty(0.8, 3.0, 5.0, "senior", "mid")
    assert result == pytest.approx(0.8 * 0.80)


def test_penalty_soft_penalty_max_gap_floor():
    # gap of exactly 2 years: penalty = 1 - (2 * 0.10) = 0.80, max(0.80, 0.5) = 0.80
    result = apply_experience_penalty(1.0, 1.0, 3.0, "any", "entry")
    assert result == pytest.approx(max(1.0 - (2 * 0.10), 0.5))


def test_penalty_seniority_mismatch_more_than_one_level():
    # years satisfied, but entry vs senior = 2 levels apart → 30% of base
    result = apply_experience_penalty(0.8, 5.0, 4.0, "senior", "entry")
    assert result == pytest.approx(0.8 * 0.3)


def test_penalty_seniority_mismatch_one_level_no_penalty():
    # years satisfied, entry vs mid = 1 level → no penalty
    result = apply_experience_penalty(0.8, 3.0, 2.0, "mid", "entry")
    assert result == pytest.approx(0.8)


def test_penalty_years_satisfied_matching_seniority_no_penalty():
    # years satisfied, seniority matches → return base unchanged
    result = apply_experience_penalty(0.75, 5.0, 4.0, "senior", "senior")
    assert result == pytest.approx(0.75)


def test_penalty_unknown_seniority_treated_as_entry():
    # unknown seniority → index 0 (entry), job "senior" = index 2 → gap of 2 → 30% penalty
    result = apply_experience_penalty(0.8, 5.0, 4.0, "senior", "unknown_level")
    assert result == pytest.approx(0.8 * 0.3)


# --- rank_jobs (unit, mocked DB) ---

def test_rank_jobs_returns_results():
    mock_db = MagicMock()
    mock_db.query.return_value.all.return_value = [
        make_mock_job(title="Python Engineer", description="python flask django")
    ]
    result = rank_jobs(mock_db, "python developer flask", ["python"])
    assert len(result) == 1
    assert result[0]["title"] == "Python Engineer"


def test_rank_jobs_sorted_descending():
    mock_db = MagicMock()
    mock_db.query.return_value.all.return_value = [
        make_mock_job(title="Java Job", description="java spring boot", skills=["java"], job_id=1),
        make_mock_job(title="Python Job", description="python django flask", skills=["python"], job_id=2),
    ]
    result = rank_jobs(mock_db, "python developer", ["python"])
    assert result[0]["score"] >= result[1]["score"]
    assert result[0]["title"] == "Python Job"


def test_rank_jobs_score_formula():
    mock_db = MagicMock()
    mock_db.query.return_value.all.return_value = [
        make_mock_job(skills=["python"])
    ]
    with patch("backend.scoring.cosine_similarity", return_value=[[0.8]]), \
         patch("backend.scoring.skill_overlap", return_value=0.5):
        result = rank_jobs(mock_db, "some text", ["python"])
    assert result[0]["score"] == round(0.7 * 0.8 + 0.3 * 0.5, 3)


def test_rank_jobs_returns_top_10():
    mock_db = MagicMock()
    mock_db.query.return_value.all.return_value = [
        make_mock_job(title=f"Job {i}", description=f"description {i}", job_id=i)
        for i in range(15)
    ]
    result = rank_jobs(mock_db, "developer", ["python"])
    assert len(result) == 10


def test_rank_jobs_string_skills_parsed():
    mock_db = MagicMock()
    mock_db.query.return_value.all.return_value = [
        make_mock_job(skills='["python", "sql"]')
    ]
    with patch("backend.scoring.skill_overlap", return_value=1.0) as mock_overlap:
        rank_jobs(mock_db, "some text", ["python"])
        mock_overlap.assert_called_with(["python"], ["python", "sql"])


def test_rank_jobs_invalid_string_skills_fallback():
    mock_db = MagicMock()
    mock_db.query.return_value.all.return_value = [
        make_mock_job(skills="invalid json")
    ]
    with patch("backend.scoring.skill_overlap", return_value=0.0) as mock_overlap:
        rank_jobs(mock_db, "some text", ["python"])
        mock_overlap.assert_called_with(["python"], [])


def test_rank_jobs_exception():
    mock_db = MagicMock()
    mock_db.query.return_value.all.return_value = [make_mock_job()]
    with patch("backend.scoring.cosine_similarity", side_effect=Exception("Sim error")):
        with pytest.raises(ValueError, match="Scoring error - Rank jobs failed: Sim error"):
            rank_jobs(mock_db, "some text", ["python"])


# --- rank_jobs (integration, real DB) ---

def test_rank_jobs_integration(db_session):
    db_session.query(Jobs).delete()
    db_session.commit()

    job1 = Jobs(title="Python Engineer", company="Co A", description="python flask django rest api", skills=["python"])
    job2 = Jobs(title="Java Engineer", company="Co B", description="java spring boot microservices", skills=["java"])
    db_session.add_all([job1, job2])
    db_session.commit()

    result = rank_jobs(db_session, "python developer with flask and django experience", ["python"])
    assert len(result) == 2
    assert result[0]["title"] == "Python Engineer"
