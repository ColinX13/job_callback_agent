from sqlalchemy import Column, Integer, String, Boolean, JSON, Float, DateTime
from sqlalchemy.sql import func
from db import Base

class Jobs(Base):
    __tablename__ = "jobs"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, index=True)
    company = Column(String, index=True)
    description = Column(String)
    remote = Column(Boolean, default=False)
    skills = Column(JSON)
    salary_min = Column(Float)
    salary_max = Column(Float)
    embedding = Column(JSON)
    created_at = Column(DateTime(timezone=True), server_default=func.now())