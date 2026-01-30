CREATE TABLE jobs (
  id BIGSERIAL PRIMARY KEY,
  title TEXT NOT NULL,
  company TEXT,
  description TEXT,
  remote BOOLEAN DEFAULT false,
  skills JSONB,
  salary_min NUMERIC,
  salary_max NUMERIC,
  embedding JSONB,
  created_at TIMESTAMP DEFAULT now()
);