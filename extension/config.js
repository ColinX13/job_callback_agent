// Toggle to true when deploying to production
const IS_PRODUCTION = true;

const CONFIG = {
    BACKEND_URL: IS_PRODUCTION
        ? "https://job-matcher-agent.onrender.com"
        : "http://localhost:8000",
};
