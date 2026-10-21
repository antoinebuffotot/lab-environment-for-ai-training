FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# dtctl (Dynatrace CLI), used by the agent as a read-only tool once the participants have configured it.
# The latest release is installed; if the download fails the image still builds (the agent then reports
# "dtctl not installed").
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && (DTCTL_INSTALL_DIR=/usr/local/bin curl -fsSL https://raw.githubusercontent.com/dynatrace-oss/dtctl/main/install.sh | sh \
        || echo "WARNING: dtctl installation failed")

WORKDIR /srv
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app ./app
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
