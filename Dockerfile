FROM python:3.12-slim

# LightGBM needs the OpenMP runtime (libgomp1); tzdata lets the container keep Buffalo time.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 tzdata \
 && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MPLBACKEND=Agg \
    TZ=America/New_York

WORKDIR /app
COPY requirements-lock.txt .
RUN pip install --no-cache-dir -r requirements-lock.txt
COPY wnyload ./wnyload

# data/ and reports/ are mounted from the host (see docker-compose.yml), so downloaded
# data, forecasts and scores survive rebuilds.
CMD ["python", "-m", "wnyload.daily"]
