FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive `
    PYTHONDONTWRITEBYTECODE=1 `
    PYTHONUNBUFFERED=1 `
    MPLBACKEND=Agg

RUN apt-get update && apt-get install -y --no-install-recommends `
    build-essential git curl ca-certificates `
    libglib2.0-0 libsm6 libxrender1 libxext6 `
    && rm -rf /var/lib/apt/lists/*

RUN useradd -ms /bin/bash appuser
WORKDIR /home/appuser/app
USER appuser

COPY --chown=appuser:appuser requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && `
    pip install --no-cache-dir -r requirements.txt

COPY --chown=appuser:appuser . .

EXPOSE 8501
CMD ["bash", "-lc", "streamlit run app/app.py --server.address=0.0.0.0 --server.port=8501"]
