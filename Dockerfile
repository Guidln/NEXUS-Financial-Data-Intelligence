FROM python:3.11-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY nexus ./nexus
COPY dashboard ./dashboard
COPY README.md .
RUN mkdir -p data/bronze data/silver data/gold artifacts
EXPOSE 8000 8501
CMD ["uvicorn", "nexus.api:app", "--host", "0.0.0.0", "--port", "8000"]
