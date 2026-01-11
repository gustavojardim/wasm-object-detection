# Minimal Dockerfile for Python Flask latency probe
FROM python:3.10-slim
WORKDIR /app
COPY latency_probe.py .
RUN pip install --no-cache-dir flask
EXPOSE 8080
CMD ["python", "latency_probe.py"]