# Minimal Dockerfile for orchestrator service
FROM python:3.10-slim

RUN apt-get update && apt-get install -y libxcb1

WORKDIR /app

COPY kubernetes/orchestration/orchestrator/orchestrator.py .
COPY kubernetes/orchestration/orchestrator/image.png /app/samples/image.png

RUN pip install --no-cache-dir flask kubernetes opencv-python-headless numpy

EXPOSE 5000

CMD ["python", "orchestrator.py"]
