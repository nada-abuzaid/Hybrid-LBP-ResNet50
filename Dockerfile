FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Disable oneDNN to avoid CPU-related SIGFPE issues on Cloud Run
ENV TF_ENABLE_ONEDNN_OPTS=0
ENV CUDA_VISIBLE_DEVICES=-1

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

# Verify model exists INSIDE Docker image
RUN echo "===== DOCKER MODEL CHECK =====" \
    && ls -lh /app/model/ \
    && test -f /app/model/paper_fingerprint_hybrid_lbp_resnet50_final.keras \
    && du -h /app/model/paper_fingerprint_hybrid_lbp_resnet50_final.keras \
    && echo "===== MODEL FOUND INSIDE IMAGE ====="

EXPOSE 8080

CMD ["gunicorn", "--bind", "0.0.0.0:8080", "web.app:app", "--workers", "1", "--threads", "1", "--timeout", "300"]