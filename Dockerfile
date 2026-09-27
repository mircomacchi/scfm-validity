FROM python:3.12-slim

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src

# CPU-only torch keeps the image small; scVI runs on CPU, Geneformer needs the [fm] extra.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch \
 && pip install ".[scvi,api]"

EXPOSE 8000
ENTRYPOINT ["scfm-validity"]
# API: docker run -p 8000:8000 scfm-validity serve --host 0.0.0.0
