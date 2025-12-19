FROM python:3.10-slim AS build-stage

WORKDIR /usr/src/app

COPY requirements.txt .

RUN python -m venv .venv
RUN .venv/bin/pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pypi.org/simple \
    -r requirements.txt

FROM python:3.10-slim

WORKDIR /usr/src/app

RUN adduser --disabled-password appuser

COPY --from=build-stage --chown=appuser:appuser /usr/src/app/.venv /usr/src/app/.venv

COPY --chown=appuser:appuser api.py utils.py ./
COPY --chown=appuser:appuser templates/ ./templates/
COPY --chown=appuser:appuser models/ ./models/

ENV PATH="/usr/src/app/.venv/bin:$PATH"
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

USER appuser

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
