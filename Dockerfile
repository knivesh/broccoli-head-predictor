FROM python:3.10-slim

WORKDIR /usr/src/app

COPY requirements.txt .
COPY api.py .
COPY utils.py .
COPY templates templates/
COPY models models/

RUN apt-get update
RUN pip install --no-cache -r requirements.txt

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
