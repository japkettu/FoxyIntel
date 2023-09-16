FROM python:3.10-slim AS base

RUN apt-get update
RUN apt-get install -y ffmpeg


FROM base as dependencies

WORKDIR /app

RUN apt-get install -y build-essential
COPY requirements.txt .

RUN python3 -m venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --upgrade pip 
RUN pip install -r requirements.txt




FROM base AS runtime

WORKDIR /app

COPY main.py .
COPY style.tcss .
COPY .env .
COPY --from=dependencies /opt/venv /opt/venv
ENV PATH /opt/venv/bin:$PATH

CMD ["python", "./main.py"]



