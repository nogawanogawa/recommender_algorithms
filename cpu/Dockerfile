FROM python:3.8.15-buster

RUN apt update
RUN apt install -y curl

ARG work_dir="/home/"
WORKDIR /home

# poetryのインストール先の指定
ENV POETRY_HOME /root/.poetry/
ENV PATH /root/.poetry/bin:$PATH

COPY pyproject.toml poetry.lock /home/

# poetry
RUN curl -sSL https://install.python-poetry.org | python - && \
    ln -s /root/.poetry/bin/poetry && \
    poetry config virtualenvs.create false

RUN poetry install