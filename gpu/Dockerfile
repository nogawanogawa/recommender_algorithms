FROM nvidia/cuda:11.6.0-base-ubuntu20.04

RUN apt update -y
RUN apt install -y python3 \
    python3-pip \
    curl

RUN ln -s /usr/bin/python3 /usr/bin/python

ENV APP_PATH=/home
ENV HOME=${APP_PATH}
ENV USERNAME=user
ENV PYTHONPATH=${APP_PATH}

WORKDIR ${APP_PATH}

# poetryのインストール先の指定
ENV POETRY_HOME=/opt/poetry

COPY ../pyproject.toml /home/

# poetry
RUN curl -sSL https://install.python-poetry.org | python3 - && \
    cd /usr/local/bin && \
    ln -s /opt/poetry/bin/poetry && \
    poetry config virtualenvs.create false

RUN poetry install
