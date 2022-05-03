FROM ubuntu:20.04

RUN apt update
RUN apt install -y python3 python3-pip

RUN apt install -y curl
RUN curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python3 -

ENV APP_PATH=/home
WORKDIR ${APP_PATH}

ENV POETRY_HOME /root/.poetry/
ENV PATH /root/.poetry/bin:$PATH

COPY pyproject.toml poetry.lock /home/
RUN poetry config virtualenvs.create false
RUN poetry install --no-dev --no-interaction


ENV HOME=${APP_PATH}
ENV USERNAME=user
ENV PYTHONPATH=${APP_PATH}
