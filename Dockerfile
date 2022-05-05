FROM ubuntu:20.04

RUN apt update
RUN apt install -y python3 python3-pip
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 0

ENV DEBIAN_FRONTEND=noninteractive
RUN apt install -y curl git  python-dev libopenblas-dev python3-setuptools graphviz
RUN curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python3 -

ENV APP_PATH=/home
WORKDIR ${APP_PATH}

ENV POETRY_HOME /root/.poetry/
ENV PATH /root/.poetry/bin:$PATH

WORKDIR ${APP_PATH}

COPY pyproject.toml poetry.lock /home/
RUN poetry config virtualenvs.create false
RUN poetry install --no-dev --no-interaction

ENV HOME=${APP_PATH}
ENV USERNAME=user
ENV PYTHONPATH=${APP_PATH}
