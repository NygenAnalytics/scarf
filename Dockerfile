FROM ubuntu:26.04

RUN apt update -y && apt autoremove -y && apt clean -y && apt autoclean -y && apt upgrade -y
RUN apt install -y wget build-essential git nano curl

ARG TZ="Europe/Stockholm"
RUN DEBIAN_FRONTEND="noninteractive" TZ=$TZ apt-get -y install tzdata
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /workspace
COPY . /workspace

# t-SNE is optional: add --extra tsne for sgtsnepi, which has wheels for
# linux/amd64 and must otherwise be compiled with a C++ compiler and FFTW.
RUN uv python install 3.14 && uv sync --extra extra
ENV PATH="/workspace/.venv/bin:$PATH"

RUN echo "export PATH=/workspace/.venv/bin:$PATH" >> /root/.bashrc
