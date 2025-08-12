# This is installing the pgvector extension for postgres
FROM postgres:16.3

RUN apt-get update && apt-get install -y \
  build-essential \
  git \
  postgresql-server-dev-all \
  python3.11 \
  python3-pip -y \
  postgresql-plpython3-16 \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /tmp
RUN git clone https://github.com/pgvector/pgvector.git


WORKDIR /tmp/pgvector
RUN make
RUN make install


