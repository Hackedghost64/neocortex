FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    bash \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -d /home/trials -s /bin/bash trials \
    && git config --system user.name "Project Genesis Sandbox" \
    && git config --system user.email "sandbox@genesis.local" \
    && git config --system init.defaultBranch main

WORKDIR /home/trials
USER trials
RUN git init /home/trials

CMD ["sleep", "infinity"]
