# Build stage for WebUI
FROM registry.access.redhat.com/ubi9/ubi-minimal:9.5-1730489338 AS webui-builder
SHELL ["/bin/bash", "-c"]

USER root

ENV PATH=/usr/node/bin:$PATH
# Set desired Helm and Kustomize versions
ENV HELM_VERSION="v3.16.2"
ENV KUSTOMIZE_VERSION="5.5.0"

RUN microdnf install -y git wget ca-certificates tar xz \
  # Detect architecture once and set it as an environment variable
  && ARCH=$(uname -m) && \
  case $ARCH in \
  x86_64) NODE_ARCH="x64"; HELM_ARCH="amd64";; \
  aarch64) NODE_ARCH="arm64"; HELM_ARCH="arm64";; \
  armv7l) NODE_ARCH="arm"; HELM_ARCH="arm";; \
  esac && \
  echo "Detected Node.js architecture: $NODE_ARCH" && \
  echo "Detected Helm architecture: $HELM_ARCH" && \
  NODE_URL="https://nodejs.org/dist/v20.18.0/node-v20.18.0-linux-${NODE_ARCH}.tar.xz" && \
  curl -L $NODE_URL -o /tmp/node.tar.xz && \
  mkdir -p /usr/node && \
  tar -xf /tmp/node.tar.xz -C /usr/node --strip-components=1 && \
  npm update -g npm && \
  npm install -g yarn

WORKDIR /webui

# First copy only package dependency files
COPY webui/package.json webui/yarn.lock ./

# Install dependencies
RUN yarn

# Now copy the rest of the webui files
COPY webui/ .

# Build the webui
RUN yarn run build

# Final stage
FROM python:3.12.9-alpine3.21

WORKDIR /app

# Copy pyproject.toml for dependencies
COPY pyproject.toml .

# Install dependencies from pyproject.toml
RUN pip install --no-cache-dir .

# Copy WebUI build from first stage
COPY --from=webui-builder /webui/dist ./webui/dist

# Copy Python source files
COPY ./*.py .
COPY ./logging.conf .
COPY ./inputs.json .

# Default command to run the API server
ENTRYPOINT ["python", "api.py"]
