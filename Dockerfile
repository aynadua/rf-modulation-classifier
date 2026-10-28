# CPU-only image. The whole pipeline is designed to run without a GPU.
FROM python:3.11-slim

WORKDIR /app

# Install the CPU build of torch explicitly; the default wheel drags in an
# unused CUDA stack that quadruples the image size.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .

COPY configs ./configs

ENTRYPOINT ["rfmc"]
CMD ["--help"]
