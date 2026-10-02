# Training / evaluation image (GPU optional). The bot itself can run here too:
#   docker run --rm -e DISCORD_TOKEN=... -v .:/app triggerbot python -m triggerbot.bot
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir -e ".[dev,bot]"
ENV PYTHONUNBUFFERED=1
CMD ["triggerbot", "--help"]
