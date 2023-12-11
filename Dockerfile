FROM python:slim
WORKDIR /llm_from_scratch
COPY . .
RUN pip install pdm
RUN pdm install
CMD ["source", "/llm_from_scratch/.venv/bin/activate"]
