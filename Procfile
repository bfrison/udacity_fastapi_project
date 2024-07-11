release: git init; dvc pull
web: gunicorn --chdir starter -k uvicorn.workers.UvicornWorker main:app
