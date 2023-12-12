FROM python:3.9-slim

RUN pip install pipenv

COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --system --deploy

COPY ["app.py", "./"]
COPY ["model.pkl", "./"]
COPY ["dv.pkl", "./"]

EXPOSE 1616

ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:1616", "app:app"]