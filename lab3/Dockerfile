FROM python:3.9

COPY requirements.txt /tmp/

RUN pip3 install --requirement /tmp/requirements.txt

COPY . /app

WORKDIR /app

RUN python3 /app/whisper_api.py

EXPOSE 8501
