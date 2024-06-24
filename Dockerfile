FROM python:3.10.11-buster

COPY requirements.txt ./requirements.txt
COPY VideoSummarizer.py ./VideoSummarizer.py
RUN pip install -r requirements.txt

COPY app.py .

# Expose port 8080
EXPOSE 8080

CMD flask run --host=0.0.0.0 --port=8080