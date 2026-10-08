FROM python:3.12-slim
WORKDIR /app/src
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY src/ ./src/
ENTRYPOINT ["python"]
CMD ["evaluate.py"]


    
