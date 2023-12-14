FROM python:3.9

WORKDIR /svm_service

COPY . /svm_service/

RUN pip install -r requirements.txt

EXPOSE 5002

CMD ["python3","svm_service.py"]
