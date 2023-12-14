FROM python:3.9

WORKDIR /vgg19_service

COPY . /vgg19_service/

RUN pip install -r requirements.txt

CMD ["python3","vgg19_service.py"]


EXPOSE 5003

