FROM ubuntu:latest

ADD main.py .
ADD style.tcss .
ADD .env .
ADD requirements.txt .

RUN apt update
RUN apt install build-essential python3-pip -y
RUN pip install --upgrade pip 
RUN pip install -r requirements.txt

CMD ["python3", "./main.py"]



