FROM python:3.6-slim-stretch
RUN apt-get update && apt-get install -y procps vim libglib2.0 libsm6 libxext6 libxrender-dev net-tools

COPY . /root
WORKDIR /root

RUN pip install -r requirements.txt


EXPOSE 5000 3000 2222
CMD [ "jupyter" , "notebook", "--allow-root"]

