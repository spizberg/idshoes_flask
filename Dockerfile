FROM pytorch/pytorch
RUN mkdir /workspace/app
WORKDIR /workspace/app
RUN apt-get update && apt-get install -y python3-opencv gcc
EXPOSE 8000
COPY . .
RUN pip install -r requirements.txt
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "app:app", "--reload"]