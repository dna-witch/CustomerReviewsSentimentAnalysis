FROM python:3.9-slim
COPY . ./
RUN pip3 install -r requirements.txt
CMD ["service.py"]
ENTRYPOINT ["python"]