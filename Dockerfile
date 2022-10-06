FROM python:3.10-slim


# Set environment variables
ENV PYTHONUNBUFFERED 1

COPY requirements.txt /
RUN pip install --upgrade pip

#RUN apt-get update -y && apt-get install -y gcc curl gnupg build-essential
#RUN apt-get -y install unixodbc unixodbc-dev tdsodbc



# install FreeTDS and dependencies
RUN apt-get update \
 && apt-get install unixodbc -y \
 && apt-get install unixodbc-dev -y \
 && apt-get install freetds-dev -y \
 && apt-get install freetds-bin -y \
 && apt-get install tdsodbc -y \
 && apt-get install --reinstall build-essential -y




# populate "ocbcinst.ini" as this is where ODBC driver config sits
RUN echo "[FreeTDS]\n\
Description = FreeTDS Driver\n\
Driver = /usr/lib/x86_64-linux-gnu/odbc/libtdsodbc.so\n\
Setup = /usr/lib/x86_64-linux-gnu/odbc/libtdsS.so" >> /etc/odbcinst.ini




# Install dependencies.
RUN pip install -r /requirements.txt --ignore-installed



# Set work directory.
RUN mkdir /code
WORKDIR /code

# Copy project code.
COPY . /code/
EXPOSE 8000

#CMD ["uwsgi", "--http", ":8080", "--ini", "./uwsgi/uwsgi.ini"]
CMD python manage.py runserver 0.0.0.0:8000
