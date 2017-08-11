FROM python:3.5

# python environment
RUN pip install pip==9.0.1

# dependencies
COPY requirements.txt /tmp/
RUN pip install -r /tmp/requirements.txt

RUN python -c "import nltk; nltk.download('punkt')"
RUN python -c "import nltk; nltk.download('stopwords')"

# install vim
RUN apt-get update
RUN apt-get install -y vim
RUN apt-get install -y psmisc

COPY .vimrc /root

CMD ["/bin/bash"]
