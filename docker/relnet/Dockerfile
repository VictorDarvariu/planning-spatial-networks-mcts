FROM relnet/base
SHELL ["/bin/bash", "-c"]

RUN apt-get update && apt-get install -y build-essential cmake sqlite3 libsqlite3-dev zip unzip \
    libboost-dev libboost-system-dev libboost-thread-dev libboost-serialization-dev \
    libboost-python-dev libboost-regex-dev libboost-numpy-dev

RUN apt-get update && apt-get install -y graphviz libgraphviz-dev graphviz-dev pkg-config

WORKDIR /relnet
ADD ./environment.yml /relnet/environment.yml
RUN conda env update --file environment.yml

CMD ["/bin/bash"]