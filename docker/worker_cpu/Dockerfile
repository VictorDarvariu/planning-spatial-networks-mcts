FROM relnet/relnet
ARG RN_GID
ARG RN_GNAME

RUN if ! [[ $(getent group ${RN_GNAME}) ]]; \
    then groupadd -g ${RN_GID} ${RN_GNAME}; \
    fi

RUN if ! [[ $(getent passwd relnetworker) ]]; \
    then useradd -r -u 20627 --create-home --home-dir /home/relnetworker -g ${RN_GNAME} relnetworker; \
    fi

RUN chgrp -R ${RN_GNAME} /relnet
RUN chmod -R g+rwx /relnet

RUN source activate relnet-cenv \
    && pip install geopy==2.0.0 \
    && pip install pyproj==2.6.1 \
    && pip install pygsp==0.5.1 \
    && pip install texttable==1.5.0

USER relnetworker

ENV PATH /opt/conda/envs/relnet-cenv/bin:$PATH
ENV CPLUS_INCLUDE_PATH $CPLUS_INCLUDE_PATH:/usr/include/python3.6/
ENV RELNET_DEVICE_PLACEMENT CPU

CMD ["bash", "/relnet/docker/worker_cpu/start_worker.sh"]