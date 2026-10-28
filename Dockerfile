# Deployment model churn dengan TensorFlow Serving (lengkap dengan konfigurasi monitoring Prometheus).
#
# Build : docker build -t fauzi19-churn-serving .
# Run   : docker run -p 8501:8501 fauzi19-churn-serving
# Cek   : http://localhost:8501/v1/models/fauzi19-churn-model/metadata
#         http://localhost:8501/monitoring/prometheus/metrics
#
# Folder serving_model berisi hasil komponen Pusher: serving_model/<versi>/saved_model.pb
FROM tensorflow/serving:latest

COPY ./serving_model /models/fauzi19-churn-model
COPY ./monitoring /model_config

ENV MODEL_NAME=fauzi19-churn-model
ENV MODEL_BASE_PATH=/models
ENV MONITORING_CONFIG="/model_config/prometheus.config"
ENV PORT=8501

RUN echo '#!/bin/bash 

tensorflow_model_server --port=8500 --rest_api_port=${PORT} --model_name=${MODEL_NAME} --model_base_path=${MODEL_BASE_PATH}/${MODEL_NAME} --monitoring_config_file=${MONITORING_CONFIG} "$@"' > /usr/bin/tf_serving_entrypoint.sh && chmod +x /usr/bin/tf_serving_entrypoint.sh

EXPOSE 8500 8501
