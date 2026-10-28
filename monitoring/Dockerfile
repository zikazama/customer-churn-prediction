# Prometheus untuk memantau TensorFlow Serving dan gateway API churn.
#
# Build : docker build -t fauzi19-prometheus ./monitoring
# Run   : docker run -p 9090:9090 --network host fauzi19-prometheus
# (target pada prometheus.yml mengarah ke 127.0.0.1, sehingga container dijalankan dengan --network host
#  pada VM yang sama dengan TensorFlow Serving dan gateway)
FROM prom/prometheus
ADD prometheus.yml /etc/prometheus/
