#!/bin/bash
# Pemasangan TF Serving + gateway + Prometheus pada VM (dijalankan sekali, idempoten).
set -e
sudo useradd --system --home /opt/fauzi19 --shell /usr/sbin/nologin fauzi19 2>/dev/null || true
sudo mkdir -p /opt/fauzi19/serving_model /opt/fauzi19/prom-data
if ! command -v tensorflow_model_server >/dev/null; then
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y tensorflow-model-server python3-venv
fi
if [ ! -x /opt/fauzi19/venv/bin/gunicorn ]; then
  sudo python3 -m venv /opt/fauzi19/venv
  sudo /opt/fauzi19/venv/bin/pip install -q -r /tmp/fauzi19-up/requirements-gateway.txt
fi
if [ ! -x /opt/fauzi19/prometheus/prometheus ]; then
  cd /tmp
  curl -fsSL -o prom.tgz https://github.com/prometheus/prometheus/releases/download/v3.5.0/prometheus-3.5.0.linux-amd64.tar.gz
  tar xzf prom.tgz
  sudo rm -rf /opt/fauzi19/prometheus
  sudo mv prometheus-3.5.0.linux-amd64 /opt/fauzi19/prometheus
fi
sudo cp /tmp/fauzi19-up/gateway.py /opt/fauzi19/
sudo mkdir -p /opt/fauzi19/monitoring
sudo cp /tmp/fauzi19-up/monitoring/prometheus.yml /tmp/fauzi19-up/monitoring/prometheus.config /opt/fauzi19/monitoring/
sudo rm -rf /opt/fauzi19/serving_model && sudo cp -r /tmp/fauzi19-up/serving_model /opt/fauzi19/serving_model
sudo cp /tmp/fauzi19-up/*.service /etc/systemd/system/
sudo chown -R fauzi19:fauzi19 /opt/fauzi19
sudo systemctl daemon-reload
sudo systemctl enable --now fauzi19-tfserving fauzi19-gateway fauzi19-prometheus
echo SETUP-DONE
