"""Gateway REST untuk model churn (TensorFlow Serving) dengan metrik Prometheus.

Menerima fitur pelanggan dalam JSON, membentuk tf.Example (serialisasi protobuf manual agar
tidak perlu memasang TensorFlow), meneruskannya ke TensorFlow Serving, lalu mengembalikan
probabilitas churn. Metrik diekspor pada /metrics untuk di-scrape Prometheus.
"""

import base64
import os
import struct
import time

import requests
from flask import Flask, Response, jsonify, request
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

TF_SERVING_URL = os.environ.get("TF_SERVING_URL", "http://127.0.0.1:8501/v1/models/fauzi19-churn-model")
THRESHOLD = 0.5

INT_FEATURES = ["SeniorCitizen", "tenure"]
FLOAT_FEATURES = ["MonthlyCharges", "TotalCharges"]
CATEGORICAL = {
    "gender": ["Female", "Male"],
    "Partner": ["Yes", "No"],
    "Dependents": ["Yes", "No"],
    "PhoneService": ["Yes", "No"],
    "MultipleLines": ["No phone service", "No", "Yes"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["No internet service", "No", "Yes"],
    "OnlineBackup": ["No internet service", "No", "Yes"],
    "DeviceProtection": ["No internet service", "No", "Yes"],
    "TechSupport": ["No internet service", "No", "Yes"],
    "StreamingTV": ["No internet service", "No", "Yes"],
    "StreamingMovies": ["No internet service", "No", "Yes"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["Yes", "No"],
    "PaymentMethod": [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ],
}
FEATURES = list(CATEGORICAL) + INT_FEATURES + FLOAT_FEATURES

REQUESTS = Counter("churn_http_requests_total", "Jumlah request HTTP", ["endpoint", "method", "status"])
LATENCY = Histogram(
    "churn_request_latency_seconds",
    "Latensi request HTTP",
    ["endpoint"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5),
)
UPSTREAM_LATENCY = Histogram(
    "churn_tfserving_latency_seconds",
    "Latensi pemanggilan TensorFlow Serving",
    buckets=(0.002, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5),
)
PREDICTIONS = Counter("churn_predictions_total", "Jumlah prediksi per kelas", ["label"])
PROBABILITY = Histogram(
    "churn_prediction_probability",
    "Sebaran probabilitas churn",
    buckets=(0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95),
)
ERRORS = Counter("churn_errors_total", "Jumlah error", ["type"])
IN_PROGRESS = Gauge("churn_requests_in_progress", "Request yang sedang diproses")
MODEL_UP = Gauge("churn_model_available", "1 jika TensorFlow Serving melaporkan model AVAILABLE")
MODEL_VERSION = Gauge("churn_model_version", "Versi model yang aktif di TensorFlow Serving")

app = Flask(__name__)


def _varint(value):
    out = bytearray()
    value &= (1 << 64) - 1
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def _len_delimited(field, payload):
    return _varint((field << 3) | 2) + _varint(len(payload)) + payload


def _feature(key, value):
    if key in INT_FEATURES:
        inner = _len_delimited(1, _varint(int(value)))
        return _len_delimited(3, inner)
    if key in FLOAT_FEATURES:
        inner = _len_delimited(1, struct.pack("<f", float(value)))
        return _len_delimited(2, inner)
    inner = _len_delimited(1, str(value).encode("utf-8"))
    return _len_delimited(1, inner)


def to_example(row):
    """Serialisasi satu pelanggan menjadi tf.train.Example."""
    entries = b""
    for key in FEATURES:
        entry = _len_delimited(1, key.encode()) + _len_delimited(2, _feature(key, row[key]))
        entries += _len_delimited(1, entry)
    return _len_delimited(1, entries)


def validate(row):
    if not isinstance(row, dict):
        raise ValueError("setiap pelanggan harus berupa objek JSON")
    missing = [key for key in FEATURES if key not in row]
    if missing:
        raise ValueError("fitur belum lengkap: " + ", ".join(missing))
    for key, allowed in CATEGORICAL.items():
        if row[key] not in allowed:
            raise ValueError(f"nilai {key} harus salah satu dari {allowed}")
    for key in INT_FEATURES + FLOAT_FEATURES:
        try:
            number = float(row[key])
        except (TypeError, ValueError):
            raise ValueError(f"{key} harus berupa angka")
        if number < 0:
            raise ValueError(f"{key} tidak boleh negatif")


def predict_rows(rows):
    for row in rows:
        validate(row)
    instances = [{"b64": base64.b64encode(to_example(row)).decode()} for row in rows]
    payload = {"signature_name": "serving_default", "instances": instances}
    started = time.perf_counter()
    try:
        response = requests.post(TF_SERVING_URL + ":predict", json=payload, timeout=30)
        response.raise_for_status()
    except requests.RequestException:
        ERRORS.labels("tf_serving").inc()
        raise
    finally:
        UPSTREAM_LATENCY.observe(time.perf_counter() - started)
    results = []
    for item in response.json()["predictions"]:
        probability = float(item[0])
        label = "churn" if probability >= THRESHOLD else "tidak_churn"
        PREDICTIONS.labels(label).inc()
        PROBABILITY.observe(probability)
        results.append({"probability": round(probability, 4), "label": label})
    return results


def refresh_model_status():
    try:
        info = requests.get(TF_SERVING_URL, timeout=3).json()
        status = info["model_version_status"][0]
        available = status["state"] == "AVAILABLE"
        MODEL_UP.set(1 if available else 0)
        MODEL_VERSION.set(int(status["version"]))
        return available
    except Exception:
        MODEL_UP.set(0)
        return False


@app.before_request
def _start():
    request._started = time.perf_counter()
    IN_PROGRESS.inc()


@app.after_request
def _finish(response):
    IN_PROGRESS.dec()
    known = ("/", "/predict", "/health", "/metrics", "/schema")
    endpoint = request.path if request.path in known else "other"
    if request.path.startswith("/v1/models/"):
        endpoint = "/v1/models"
    elif request.path == "/monitoring/prometheus/metrics":
        endpoint = request.path
    REQUESTS.labels(endpoint, request.method, str(response.status_code)).inc()
    LATENCY.labels(endpoint).observe(time.perf_counter() - request._started)
    return response


@app.get("/health")
def health():
    ok = refresh_model_status()
    return jsonify({"status": "ok" if ok else "model_unavailable"}), (200 if ok else 503)


@app.get("/metrics")
def metrics():
    refresh_model_status()
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)


TF_SERVING_ROOT = TF_SERVING_URL.rsplit("/v1/models/", 1)[0]


@app.get("/monitoring/prometheus/metrics")
def tf_serving_metrics():
    """Meneruskan metrik Prometheus bawaan TensorFlow Serving (tensorflow:serving:request_count, dst.)."""
    try:
        upstream = requests.get(f"{TF_SERVING_ROOT}/monitoring/prometheus/metrics", timeout=10)
    except requests.RequestException:
        ERRORS.labels("tf_serving").inc()
        return jsonify({"error": "TensorFlow Serving tidak merespons"}), 502
    return Response(upstream.content, status=upstream.status_code, mimetype=CONTENT_TYPE_LATEST)


@app.get("/v1/models/<path:rest>")
def tf_serving_status(rest):
    """Meneruskan endpoint status dan metadata REST TensorFlow Serving (hanya baca)."""
    try:
        upstream = requests.get(f"{TF_SERVING_ROOT}/v1/models/{rest}", timeout=10)
    except requests.RequestException:
        ERRORS.labels("tf_serving").inc()
        return jsonify({"error": "TensorFlow Serving tidak merespons"}), 502
    return Response(upstream.content, status=upstream.status_code, mimetype="application/json")


@app.post("/v1/models/<path:rest>")
def tf_serving_predict(rest):
    """Meneruskan request prediksi REST TensorFlow Serving apa adanya."""
    try:
        upstream = requests.post(
            f"{TF_SERVING_ROOT}/v1/models/{rest}", data=request.get_data(),
            headers={"Content-Type": "application/json"}, timeout=30,
        )
    except requests.RequestException:
        ERRORS.labels("tf_serving").inc()
        return jsonify({"error": "TensorFlow Serving tidak merespons"}), 502
    return Response(upstream.content, status=upstream.status_code, mimetype="application/json")


@app.get("/schema")
def schema():
    return jsonify({"features": FEATURES, "categorical": CATEGORICAL, "numeric": INT_FEATURES + FLOAT_FEATURES})


@app.post("/predict")
def predict():
    body = request.get_json(silent=True)
    if body is None:
        ERRORS.labels("bad_request").inc()
        return jsonify({"error": "body harus JSON"}), 400
    rows = body if isinstance(body, list) else [body]
    if not 1 <= len(rows) <= 500:
        ERRORS.labels("bad_request").inc()
        return jsonify({"error": "kirim 1 sampai 500 pelanggan"}), 400
    try:
        results = predict_rows(rows)
    except ValueError as error:
        ERRORS.labels("validation").inc()
        return jsonify({"error": str(error)}), 422
    except requests.RequestException:
        return jsonify({"error": "TensorFlow Serving tidak merespons"}), 502
    return jsonify({"predictions": results} if isinstance(body, list) else results[0])


PAGE = """<!doctype html><html lang="id"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Prediksi Customer Churn</title>
<style>body{font-family:system-ui,sans-serif;max-width:760px;margin:2rem auto;padding:0 1rem;color:#1a2433}
h1{font-size:1.4rem}form{display:grid;grid-template-columns:1fr 1fr;gap:.6rem 1rem}label{font-size:.85rem;display:flex;flex-direction:column;gap:.2rem}
select,input{padding:.4rem;border:1px solid #b8c2d0;border-radius:6px}button{grid-column:1/-1;padding:.7rem;border:0;border-radius:8px;background:#1f6feb;color:#fff;font-size:1rem;cursor:pointer}
#out{margin-top:1rem;padding:1rem;border-radius:8px;background:#eef3fa;font-size:1.05rem}small{color:#5b6779}</style></head><body>
<h1>Prediksi Customer Churn (TFX + TensorFlow Serving)</h1>
<p><small>Model pipeline TFX milik fauzi19. API: <code>POST /predict</code>, metrik: <code>/metrics</code>, skema: <code>/schema</code>.</small></p>
<form id="f"></form><div id="out">Isi data pelanggan lalu tekan Prediksi.</div>
<script>
const DEFAULT={gender:"Female",SeniorCitizen:0,Partner:"No",Dependents:"No",tenure:2,PhoneService:"Yes",MultipleLines:"No",InternetService:"Fiber optic",OnlineSecurity:"No",OnlineBackup:"No",DeviceProtection:"No",TechSupport:"No",StreamingTV:"No",StreamingMovies:"No",Contract:"Month-to-month",PaperlessBilling:"Yes",PaymentMethod:"Electronic check",MonthlyCharges:80.35,TotalCharges:160.7};
fetch('/schema').then(r=>r.json()).then(s=>{const f=document.getElementById('f');
for(const k of s.features){const l=document.createElement('label');l.textContent=k;let e;
if(s.categorical[k]){e=document.createElement('select');for(const v of s.categorical[k]){const o=document.createElement('option');o.value=o.textContent=v;e.appendChild(o)}}
else{e=document.createElement('input');e.type='number';e.step='any';e.min=0}
e.name=k;e.value=DEFAULT[k];l.appendChild(e);f.appendChild(l)}
const b=document.createElement('button');b.type='submit';b.textContent='Prediksi';f.appendChild(b);
f.onsubmit=async ev=>{ev.preventDefault();const d={};for(const k of s.features){const v=f.elements[k].value;d[k]=s.numeric.includes(k)?Number(v):v}
const r=await fetch('/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(d)});const j=await r.json();
document.getElementById('out').textContent=r.ok?('Probabilitas churn: '+(j.probability*100).toFixed(2)+'% , prediksi: '+(j.label==='churn'?'CHURN':'Tidak churn')):('Error: '+j.error)}});
</script></body></html>"""


@app.get("/")
def index():
    return Response(PAGE, mimetype="text/html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
