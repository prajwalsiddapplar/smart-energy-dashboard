"""
Smart Energy Backend API
Provides data ingestion, anomaly detection, and energy usage forecasting
for the Smart Building Energy Optimiser Dashboard.
"""
import os
import json
import random
import smtplib
import threading
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import pandas as pd
import azure.functions as func
import numpy as np
from datetime import datetime, timedelta

# Configure standard logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = func.FunctionApp()

# ----------------------------------------------------
# LOAD DATASET
# ----------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "house_power.txt")

df = pd.read_csv(DATA_FILE, sep=",", engine="python", na_values="?")
df.columns = df.columns.str.strip()
df["Datetime"] = pd.to_datetime(df["Datetime"])
df["Global_active_power"] = pd.to_numeric(df["Global_active_power"], errors="coerce")
df = df.dropna()

energy_values = df["Global_active_power"].tolist()
timestamps_raw = df["Datetime"].tolist()

current_index = 30
last_alert_signature = None
last_email_time = None

# Anomaly log (in-memory, survives hot reloads)
anomaly_log: list[dict] = []
auth_cache = {}


# ----------------------------------------------------
# EMA  (Exponential Moving Average) — more accurate,
#       responds faster to recent spikes than simple MA
# ----------------------------------------------------
def compute_ema(values: list[float], span: int) -> list[float | None]:
    """
    Returns EMA series same length as `values`.
    First (span-1) positions are None (warm-up period).
    """
    if len(values) < span:
        return [None] * len(values)

    alpha = 2.0 / (span + 1)
    result: list[float | None] = [None] * (span - 1)

    # Seed with simple average of first `span` values
    seed = float(np.mean(values[:span]))
    result.append(seed)

    for v in values[span:]:
        prev = result[-1]
        result.append(alpha * v + (1.0 - alpha) * prev)

    return result


# ----------------------------------------------------
# ACCURACY METRICS
# ----------------------------------------------------
def compute_accuracy(actual: list[float], predicted: list[float | None]) -> dict:
    pairs = [(a, p) for a, p in zip(actual, predicted) if p is not None]
    if not pairs:
        return {"rmse": 0, "mae": 0, "mape": 0}

    a_arr = np.array([a for a, _ in pairs])
    p_arr = np.array([p for _, p in pairs])

    rmse = float(np.sqrt(np.mean((a_arr - p_arr) ** 2)))
    mae  = float(np.mean(np.abs(a_arr - p_arr)))
    mape = float(np.mean(np.abs((a_arr - p_arr) / (a_arr + 1e-6))) * 100)

    return {
        "rmse": round(rmse, 4),
        "mae":  round(mae, 4),
        "mape": round(mape, 2),
    }


# ----------------------------------------------------
# SEVERITY HELPER
# ----------------------------------------------------
def get_severity(actual: float, predicted: float) -> str:
    ratio = actual / (predicted + 1e-6)
    if ratio > 1.50:
        return "HIGH"
    if ratio > 1.30:
        return "MEDIUM"
    return "LOW"


# ----------------------------------------------------
# EMAIL FUNCTION
# ----------------------------------------------------
def send_email_alert(actual: float, predicted: float, time_label: str, severity: str, dynamic_receiver: str = None):
    sender   = os.environ.get("EMAIL_SENDER")
    password = os.environ.get("EMAIL_PASSWORD")
    receiver = dynamic_receiver or os.environ.get("EMAIL_RECEIVER")

    if not sender or not password or not receiver:
        logger.warning("Email config missing — skipping alert")
        return

    icons = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}
    icon  = icons.get(severity, "⚠")

    msg = MIMEMultipart()
    msg["From"]    = sender
    msg["To"]      = receiver
    msg["Subject"] = f"{icon} [{severity}] Energy Anomaly Detected"

    deviation_pct = round((actual / (predicted + 1e-6) - 1) * 100, 1)

    body = f"""
{icon} Energy Anomaly Alert [{severity}]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Time       : {time_label}
Actual     : {actual:.3f} kW
Predicted  : {predicted:.3f} kW (EMA)
Deviation  : +{deviation_pct}%
Severity   : {severity}

Condition  : Actual > EMA Predicted + 15%

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Smart Building Energy Optimiser
"""
    msg.attach(MIMEText(body, "plain"))

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(sender, password)
        server.sendmail(sender, receiver, msg.as_string())
        server.quit()
        logger.info(f"Alert email sent [{severity}]")
    except Exception as e:
        logger.error(f"Email error: {str(e)}")


# ----------------------------------------------------
# CORS HELPER
# ----------------------------------------------------
def cors(response: func.HttpResponse) -> func.HttpResponse:
    response.headers["Access-Control-Allow-Origin"]  = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response


# ----------------------------------------------------
# AUTHENTICATION
# ----------------------------------------------------
@app.route(route="AuthRequest", auth_level=func.AuthLevel.ANONYMOUS, methods=["POST", "OPTIONS"])
def auth_request(req: func.HttpRequest) -> func.HttpResponse:
    if req.method == "OPTIONS": return cors(func.HttpResponse("", status_code=200))
    try:
        body = req.get_json()
        email = body.get("email")
        if not email:
            return cors(func.HttpResponse(json.dumps({"error": "Email required"}), status_code=400))

        otp = str(random.randint(100000, 999999))
        auth_cache[email] = {
            "otp": otp,
            "expires": datetime.now() + timedelta(minutes=10)
        }

        sender   = os.environ.get("EMAIL_SENDER")
        password = os.environ.get("EMAIL_PASSWORD")

        smtp_ok = False
        if sender and password:
            try:
                msg = MIMEMultipart()
                msg["From"]    = sender
                msg["To"]      = email
                msg["Subject"] = f"EnergyPulse Verification Code: {otp}"
                body_text = (
                    f"Your EnergyPulse access code is: {otp}\n"
                    "This code expires in 10 minutes.\n\nPlease do not share this."
                )
                msg.attach(MIMEText(body_text, "plain"))

                server = smtplib.SMTP("smtp.gmail.com", 587)
                server.starttls()
                server.login(sender, password)
                server.sendmail(sender, email, msg.as_string())
                server.quit()
                smtp_ok = True
                logger.info(f"Auth OTP sent to {email}")
            except Exception as smtp_err:
                logger.warning(f"SMTP failed (dev-mode fallback active): {smtp_err}")

        # ── Dev-mode fallback: return OTP in response when SMTP is unavailable ──
        if smtp_ok:
            return cors(func.HttpResponse(
                json.dumps({"success": True}),
                mimetype="application/json"
            ))
        else:
            logger.info(f"[DEV] OTP for {email}: {otp}")
            return cors(func.HttpResponse(
                json.dumps({"success": True, "dev_otp": otp,
                            "dev_note": "SMTP unavailable — OTP returned for local dev only"}),
                mimetype="application/json"
            ))

    except Exception as e:
        logger.error(f"Auth request error: {e}")
        return cors(func.HttpResponse(
            json.dumps({"error": "Internal server error"}), status_code=500
        ))

@app.route(route="AuthVerify", auth_level=func.AuthLevel.ANONYMOUS, methods=["POST", "OPTIONS"])
def auth_verify(req: func.HttpRequest) -> func.HttpResponse:
    if req.method == "OPTIONS": return cors(func.HttpResponse("", status_code=200))
    try:
        body = req.get_json()
        email, otp = body.get("email"), body.get("otp")
        cached = auth_cache.get(email)
        
        if not cached:
            return cors(func.HttpResponse(json.dumps({"error": "No pending request/expired"}), status_code=400))
        if datetime.now() > cached["expires"]:
             return cors(func.HttpResponse(json.dumps({"error": "OTP expired"}), status_code=400))
        if str(cached["otp"]) != str(otp):
             return cors(func.HttpResponse(json.dumps({"error": "Invalid Validation Code"}), status_code=400))
             
        del auth_cache[email]
        logger.info(f"Auth successful for {email}")
        return cors(func.HttpResponse(json.dumps({"verified": True}), mimetype="application/json"))
    except Exception as e:
        return cors(func.HttpResponse(json.dumps({"error": str(e)}), status_code=500))


# ----------------------------------------------------
# INGEST + CLEAN  →  returns JSON
# ----------------------------------------------------
@app.route(route="IngestAndClean", auth_level=func.AuthLevel.ANONYMOUS)
def ingest(req: func.HttpRequest) -> func.HttpResponse:
    if req.method == "OPTIONS":
        return cors(func.HttpResponse("", status_code=200))

    try:
        raw_df = pd.read_csv(DATA_FILE, sep=",", engine="python", dtype=str)
        raw_df.columns = raw_df.columns.str.strip()

        total_raw    = len(raw_df)
        missing_mask = raw_df["Global_active_power"] == "?"
        missing_rows = raw_df[missing_mask]
        missing_count = int(missing_mask.sum())

        clean_df = pd.read_csv(DATA_FILE, sep=",", engine="python", na_values="?")
        clean_df.columns = clean_df.columns.str.strip()
        clean_df["Global_active_power"] = pd.to_numeric(
            clean_df["Global_active_power"], errors="coerce"
        )
        clean_df["Voltage"]  = pd.to_numeric(clean_df["Voltage"],  errors="coerce")
        clean_df["Datetime"] = pd.to_datetime(clean_df["Datetime"])
        clean_df = clean_df.dropna()

        total_clean   = len(clean_df)
        removed_count = total_raw - total_clean

        before_after = []
        for _, bad_row in missing_rows.head(5).iterrows():
            try:
                ts = pd.to_datetime(bad_row["Datetime"])
                next_valid = clean_df[clean_df["Datetime"] > ts].head(1)
                after = next_valid.iloc[0] if not next_valid.empty else None
            except Exception:
                after = None

            before_after.append({
                "before": {
                    "datetime": bad_row["Datetime"],
                    "power":    bad_row["Global_active_power"],
                    "voltage":  bad_row.get("Voltage", "?"),
                },
                "after": {
                    "datetime": str(after["Datetime"]) if after is not None else None,
                    "power":    round(float(after["Global_active_power"]), 4) if after is not None else None,
                    "voltage":  round(float(after["Voltage"]), 4) if after is not None else None,
                } if after is not None else None,
            })

        raw_numeric = raw_df.replace("?", np.nan)
        missing_by_col = {
            col: int((raw_numeric[col] == "?").sum())
            if raw_numeric[col].dtype == object
            else int(raw_df[col].isna().sum())
            for col in ["Global_active_power", "Voltage", "Sub_metering_1",
                        "Sub_metering_2", "Sub_metering_3"]
            if col in raw_df.columns
        }

        payload = {
            "summary": {
                "total_raw":     total_raw,
                "total_clean":   total_clean,
                "removed":       removed_count,
                "missing_count": missing_count,
                "clean_pct":     round(total_clean / total_raw * 100, 2),
            },
            "missing_by_col": missing_by_col,
            "before_after":   before_after,
        }

        return cors(func.HttpResponse(
            json.dumps(payload), mimetype="application/json"
        ))

    except Exception as e:
        return cors(func.HttpResponse(
            json.dumps({"error": str(e)}), status_code=500, mimetype="application/json"
        ))


# ----------------------------------------------------
# PREDICT USAGE  (EMA-based forecasting — more accurate
#                 than simple moving average; tracks
#                 trends and responds to recent spikes)
# ----------------------------------------------------
@app.route(route="PredictUsage", auth_level=func.AuthLevel.ANONYMOUS)
def predict(req: func.HttpRequest) -> func.HttpResponse:
    if req.method == "OPTIONS":
        return cors(func.HttpResponse("", status_code=200))

    global current_index

    window_size = 24   # Larger window → smoother stock-chart look
    ema_span    = 5    # EMA span (≈ 5-period EMA, sensitive to recent values)
    band_pct    = 0.12 # ±12 % confidence band

    if current_index >= len(energy_values):
        current_index = window_size

    if current_index < window_size:
        current_index += 1
        return cors(func.HttpResponse(
            json.dumps({"actual": [], "predicted": [], "timestamps": [],
                        "upper_band": [], "lower_band": [], "stats": {}}),
            mimetype="application/json"
        ))

    actual = energy_values[current_index - window_size: current_index]
    now    = datetime.now()
    timestamps = [
        (now - timedelta(seconds=(window_size - 1 - i) * 5)).strftime("%H:%M:%S")
        for i in range(window_size)
    ]

    # EMA prediction
    ema_series = compute_ema(actual, ema_span)

    # ±band_pct confidence envelope
    upper_band = [
        round(p * (1 + band_pct), 4) if p is not None else None
        for p in ema_series
    ]
    lower_band = [
        round(p * (1 - band_pct), 4) if p is not None else None
        for p in ema_series
    ]

    # Trend direction (last EMA vs EMA 3 steps back)
    valid_ema = [e for e in ema_series if e is not None]
    if len(valid_ema) >= 4:
        trend = "up" if valid_ema[-1] > valid_ema[-4] else "down"
        trend_pct = round((valid_ema[-1] / (valid_ema[-4] + 1e-6) - 1) * 100, 2)
    else:
        trend = "flat"
        trend_pct = 0.0

    # Accuracy metrics for current window
    accuracy = compute_accuracy(actual, ema_series)

    # Stats for metric cards
    valid_pred = [p for p in ema_series if p is not None]
    avg_actual  = round(float(np.mean(actual)), 4)
    avg_pred    = round(float(np.mean(valid_pred)), 4) if valid_pred else 0
    peak_actual = round(float(np.max(actual)), 4)
    min_actual  = round(float(np.min(actual)), 4)

    current_index += 1

    return cors(func.HttpResponse(
        json.dumps({
            "actual":     [round(v, 4) for v in actual],
            "predicted":  ema_series,
            "timestamps": timestamps,
            "upper_band": upper_band,
            "lower_band": lower_band,
            "stats": {
                "avg_actual":  avg_actual,
                "avg_pred":    avg_pred,
                "peak_actual": peak_actual,
                "min_actual":  min_actual,
                "window":      window_size,
                "trend":       trend,
                "trend_pct":   trend_pct,
                "accuracy":    accuracy,
            },
        }),
        mimetype="application/json"
    ))


# ----------------------------------------------------
# ANOMALY DETECTION
#   Flags ONLY when actual > EMA_predicted × 1.15
#   (i.e., actual is at least 15 % above EMA forecast)
# ----------------------------------------------------
@app.route(route="DetectAnomaly", auth_level=func.AuthLevel.ANONYMOUS)
def detect_anomaly(req: func.HttpRequest) -> func.HttpResponse:
    if req.method == "OPTIONS":
        return cors(func.HttpResponse("", status_code=200))
    receiver_email = req.params.get("receiver_email")

    global current_index, last_alert_signature, last_email_time

    window_size = 24
    ema_span    = 5
    # ── STRICT 15 % threshold ──
    ANOMALY_THRESHOLD = 1.15

    if current_index < window_size:
        return cors(func.HttpResponse(
            json.dumps({"current": [], "log": anomaly_log[-20:]}),
            mimetype="application/json"
        ))

    detect_index = max(current_index - 1, window_size)
    actual = energy_values[detect_index - window_size: detect_index]
    ema_series = compute_ema(actual, ema_span)

    current_anomalies = []

    for i, predicted_val in enumerate(ema_series):
        if predicted_val is None:
            continue

        actual_val = actual[i]

        # ── Only flag if actual is >15 % above EMA prediction ──
        if actual_val > predicted_val * ANOMALY_THRESHOLD:
            sev = get_severity(actual_val, predicted_val)
            deviation_pct = round((actual_val / (predicted_val + 1e-6) - 1) * 100, 1)

            current_anomalies.append({
                "index":         i,
                "actual":        float(round(actual_val, 4)),
                "predicted":     float(round(predicted_val, 4)),
                "deviation_pct": deviation_pct,
                "timestamp":     datetime.now().strftime("%H:%M:%S"),
                "type":          "Energy Spike",
                "severity":      sev,
            })

    # Email alert (with 30-second cooldown, dedup by signature)
    if current_anomalies:
        latest    = current_anomalies[-1]
        signature = f"{latest['index']}-{latest['actual']:.3f}-{latest['predicted']:.3f}"
        now_time  = datetime.now()

        if signature != last_alert_signature:
            cooldown_ok = (
                last_email_time is None
                or (now_time - last_email_time).total_seconds() > 30
            )
            if cooldown_ok:
                threading.Thread(
                    target=send_email_alert,
                    args=(
                        latest["actual"],
                        latest["predicted"],
                        latest["timestamp"],
                        latest["severity"],
                        receiver_email
                    )
                ).start()
                last_alert_signature = signature
                last_email_time      = now_time

        for a in current_anomalies:
            if not anomaly_log or anomaly_log[-1]["timestamp"] != a["timestamp"]:
                anomaly_log.append(a)
        if len(anomaly_log) > 100:
            del anomaly_log[:-100]

    return cors(func.HttpResponse(
        json.dumps({
            "current": current_anomalies,
            "log":     anomaly_log[-20:],
        }),
        mimetype="application/json"
    ))


# ----------------------------------------------------
# ANOMALY LOG  (dedicated endpoint for history tab)
# ----------------------------------------------------
@app.route(route="AnomalyLog", auth_level=func.AuthLevel.ANONYMOUS)
def anomaly_log_endpoint(req: func.HttpRequest) -> func.HttpResponse:
    if req.method == "OPTIONS":
        return cors(func.HttpResponse("", status_code=200))

    return cors(func.HttpResponse(
        json.dumps(anomaly_log[-50:]),
        mimetype="application/json"
    ))