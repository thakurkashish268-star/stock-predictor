from typing import TYPE_CHECKING

# Some linters/IDEs may show "Import flask could not be resolved" in environments
# where Flask isn't installed. Use TYPE_CHECKING to keep static type checkers
# happy and provide a lightweight runtime fallback if Flask is unavailable.
if TYPE_CHECKING:
    from flask import Flask, render_template, jsonify, request  # pragma: no cover  # type: ignore
else:
    try:
        from flask import Flask, render_template, jsonify, request
    except Exception:
        # Minimal runtime fallbacks so the module can be imported in environments
        # without Flask (e.g., static analysis or lightweight testing).
        class Flask:  # very small stub
            def __init__(self, *args, **kwargs):
                pass

            def route(self, *args, **kwargs):
                def _decorator(f):
                    return f

                return _decorator

            def run(self, *args, **kwargs):
                return None


        def render_template(name, **context):
            return f"<html>Stub for {name}</html>"


        def jsonify(obj):
            return obj


        class _RequestStub:
            def get_json(self):
                return {}


        request = _RequestStub()
from data import fetch_stock_data, get_company_info
from model import train_and_predict
import traceback

app = Flask(__name__)

# Cache to avoid re-training on every request
_cache = {}


@app.route("/")
def index():
    """Serves the main HTML page"""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    Main API endpoint.
    
    Receives a ticker symbol from the frontend,
    downloads data, trains models, and returns predictions as JSON.
    
    JSON Body: { "ticker": "AAPL", "forecast_days": 30 }
    """
    try:
        body = request.get_json()
        ticker = body.get("ticker", "AAPL").upper().strip()
        forecast_days = int(body.get("forecast_days", 30))

        # Validate ticker
        if not ticker.isalpha() or len(ticker) > 10:
            return jsonify({"error": "Invalid ticker symbol"}), 400

        cache_key = f"{ticker}_{forecast_days}"

        # Return cached result if available (avoid re-training)
        if cache_key in _cache:
            print(f"Cache hit for {ticker}")
            return jsonify(_cache[cache_key])

        print(f"Processing request for {ticker}...")

        # Step 1: Download data
        df = fetch_stock_data(ticker, period="3y")

        # Step 2: Get company info
        company = get_company_info(ticker)

        # Step 3: Train models and get predictions
        results = train_and_predict(df, forecast_days=forecast_days)

        # Step 4: Build response
        response = {
            "ticker": ticker,
            "company": company,
            "results": results,
            "status": "success"
        }

        # Cache the result
        _cache[cache_key] = response

        return jsonify(response)

    except ValueError as e:
        return jsonify({"error": str(e), "status": "error"}), 400
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Prediction failed: {str(e)}", "status": "error"}), 500


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    print("Starting Stock Predictor Web App...")
    app.run(debug=False, host="0.0.0.0", port=5000)