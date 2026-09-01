"""
Streamlit UI for the Multi-Series Forecasting API.

Talks to the FastAPI service (see api/main.py) over HTTP - it never touches
the models directly. Configure the backend location with the API_URL env
var (defaults to http://localhost:8000; docker-compose sets it to
http://api:8000 so the UI container can reach the API container by name).

Usage:
    streamlit run ui/app.py
"""
import os
import io

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")
REQUEST_TIMEOUT = 120  # seconds - foundation model can be slow on first call

st.set_page_config(page_title="Multi-Series Forecasting", page_icon="📈", layout="wide")


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────
def api_get(path: str, **kwargs):
    resp = requests.get(f"{API_URL}{path}", timeout=REQUEST_TIMEOUT, **kwargs)
    resp.raise_for_status()
    return resp.json()


def api_post(path: str, **kwargs):
    resp = requests.post(f"{API_URL}{path}", timeout=REQUEST_TIMEOUT, **kwargs)
    if not resp.ok:
        try:
            detail = resp.json().get("detail", resp.text)
        except Exception:
            detail = resp.text
        raise RuntimeError(f"{resp.status_code}: {detail}")
    return resp.json()


@st.cache_data(show_spinner=False)
def check_health():
    try:
        return api_get("/health"), None
    except Exception as e:
        return None, str(e)


def plot_forecast(history: pd.DataFrame, forecast: list, interval: int):
    fig = go.Figure()

    hist_tail = history.tail(200)
    fig.add_trace(go.Scatter(
        x=hist_tail["timestamp"], y=hist_tail["target"],
        mode="lines", name="History", line=dict(color="#4C78A8"),
    ))

    fdf = pd.DataFrame(forecast)
    fdf["timestamp"] = pd.to_datetime(fdf["timestamp"])

    lower_col, upper_col = (f"lower_{interval}", f"upper_{interval}")
    if lower_col in fdf.columns:
        fig.add_trace(go.Scatter(
            x=pd.concat([fdf["timestamp"], fdf["timestamp"][::-1]]),
            y=pd.concat([fdf[upper_col], fdf[lower_col][::-1]]),
            fill="toself", fillcolor="rgba(244,124,53,0.18)",
            line=dict(color="rgba(255,255,255,0)"),
            name=f"{interval}% interval", hoverinfo="skip",
        ))

    fig.add_trace(go.Scatter(
        x=fdf["timestamp"], y=fdf["point"],
        mode="lines+markers", name="Forecast (median)",
        line=dict(color="#F47C35"),
    ))

    fig.update_layout(
        height=480,
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_title="Timestamp", yaxis_title="Target",
    )
    return fig


# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────
with st.sidebar:
    st.title("📈 Forecasting")
    st.caption(f"API: `{API_URL}`")

    health, health_err = check_health()
    if st.button("Refresh connection"):
        check_health.clear()
        st.rerun()

    if health_err:
        st.error(f"API unreachable: {health_err}")
    else:
        st.success("API connected")
        st.write(f"LightGBM (exog): {'✅' if health['exog_loaded'] else '❌ not trained'}")
        st.write(f"LightGBM (no-exog): {'✅' if health['noexog_loaded'] else '❌ not trained'}")

    st.divider()
    st.subheader("Model")
    model_choice = st.radio(
        "Which model?",
        ["LightGBM (global)", "Chronos-2 (foundation)"],
        help="LightGBM is trained on your data with optional covariates. "
             "Chronos-2 is a pretrained transformer that needs no training "
             "and works zero-shot on raw target values only.",
    )
    use_exog = True
    if model_choice == "LightGBM (global)":
        use_exog = st.toggle("Use covariates (exog)", value=True)

    max_horizon = 168 if model_choice.startswith("Chronos") else 720
    horizon = st.slider("Horizon (hours ahead)", 1, max_horizon, 24)
    interval = st.radio("Prediction interval", [90, 50], horizontal=True)

tab_forecast, tab_models, tab_evaluate, tab_train = st.tabs(
    ["Forecast", "Model Info", "Evaluate", "Train"]
)


# ──────────────────────────────────────────────
# Forecast tab
# ──────────────────────────────────────────────
with tab_forecast:
    st.subheader("Upload history & forecast")
    st.caption(
        "CSV/TSV with columns: timestamp, series_id, target "
        "(plus cov_1..5, cov_cat if using covariates). At least 50 rows of history."
    )
    uploaded = st.file_uploader("History file", type=["csv", "tsv"])

    if uploaded is not None:
        raw_bytes = uploaded.getvalue()
        try:
            preview = pd.read_csv(io.BytesIO(raw_bytes), sep=None, engine="python")
            preview["timestamp"] = pd.to_datetime(preview["timestamp"])
            preview = preview.sort_values(["series_id", "timestamp"])
            st.dataframe(preview.tail(10), use_container_width=True)
        except Exception as e:
            preview = None
            st.warning(f"Couldn't preview file locally (will still try to send it): {e}")

        run = st.button("Run forecast", type="primary")
        if run:
            endpoint = "/forecast/global" if model_choice == "LightGBM (global)" else "/forecast/foundation"
            data = {"horizon": horizon, "interval": interval}
            if model_choice == "LightGBM (global)":
                data["use_exog"] = use_exog

            files = {"file": (uploaded.name, raw_bytes, "text/csv")}
            with st.spinner(f"Forecasting via {endpoint} ..."):
                try:
                    result = api_post(endpoint, files=files, data=data)
                except Exception as e:
                    st.error(f"Forecast failed: {e}")
                    result = None

            if result:
                st.session_state["last_result"] = result
                st.session_state["last_history"] = preview

    result = st.session_state.get("last_result")
    history = st.session_state.get("last_history")
    if result:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Model", result["model"])
        c2.metric("Series", result["series_id"])
        c3.metric("Horizon", f"{result['horizon']}h")
        c4.metric("Covariates", "on" if result.get("use_exog") else "off")

        if history is not None:
            st.plotly_chart(
                plot_forecast(history, result["forecast"], result["interval"]),
                use_container_width=True,
            )

        fdf = pd.DataFrame(result["forecast"])
        st.dataframe(fdf, use_container_width=True)
        st.download_button(
            "Download forecast CSV",
            fdf.to_csv(index=False).encode("utf-8"),
            file_name=f"forecast_{result['model']}_{result['series_id']}.csv",
            mime="text/csv",
        )


# ──────────────────────────────────────────────
# Model info tab
# ──────────────────────────────────────────────
with tab_models:
    st.subheader("Loaded models & last training results")
    if st.button("Refresh model info"):
        st.cache_data.clear()
    try:
        info = api_get("/models")
        st.write("**Loaded variants:**", info["variants"] or "none")
        st.write("**Quantiles:**", info["quantiles"])
        if info["results"]:
            st.json(info["results"])
        else:
            st.info("No results.json yet - train the models first.")
    except Exception as e:
        st.error(f"Could not fetch model info: {e}")


# ──────────────────────────────────────────────
# Evaluate tab
# ──────────────────────────────────────────────
with tab_evaluate:
    st.subheader("Evaluate LightGBM on the saved test set")
    eval_exog = st.toggle("Evaluate exog variant", value=True, key="eval_exog")
    if st.button("Run evaluation"):
        with st.spinner("Evaluating..."):
            try:
                eval_result = api_post("/evaluate/global", data={"use_exog": eval_exog})
                st.json(eval_result)
            except Exception as e:
                st.error(f"Evaluation failed: {e}")


# ──────────────────────────────────────────────
# Train tab
# ──────────────────────────────────────────────
with tab_train:
    st.subheader("Retrain LightGBM models")
    st.warning(
        "Retrains both exog/no-exog LightGBM variants from data/raw/ on the "
        "API server and overwrites the current models. This can take a while "
        "and cannot be undone."
    )
    confirm = st.checkbox("I understand this overwrites the current models")
    if st.button("Start training", disabled=not confirm, type="primary"):
        with st.spinner("Training - this can take a few minutes..."):
            try:
                train_result = api_post("/train")
                st.success("Training complete")
                st.json(train_result)
                check_health.clear()
            except Exception as e:
                st.error(f"Training failed: {e}")
