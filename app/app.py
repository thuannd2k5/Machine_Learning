import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path


st.set_page_config(page_title="Dự đoán số lượng xe đạp thuê", layout="wide")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "hour.csv"
MODEL_DIR = BASE_DIR / "models"

MODEL_PATHS = {
    "Linear Regression": MODEL_DIR / "linear_regression_model.pkl",
    "Random Forest": MODEL_DIR / "random_forest_model.pkl",
    "XGBoost": MODEL_DIR / "xgboost_model.pkl",
}

METRIC_DF = pd.DataFrame(
    {
        "Model": ["Linear Regression", "Random Forest", "XGBoost"],
        "MAE": [78.04, 49.64, 51.98],
        "RMSE": [109.22, 74.98, 75.32],
        "R2": [0.623, 0.822, 0.821],
    }
)


@st.cache_data(show_spinner=False)
def load_data():
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)
    return None


@st.cache_resource(show_spinner=False)
def load_models():
    models = {}
    for model_name, model_path in MODEL_PATHS.items():
        if model_path.exists():
            models[model_name] = joblib.load(model_path)
        else:
            models[model_name] = None
    return models


def model_feature_names(model):
    if model is not None and hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    return []


def build_model_input(raw_input, model):
    """
    Người dùng nhập giá trị thực tế; hàm này tự convert về schema train của model.
    """
    columns = model_feature_names(model)
    row = {c: 0.0 for c in columns}

    hr = int(raw_input["hr"])
    workingday = int(raw_input["workingday"])
    season = int(raw_input["season"])
    weathersit = int(raw_input["weathersit"])
    temp_c = float(raw_input["temp_combined"])
    hum_pct = float(raw_input["hum"])
    wind_kmh = float(raw_input["windspeed"])

    # Convert về scale dataset gốc (0..1)
    temp_norm = float(np.clip(temp_c / 41.0, 0.0, 1.0))
    atemp_norm = float(np.clip(temp_c / 50.0, 0.0, 1.0))
    hum_norm = float(np.clip(hum_pct / 100.0, 0.0, 1.0))
    wind_norm = float(np.clip(wind_kmh / 67.0, 0.0, 1.0))

    # Numeric features
    if "temp" in row:
        row["temp"] = temp_norm
    if "atemp" in row:
        row["atemp"] = atemp_norm
    if "temp_combined" in row:
        row["temp_combined"] = (temp_norm + atemp_norm) / 2.0
    if "hum" in row:
        row["hum"] = hum_norm
    if "windspeed" in row:
        row["windspeed"] = wind_norm

    # One-hot features
    for base, value in [
        ("hr", hr),
        ("workingday", workingday),
        ("season", season),
        ("weathersit", weathersit),
    ]:
        col_name = f"{base}_{value}"
        if col_name in row:
            row[col_name] = 1.0

    # Backfill cho model cũ có holiday/weekday
    if "holiday_1" in row:
        row["holiday_1"] = 0.0
    weekday_guess = 1 if workingday == 1 else 0
    weekday_col = f"weekday_{weekday_guess}"
    if weekday_col in row:
        row[weekday_col] = 1.0

    return pd.DataFrame([row], columns=columns)


def predict_with_model(model, raw_input):
    if model is None:
        return None
    x = build_model_input(raw_input, model)
    y_pred = float(model.predict(x)[0])
    return max(0, int(round(y_pred)))


def base_feature_name(feature):
    for prefix in ["hr_", "season_", "weathersit_", "workingday_", "weekday_", "holiday_"]:
        if feature.startswith(prefix):
            return prefix[:-1]
    if feature in ["temp", "atemp", "temp_combined", "hum", "windspeed"]:
        return feature
    return feature


def feature_label_vi(name):
    mapping = {
        "hr": "Giờ",
        "workingday": "Ngày làm việc",
        "season": "Mùa",
        "weathersit": "Thời tiết",
        "temp": "Nhiệt độ",
        "atemp": "Nhiệt độ cảm nhận",
        "temp_combined": "Nhiệt độ tổng hợp",
        "hum": "Độ ẩm",
        "windspeed": "Tốc độ gió",
    }
    return mapping.get(name, name)


# Session state cho màn hình welcome
if "start" not in st.session_state:
    st.session_state.start = False


# Màn hình welcome
if not st.session_state.start:
    st.markdown(
        "<h1 style='text-align:center; font-size:48px;'>HỆ THỐNG DỰ ĐOÁN SỐ LƯỢNG XE ĐẠP THUÊ</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align:center; font-size:20px;'>Dự án Machine Learning dự đoán số lượng xe thuê theo giờ, mùa, thời tiết và môi trường.</p>",
        unsafe_allow_html=True,
    )
    c1, c2, c3 = st.columns([2, 1, 2])
    with c2:
        if st.button("BẮT ĐẦU", use_container_width=True):
            st.session_state.start = True
            st.rerun()
else:
    # Load dữ liệu và model
    df = load_data()
    if df is None:
        st.warning("Không tìm thấy file dữ liệu data/hour.csv")

    models = load_models()
    for model_name, model_obj in models.items():
        if model_obj is None:
            st.warning(f"Không tìm thấy model {MODEL_PATHS[model_name].name}")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Dự đoán", "So sánh mô hình", "Feature Importance", "Kết luận hệ thống"]
    )

    with tab1:
        st.header("DỰ ĐOÁN SỐ LƯỢNG XE THUÊ")
        left_col, right_col = st.columns([1, 1.3], gap="large")

        with left_col:
            st.subheader("Nhập điều kiện dự đoán")
            hr = st.selectbox("Giờ trong ngày", options=list(range(0, 24)), index=8)
            workingday = st.selectbox("Ngày làm việc", options=[0, 1], index=1)
            season = st.selectbox("Mùa", options=[1, 2, 3, 4], index=3)
            weathersit = st.selectbox("Thời tiết", options=[1, 2, 3], index=0)
            temp_combined = st.slider("Nhiệt độ (°C)", min_value=0.0, max_value=41.0, value=25.0, step=0.5)
            hum = st.slider("Độ ẩm (%)", min_value=0.0, max_value=100.0, value=70.0, step=1.0)
            windspeed = st.slider("Tốc độ gió (km/h)", min_value=0.0, max_value=67.0, value=10.0, step=0.5)

            raw_input = {
                "hr": hr,
                "workingday": workingday,
                "season": season,
                "weathersit": weathersit,
                "temp_combined": temp_combined,
                "hum": hum,
                "windspeed": windspeed,
            }
            st.session_state["latest_input"] = raw_input

        with right_col:
            st.subheader("Kết quả dự đoán theo từng model")
            c1, c2, c3 = st.columns(3)

            with c1:
                if models["Linear Regression"] is not None:
                    if st.button("Dự đoán Linear Regression", use_container_width=True):
                        st.session_state["pred_lr"] = predict_with_model(models["Linear Regression"], raw_input)
                    if "pred_lr" in st.session_state:
                        st.metric("Linear Regression", f"{st.session_state['pred_lr']} xe")

            with c2:
                if models["Random Forest"] is not None:
                    if st.button("Dự đoán Random Forest", use_container_width=True):
                        st.session_state["pred_rf"] = predict_with_model(models["Random Forest"], raw_input)
                    if "pred_rf" in st.session_state:
                        st.metric("Random Forest", f"{st.session_state['pred_rf']} xe")

            with c3:
                if models["XGBoost"] is not None:
                    if st.button("Dự đoán XGBoost", use_container_width=True):
                        st.session_state["pred_xgb"] = predict_with_model(models["XGBoost"], raw_input)
                    if "pred_xgb" in st.session_state:
                        st.metric("XGBoost", f"{st.session_state['pred_xgb']} xe")

    with tab2:
        st.header("SO SÁNH HIỆU NĂNG CÁC MÔ HÌNH")
        st.dataframe(METRIC_DF.style.highlight_max(subset=["R2"], color="lightgreen"), use_container_width=True)

        st.subheader("Biểu đồ so sánh metric")
        for metric_name in ["MAE", "RMSE", "R2"]:
            st.bar_chart(METRIC_DF.set_index("Model")[metric_name])

        st.markdown(
            "Random Forest là model tốt nhất vì MAE và RMSE thấp, R2 cao nhất, "
            "bắt tốt pattern phi tuyến của dữ liệu."
        )

    with tab3:
        st.header("FEATURE IMPORTANCE - Random Forest")
        rf_model = models.get("Random Forest")
        if rf_model is None or not hasattr(rf_model, "feature_importances_"):
            st.warning("Không thể hiển thị Feature Importance")
        else:
            cols = model_feature_names(rf_model)
            fi_df = pd.DataFrame(
                {"Feature": cols, "Importance": rf_model.feature_importances_}
            )
            fi_df["FeatureBase"] = fi_df["Feature"].apply(base_feature_name)
            fi_grouped = (
                fi_df.groupby("FeatureBase", as_index=False)["Importance"]
                .sum()
                .sort_values("Importance", ascending=False)
            )
            fi_grouped["FeatureVI"] = fi_grouped["FeatureBase"].apply(feature_label_vi)

            st.bar_chart(fi_grouped.set_index("FeatureVI")["Importance"])
            st.markdown(f"Top feature quan trọng nhất: **{fi_grouped.iloc[0]['FeatureVI']}**")

            raw_input = st.session_state.get(
                "latest_input",
                {
                    "hr": 8,
                    "workingday": 1,
                    "season": 3,
                    "weathersit": 1,
                    "temp_combined": 25.0,
                    "hum": 70.0,
                    "windspeed": 10.0,
                },
            )
            st.subheader("Giá trị thực tế của input hiện tại")
            v1, v2, v3, v4 = st.columns(4)
            v1.metric("Giờ", f"{raw_input['hr']}")
            v2.metric("Nhiệt độ", f"{raw_input['temp_combined']} °C")
            v3.metric("Độ ẩm", f"{raw_input['hum']} %")
            v4.metric("Tốc độ gió", f"{raw_input['windspeed']} km/h")

    with tab4:
        st.header("KẾT LUẬN HỆ THỐNG")
        st.markdown(
            """
            - Giờ trong ngày là yếu tố ảnh hưởng mạnh nhất đến nhu cầu thuê xe.
            - Thời tiết xấu làm giảm đáng kể số lượng thuê xe.
            - Nhiệt độ dễ chịu và độ ẩm vừa phải giúp tăng nhu cầu thuê xe.
            - Random Forest là model tốt nhất, dự đoán ổn định các giờ cao điểm và điều kiện phi tuyến.
            """
        )
