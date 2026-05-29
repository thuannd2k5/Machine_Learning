from pathlib import Path

import altair as alt
import joblib
import pandas as pd
import streamlit as st
from PIL import Image


st.set_page_config(
    page_title="Dự đoán nhu cầu thuê xe đạp",
    page_icon="🚲",
    layout="wide",
    initial_sidebar_state="collapsed",
)

APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent
MODEL_DIR = APP_DIR / "../models"
IMAGE_DIR = APP_DIR / "../images"

FEATURES = ["hr", "workingday", "season", "weathersit", "temp_combined", "hum", "windspeed"]
DEFAULT_COLUMNS = [
    "temp_combined",
    "hum",
    "windspeed",
    *[f"hr_{i}" for i in range(1, 24)],
    "workingday_1",
    "season_2",
    "season_3",
    "season_4",
    "weathersit_2",
    "weathersit_3",
]

MODEL_METRICS = pd.DataFrame(
    {
        "Model": ["Linear Regression", "Random Forest", "XGBoost"],
        "MAE": [78.04, 49.64, 51.98],
        "RMSE": [109.22, 74.98, 75.32],
        "R²": [0.623, 0.822, 0.821],
    }
)

SEASON_LABELS = {
    1: "1 - Xuân",
    2: "2 - Hè",
    3: "3 - Thu",
    4: "4 - Đông",
}

WEATHER_LABELS = {
    1: "1 - Trời đẹp / ít mây",
    2: "2 - Nhiều mây / sương nhẹ",
    3: "3 - Mưa nhẹ / tuyết nhẹ",
    4: "4 - Thời tiết rất xấu",
}

WORKINGDAY_LABELS = {
    0: "0 - Ngày nghỉ",
    1: "1 - Ngày làm việc",
}


st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(29, 78, 216, 0.10), transparent 30%),
            linear-gradient(180deg, #f8fafc 0%, #eef2f7 100%);
    }
    .welcome {
        min-height: 82vh;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
        animation: fadeUp 700ms ease-out;
    }
    .welcome h1 {
        font-size: clamp(2.4rem, 6vw, 5.1rem);
        line-height: 1.02;
        margin-bottom: 0.8rem;
        color: #0f172a;
    }
    .welcome p {
        max-width: 780px;
        font-size: 1.2rem;
        color: #475569;
    }
    @keyframes fadeUp {
        from { opacity: 0; transform: translateY(16px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .section-box {
        padding: 1.25rem;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        background: rgba(255, 255, 255, 0.82);
    }
    .prediction-card {
        padding: 1.15rem;
        border-radius: 8px;
        background: #ffffff;
        border: 1px solid #dbe3ee;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
        min-height: 138px;
    }
    .prediction-card h3 {
        margin: 0 0 0.45rem 0;
        font-size: 1rem;
        color: #334155;
    }
    .prediction-value {
        font-size: 2.1rem;
        font-weight: 760;
        color: #0f766e;
        line-height: 1.1;
    }
    .prediction-note {
        margin-top: 0.35rem;
        color: #64748b;
        font-size: 0.9rem;
    }
    .insight {
        padding: 1rem;
        border-left: 5px solid #0f766e;
        background: #f0fdfa;
        color: #134e4a;
        border-radius: 6px;
    }
    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_model(file_name: str):
    model_path = MODEL_DIR / file_name
    if not model_path.exists():
        return None, f"Không tìm thấy model: {model_path.as_posix()}"
    try:
        return joblib.load(model_path), None
    except Exception as exc:
        return None, f"Không thể load {file_name}: {exc}"


@st.cache_resource(show_spinner=False)
def load_training_columns():
    columns_path = MODEL_DIR / "X_train_columns.pkl"
    if columns_path.exists():
        try:
            return list(joblib.load(columns_path))
        except Exception:
            return None
    return None


def model_columns(model):
    if model is not None and hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    saved_columns = load_training_columns()
    return saved_columns or DEFAULT_COLUMNS


def build_encoded_input(values: dict, columns: list[str], scale_numeric: bool) -> pd.DataFrame:
    numeric_values = {
        "temp_combined": values["temp_combined"] / 41 if scale_numeric else values["temp_combined"],
        "hum": values["hum"] / 100 if scale_numeric else values["hum"],
        "windspeed": values["windspeed"] / 67 if scale_numeric else values["windspeed"],
    }

    row = {column: 0 for column in columns}
    for key, value in numeric_values.items():
        if key in row:
            row[key] = value

    for prefix, selected in {
        "hr": values["hr"],
        "workingday": values["workingday"],
        "season": values["season"],
        "weathersit": values["weathersit"],
    }.items():
        column = f"{prefix}_{selected}"
        if column in row:
            row[column] = 1

    return pd.DataFrame([row], columns=columns)


def predict_model(model, input_df: pd.DataFrame):
    if model is None:
        return None
    prediction = float(model.predict(input_df)[0])
    return max(0, round(prediction))


def render_prediction_card(title: str, value, note: str):
    display_value = "Chưa có model" if value is None else f"{value:,} xe"
    st.markdown(
        f"""
        <div class="prediction-card">
            <h3>{title}</h3>
            <div class="prediction-value">{display_value}</div>
            <div class="prediction-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def current_input_table(values: dict) -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("⏰ Giờ", "hr", values["hr"]),
            ("📅 Ngày làm việc", "workingday", WORKINGDAY_LABELS[values["workingday"]]),
            ("🌱 Mùa", "season", SEASON_LABELS[values["season"]]),
            ("🌤️ Thời tiết", "weathersit", WEATHER_LABELS[values["weathersit"]]),
            ("🌡️ Nhiệt độ", "temp_combined", f"{values['temp_combined']:.1f} °C"),
            ("💧 Độ ẩm", "hum", f"{values['hum']:.0f}%"),
            ("🌪️ Tốc độ gió", "windspeed", f"{values['windspeed']:.1f} km/h"),
        ],
        columns=["Thông tin", "Feature", "Giá trị thực tế"],
    )


def render_warning(message: str):
    st.warning(message, icon="⚠️")


if "started" not in st.session_state:
    st.session_state.started = False

if "latest_input" not in st.session_state:
    st.session_state.latest_input = {
        "hr": 8,
        "workingday": 1,
        "season": 2,
        "weathersit": 1,
        "temp_combined": 25.0,
        "hum": 60.0,
        "windspeed": 12.0,
    }


if not st.session_state.started:
    st.markdown(
        """
        <div class="welcome">
            <h1>🚲 Dashboard AI Dự Đoán Nhu Cầu Thuê Xe Đạp</h1>
            <p>Phân tích EDA, mức độ quan trọng của feature, so sánh model và dự đoán nhu cầu thuê xe theo giờ bằng Machine Learning.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    _, center, _ = st.columns([1, 1, 1])
    with center:
        if st.button("Bắt đầu phân tích", use_container_width=True, type="primary"):
            st.session_state.started = True
            st.rerun()
    st.stop()


lr_model, lr_warning = load_model("linear_regression_model.pkl")
rf_model, rf_warning = load_model("random_forest_model.pkl")
xgb_model, xgb_warning = load_model("xgboost_model.pkl")

st.title("🚲 Dự đoán nhu cầu thuê xe đạp theo giờ")
st.caption("Dashboard Machine Learning cho bài toán bike sharing demand.")

warnings = [warning for warning in [lr_warning, rf_warning, xgb_warning] if warning]
if warnings:
    with st.container():
        for warning in warnings:
            render_warning(warning)

tab_predict, tab_feature, tab_eda, tab_compare, tab_conclusion = st.tabs(
    ["🧠 Dự đoán", "📌 Feature Importance", "📊 EDA", "📈 So sánh Model", "✅ Kết luận"]
)

with tab_predict:
    st.subheader("Nhập điều kiện dự đoán")
    input_area, result_area = st.columns([1.05, 1.35], gap="large")

    with input_area:
        with st.container(border=True):
            c1, c2 = st.columns(2)
            with c1:
                hr = st.slider("⏰ Giờ trong ngày", 0, 23, st.session_state.latest_input["hr"], 1)
                season = st.selectbox(
                    "🌱 Mùa",
                    options=list(SEASON_LABELS.keys()),
                    format_func=lambda value: SEASON_LABELS[value],
                    index=list(SEASON_LABELS.keys()).index(st.session_state.latest_input["season"]),
                )
                temp_combined = st.slider(
                    "🌡️ Nhiệt độ thực tế (°C)",
                    0.0,
                    41.0,
                    float(st.session_state.latest_input["temp_combined"]),
                    0.5,
                )
                windspeed = st.slider(
                    "🌪️ Tốc độ gió thực tế (km/h)",
                    0.0,
                    67.0,
                    float(st.session_state.latest_input["windspeed"]),
                    0.5,
                )
            with c2:
                workingday = st.selectbox(
                    "📅 Loại ngày",
                    options=list(WORKINGDAY_LABELS.keys()),
                    format_func=lambda value: WORKINGDAY_LABELS[value],
                    index=list(WORKINGDAY_LABELS.keys()).index(st.session_state.latest_input["workingday"]),
                )
                weathersit = st.selectbox(
                    "🌤️ Tình trạng thời tiết",
                    options=list(WEATHER_LABELS.keys()),
                    format_func=lambda value: WEATHER_LABELS[value],
                    index=list(WEATHER_LABELS.keys()).index(st.session_state.latest_input["weathersit"]),
                )
                hum = st.slider(
                    "💧 Độ ẩm thực tế (%)",
                    0.0,
                    100.0,
                    float(st.session_state.latest_input["hum"]),
                    1.0,
                )

    values = {
        "hr": hr,
        "workingday": workingday,
        "season": season,
        "weathersit": weathersit,
        "temp_combined": temp_combined,
        "hum": hum,
        "windspeed": windspeed,
    }
    st.session_state.latest_input = values

    lr_input = build_encoded_input(values, model_columns(lr_model), scale_numeric=True)
    rf_input = build_encoded_input(values, model_columns(rf_model), scale_numeric=False)
    xgb_input = build_encoded_input(values, model_columns(xgb_model), scale_numeric=False)

    lr_prediction = predict_model(lr_model, lr_input)
    rf_prediction = predict_model(rf_model, rf_input)
    xgb_prediction = predict_model(xgb_model, xgb_input)

    with result_area:
        st.markdown("#### Kết quả dự đoán từng model")
        p1, p2, p3 = st.columns(3)
        with p1:
            render_prediction_card(
                "Linear Regression",
                lr_prediction,
                "Numeric tự chuẩn hóa: temp/41, hum/100, windspeed/67.",
            )
        with p2:
            render_prediction_card(
                "Random Forest",
                rf_prediction,
                "Nhận giá trị thực tế từ người dùng.",
            )
        with p3:
            render_prediction_card(
                "XGBoost",
                xgb_prediction,
                "Nhận giá trị thực tế từ người dùng.",
            )

        st.markdown("#### Input đang sử dụng")
        st.dataframe(current_input_table(values), use_container_width=True, hide_index=True)

with tab_feature:
    left, right = st.columns([1.4, 1], gap="large")
    with left:
        st.subheader("Feature Importance - Random Forest")
        fi_path = IMAGE_DIR / "feature_importance_rf.png"
        if fi_path.exists():
            st.image(Image.open(fi_path), use_container_width=True)
        else:
            render_warning("Không tìm thấy ảnh images/feature_importance_rf.png.")

    with right:
        st.subheader("Giá trị feature hiện tại")
        st.dataframe(current_input_table(st.session_state.latest_input), use_container_width=True, hide_index=True)
        st.markdown(
            """
            <div class="insight">
            <b>Insight business:</b> giờ trong ngày thường là tín hiệu mạnh nhất vì nhu cầu tăng rõ vào khung đi làm và tan làm.
            Thời tiết xấu làm giảm nhu cầu, trong khi nhiệt độ dễ chịu hỗ trợ lượng thuê xe cao hơn.
            </div>
            """,
            unsafe_allow_html=True,
        )

with tab_eda:
    st.subheader("Khám phá dữ liệu EDA")
    eda_files = sorted(
        [
            path
            for path in IMAGE_DIR.glob("*.png")
            if path.name.startswith(("hist_", "box_", "count_", "line_", "bar_", "heatmap_"))
        ]
    )
    if not eda_files:
        render_warning("Không tìm thấy biểu đồ EDA trong thư mục images/.")
    else:
        labels = {path.name: path for path in eda_files}
        selector_col, _ = st.columns([1, 2])
        with selector_col:
            selected_name = st.selectbox("Chọn biểu đồ EDA", list(labels.keys()))
        st.image(Image.open(labels[selected_name]), use_container_width=True)

with tab_compare:
    st.subheader("So sánh hiệu năng 3 model")

    table_df = MODEL_METRICS.copy()
    styled_table = table_df.style.apply(
        lambda row: ["background-color: #dcfce7; font-weight: 700" if row["Model"] == "Random Forest" else "" for _ in row],
        axis=1,
    )
    st.dataframe(styled_table, use_container_width=True, hide_index=True)

    metric_cols = st.columns(3)
    best_values = {"MAE": "thấp nhất", "RMSE": "thấp nhất", "R²": "cao nhất"}
    for col, metric in zip(metric_cols, ["MAE", "RMSE", "R²"]):
        with col:
            chart = (
                alt.Chart(MODEL_METRICS)
                .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
                .encode(
                    x=alt.X("Model:N", sort=None, title=None),
                    y=alt.Y(f"{metric}:Q", title=metric),
                    color=alt.condition(
                        alt.datum.Model == "Random Forest",
                        alt.value("#0f766e"),
                        alt.value("#94a3b8"),
                    ),
                    tooltip=["Model", metric],
                )
                .properties(height=260, title=f"{metric} - tốt nhất là {best_values[metric]}")
            )
            st.altair_chart(chart, use_container_width=True)

    st.markdown(
        """
        <div class="insight">
        <b>Random Forest là model tốt nhất</b> vì có MAE và RMSE thấp nhất, đồng thời R² cao nhất.
        Mô hình cây ensemble bắt được quan hệ phi tuyến giữa giờ, thời tiết, mùa và nhu cầu thuê xe tốt hơn Linear Regression.
        </div>
        """,
        unsafe_allow_html=True,
    )

with tab_conclusion:
    st.subheader("Kết luận")

    w1, w2, w3, w4 = st.columns(4)
    for col, step, text in [
        (w1, "EDA", "Hiểu phân phối, outlier và xu hướng theo thời gian."),
        (w2, "Feature", "Xác định biến ảnh hưởng mạnh đến nhu cầu."),
        (w3, "Model", "Huấn luyện và so sánh Linear Regression, Random Forest, XGBoost."),
        (w4, "Prediction", "Dự đoán số lượt thuê xe từ input thực tế."),
    ]:
        with col:
            st.metric(step, text)

    st.markdown("#### Insight BI")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.info("Giờ cao điểm ảnh hưởng mạnh đến nhu cầu thuê xe.", icon="⏰")
    with c2:
        st.info("Thời tiết xấu làm giảm nhu cầu thuê xe.", icon="🌤️")
    with c3:
        st.info("Random Forest là model tốt nhất trong 3 model.", icon="📈")
