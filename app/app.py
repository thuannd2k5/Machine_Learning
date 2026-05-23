"""
Hệ thống dự đoán số lượng xe đạp thuê bằng Machine Learning.

Chạy ứng dụng:
    streamlit run app/app.py
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


# =============================================================================
# CẤU HÌNH CHUNG
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"

MODEL_FILES = {
    "Linear Regression": "linear_regression.pkl",
    "Random Forest": "random_forest.pkl",
}

MODEL_METRICS = pd.DataFrame(
    {
        "Mô hình": ["Linear Regression", "Random Forest"],
        "MAE": [77.958290, 49.639018],
        "RMSE": [109.163930, 74.978296],
        "R² Score": [0.623666, 0.822464],
    }
)

FALLBACK_FEATURES = [
    "temp",
    "atemp",
    "hum",
    "windspeed",
    "season_2",
    "season_3",
    "season_4",
    "hr_1",
    "hr_2",
    "hr_3",
    "hr_4",
    "hr_5",
    "hr_6",
    "hr_7",
    "hr_8",
    "hr_9",
    "hr_10",
    "hr_11",
    "hr_12",
    "hr_13",
    "hr_14",
    "hr_15",
    "hr_16",
    "hr_17",
    "hr_18",
    "hr_19",
    "hr_20",
    "hr_21",
    "hr_22",
    "hr_23",
    "holiday_1",
    "weekday_1",
    "weekday_2",
    "weekday_3",
    "weekday_4",
    "weekday_5",
    "weekday_6",
    "workingday_1",
    "weathersit_2",
    "weathersit_3",
]

SEASON_OPTIONS = {
    "🌸 Xuân": 1,
    "☀️ Hè": 2,
    "🍂 Thu": 3,
    "❄️ Đông": 4,
}

WEEKDAY_OPTIONS = {
    "Chủ nhật": 0,
    "Thứ hai": 1,
    "Thứ ba": 2,
    "Thứ tư": 3,
    "Thứ năm": 4,
    "Thứ sáu": 5,
    "Thứ bảy": 6,
}

WEATHER_OPTIONS = {
    "☀️ Trời quang, ít mây": 1,
    "🌥️ Có mây, sương nhẹ": 2,
    "🌧️ Mưa nhẹ hoặc tuyết nhẹ": 3,
    "⛈️ Mưa lớn hoặc thời tiết xấu": 4,
}

MODEL_DESCRIPTIONS = {
    "Linear Regression": {
        "icon": "📈",
        "summary": "Mô hình tuyến tính dùng làm baseline, dễ giải thích và phản hồi rất nhanh.",
        "strength": "Phù hợp để tham chiếu xu hướng tổng quát.",
    },
    "Random Forest": {
        "icon": "🌲",
        "summary": "Mô hình ensemble bắt tốt quan hệ phi tuyến giữa giờ, thời tiết và nhu cầu thuê xe.",
        "strength": "Model tối ưu nhất của hệ thống.",
    },
}


st.set_page_config(
    page_title="Dự đoán xe đạp thuê",
    page_icon="🚲",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# =============================================================================
# CSS GIAO DIỆN
# =============================================================================

def inject_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg: #f5f7fb;
            --panel: rgba(255, 255, 255, 0.92);
            --ink: #14213d;
            --muted: #607086;
            --blue: #2563eb;
            --cyan: #0891b2;
            --green: #059669;
            --line: rgba(15, 23, 42, 0.10);
            --shadow: 0 18px 45px rgba(15, 23, 42, 0.10);
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(37, 99, 235, 0.14), transparent 32rem),
                radial-gradient(circle at top right, rgba(5, 150, 105, 0.12), transparent 28rem),
                linear-gradient(180deg, #f8fbff 0%, #eef3f8 100%);
            color: var(--ink);
        }

        .block-container {
            max-width: 1280px;
            padding-top: 1.4rem;
            padding-bottom: 2.2rem;
        }

        [data-testid="stHeader"] {
            background: rgba(248, 251, 255, 0.72);
            backdrop-filter: blur(14px);
        }

        h1, h2, h3 { letter-spacing: 0 !important; }

        .welcome-screen {
            min-height: calc(100vh - 5rem);
            display: flex;
            align-items: center;
            justify-content: center;
            position: relative;
            overflow: hidden;
            border-radius: 8px;
            padding: 2rem;
            background:
                linear-gradient(135deg, rgba(6, 18, 43, 0.82), rgba(14, 116, 144, 0.70)),
                url("https://images.unsplash.com/photo-1507035895480-2b3156c31fc8?auto=format&fit=crop&w=1800&q=80");
            background-size: cover;
            background-position: center;
            box-shadow: var(--shadow);
        }

        .welcome-screen:before {
            content: "";
            position: absolute;
            inset: 0;
            background:
                linear-gradient(90deg, rgba(255,255,255,0.08) 1px, transparent 1px),
                linear-gradient(180deg, rgba(255,255,255,0.08) 1px, transparent 1px);
            background-size: 56px 56px;
            animation: gridMove 14s linear infinite;
        }

        @keyframes gridMove {
            from { transform: translate3d(0, 0, 0); }
            to { transform: translate3d(56px, 56px, 0); }
        }

        .welcome-content {
            position: relative;
            z-index: 2;
            max-width: 920px;
            text-align: center;
            color: white;
            animation: fadeUp 800ms ease both;
        }

        @keyframes fadeUp {
            from { opacity: 0; transform: translateY(16px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .welcome-kicker {
            display: inline-flex;
            gap: 0.5rem;
            align-items: center;
            padding: 0.45rem 0.75rem;
            border: 1px solid rgba(255, 255, 255, 0.28);
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.12);
            backdrop-filter: blur(12px);
            font-weight: 700;
            margin-bottom: 1.1rem;
        }

        .welcome-title {
            font-size: clamp(2.35rem, 6vw, 5rem);
            line-height: 1.03;
            margin: 0;
            font-weight: 900;
        }

        .welcome-subtitle {
            max-width: 760px;
            margin: 1.1rem auto 1.6rem;
            font-size: clamp(1rem, 2.2vw, 1.25rem);
            color: rgba(255, 255, 255, 0.90);
        }

        .dashboard-hero {
            padding: 1.5rem;
            border: 1px solid var(--line);
            border-radius: 8px;
            background:
                linear-gradient(135deg, rgba(255,255,255,0.95), rgba(241,245,249,0.92)),
                linear-gradient(90deg, rgba(37,99,235,0.12), rgba(5,150,105,0.10));
            box-shadow: var(--shadow);
        }

        .dashboard-title {
            font-size: clamp(1.8rem, 4vw, 3.1rem);
            line-height: 1.08;
            margin: 0.2rem 0 0.5rem;
            font-weight: 900;
            color: var(--ink);
        }

        .dashboard-subtitle {
            color: var(--muted);
            font-size: 1.02rem;
            margin: 0;
        }

        .card {
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 1.05rem;
            background: var(--panel);
            box-shadow: 0 12px 30px rgba(15, 23, 42, 0.07);
            height: 100%;
        }

        .metric-card {
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 1rem;
            background: linear-gradient(180deg, #ffffff, #f8fafc);
            box-shadow: 0 10px 26px rgba(15, 23, 42, 0.06);
        }

        .metric-label {
            color: var(--muted);
            font-size: 0.86rem;
            font-weight: 700;
            margin-bottom: 0.35rem;
        }

        .metric-value {
            color: var(--ink);
            font-size: clamp(1.45rem, 3vw, 2rem);
            font-weight: 900;
        }

        .badge {
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            padding: 0.32rem 0.58rem;
            border-radius: 999px;
            background: rgba(37, 99, 235, 0.10);
            color: #1d4ed8;
            font-weight: 800;
            font-size: 0.82rem;
        }

        .section-title {
            font-size: 1.25rem;
            font-weight: 900;
            margin: 0 0 0.35rem;
            color: var(--ink);
        }

        .muted { color: var(--muted); }
        .big-result {
            text-align: center;
            padding: 1.15rem;
            border-radius: 8px;
            background: linear-gradient(135deg, rgba(37,99,235,0.12), rgba(5,150,105,0.14));
            border: 1px solid rgba(37,99,235,0.18);
            margin-top: 0.9rem;
        }
        .big-result-value {
            font-size: clamp(2rem, 6vw, 3.5rem);
            line-height: 1;
            font-weight: 900;
            color: #075985;
        }

        div[data-testid="stButton"] > button {
            border-radius: 8px;
            min-height: 3rem;
            font-weight: 850;
            border: 1px solid rgba(37, 99, 235, 0.25);
            box-shadow: 0 10px 22px rgba(37, 99, 235, 0.14);
        }

        .welcome-button div[data-testid="stButton"] > button {
            min-height: 4rem;
            font-size: 1.15rem;
            background: linear-gradient(135deg, #22c55e, #2563eb);
            color: white;
            border: 1px solid rgba(255,255,255,0.32);
        }

        div[data-testid="stMetric"] {
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 0.9rem;
            background: #ffffff;
            box-shadow: 0 8px 22px rgba(15, 23, 42, 0.06);
        }

        @media (max-width: 768px) {
            .block-container { padding-left: 1rem; padding-right: 1rem; }
            .welcome-screen { padding: 1rem; min-height: calc(100vh - 4rem); }
            .card, .metric-card, .dashboard-hero { padding: 0.9rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# LOAD MODEL VÀ TIỀN XỬ LÝ
# =============================================================================

@st.cache_resource(show_spinner=False)
def load_models() -> tuple[dict[str, object], list[str]]:
    models: dict[str, object] = {}
    warnings: list[str] = []

    for model_name, file_name in MODEL_FILES.items():
        model_path = MODEL_DIR / file_name
        if model_path.exists():
            models[model_name] = joblib.load(model_path)
        else:
            warnings.append(f"Không tìm thấy file `{model_path}`.")

    return models, warnings


def get_feature_names(model: object | None = None) -> list[str]:
    if model is not None and hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    return FALLBACK_FEATURES


def normalize_environment(temp_c: float, atemp_c: float, hum_pct: float, windspeed_kmh: float) -> dict[str, float]:
    """Chuẩn hóa input thực tế về thang đo của Bike Sharing Dataset."""
    return {
        "temp": float(np.clip(temp_c / 41.0, 0.0, 1.0)),
        "atemp": float(np.clip(atemp_c / 50.0, 0.0, 1.0)),
        "hum": float(np.clip(hum_pct / 100.0, 0.0, 1.0)),
        "windspeed": float(np.clip(windspeed_kmh / 67.0, 0.0, 1.0)),
    }


def build_model_input(raw_input: dict[str, int | float], feature_names: list[str]) -> pd.DataFrame:
    row = {feature: 0.0 for feature in feature_names}

    for feature in ["temp", "atemp", "hum", "windspeed"]:
        if feature in row:
            row[feature] = float(raw_input[feature])

    for feature in ["season", "hr", "holiday", "weekday", "workingday", "weathersit"]:
        value = int(raw_input[feature])
        col = f"{feature}_{value}"

        if col in row:
            row[col] = 1.0
        elif feature == "weathersit" and value == 4 and "weathersit_3" in row:
            # Model đã train không có cột weathersit_4, nên thời tiết rất xấu
            # được đưa về nhóm mưa/tuyết nhẹ gần nhất.
            row["weathersit_3"] = 1.0

    return pd.DataFrame([row], columns=feature_names)


def predict_with_model(model: object, raw_input: dict[str, int | float]) -> int:
    feature_names = get_feature_names(model)
    model_input = build_model_input(raw_input, feature_names)
    prediction = float(model.predict(model_input)[0])
    return max(0, int(round(prediction)))


def get_base_feature(feature_name: str) -> str:
    if feature_name.startswith("hr_"):
        return "hr"
    if feature_name.startswith("season_"):
        return "season"
    if feature_name.startswith("weekday_"):
        return "weekday"
    if feature_name.startswith("holiday_"):
        return "holiday"
    if feature_name.startswith("workingday_"):
        return "workingday"
    if feature_name.startswith("weathersit_"):
        return "weathersit"
    return feature_name


# =============================================================================
# COMPONENTS
# =============================================================================

def metric_card(label: str, value: str, note: str = "") -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="muted">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_welcome() -> None:
    st.markdown(
        """
        <div class="welcome-screen">
            <div class="welcome-content">
                <div class="welcome-kicker">🧠 Machine Learning Dashboard</div>
                <h1 class="welcome-title">🚲 HỆ THỐNG DỰ ĐOÁN SỐ LƯỢNG XE ĐẠP THUÊ</h1>
                <p class="welcome-subtitle">
                    Ứng dụng sử dụng dữ liệu thời tiết, thời gian trong ngày và các yếu tố vận hành
                    để dự báo nhu cầu thuê xe theo giờ, hỗ trợ phân bổ xe và lập kế hoạch kinh doanh.
                </p>
                <div style="display:flex; gap:0.7rem; justify-content:center; flex-wrap:wrap;">
                    <span class="badge">📊 EDA</span>
                    <span class="badge">🌲 Random Forest</span>
                    <span class="badge">📈 Linear Regression</span>
                    <span class="badge">⚡ Streamlit</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="welcome-button">', unsafe_allow_html=True)
    left, center, right = st.columns([1.2, 1, 1.2])
    with center:
        if st.button("🚀 BẮT ĐẦU", use_container_width=True, type="primary"):
            st.session_state.started = True
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)


def render_header(models: dict[str, object]) -> None:
    st.markdown(
        """
        <div class="dashboard-hero">
            <span class="badge">🚲 AI Demand Forecasting</span>
            <h1 class="dashboard-title">Dashboard dự đoán nhu cầu thuê xe đạp</h1>
            <p class="dashboard-subtitle">
                Nhập điều kiện vận hành, chọn mô hình và xem kết quả dự báo theo từng thuật toán.
                Giao diện không sử dụng input năm hoặc tháng vì hai cột này đã được loại bỏ khỏi pipeline.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        metric_card("Mô hình đã load", f"{len(models)}/2", "Linear Regression, Random Forest")
    with col2:
        metric_card("Best model", "Random Forest", "R² Score = 0.822")
    with col3:
        metric_card("Feature đầu vào", "10", "Không có yr, mnth")
    with col4:
        metric_card("Bài toán", "Regression", "Dự đoán cnt theo giờ")


def render_input_panel() -> dict[str, int | float | str]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<p class="section-title">🎛️ Cấu hình đầu vào</p>', unsafe_allow_html=True)
    st.caption("Các giá trị nhiệt độ, độ ẩm và tốc độ gió được nhập theo đơn vị thực tế, sau đó hệ thống tự chuẩn hóa.")

    col_a, col_b = st.columns(2)
    with col_a:
        season_label = st.selectbox("🍂 Mùa trong năm", list(SEASON_OPTIONS.keys()))
        hour = st.slider("⏰ Giờ trong ngày", 0, 23, 8, 1, format="%d:00")
        holiday_label = st.selectbox("🎉 Ngày lễ", ["Không", "Có"])
        weekday_label = st.selectbox("📅 Thứ trong tuần", list(WEEKDAY_OPTIONS.keys()), index=1)
        workingday_label = st.selectbox("💼 Ngày làm việc", ["Có", "Không"])

    with col_b:
        weather_label = st.selectbox("🌤️ Thời tiết", list(WEATHER_OPTIONS.keys()))
        temp_c = st.slider("🌡️ Nhiệt độ thực tế (°C)", -5.0, 41.0, 24.0, 0.5)
        atemp_c = st.slider("🌡️ Nhiệt độ cảm nhận (°C)", -10.0, 50.0, 26.0, 0.5)
        hum_pct = st.slider("💧 Độ ẩm (%)", 0, 100, 62, 1)
        windspeed_kmh = st.slider("🌪️ Tốc độ gió (km/h)", 0.0, 67.0, 12.0, 0.5)

    normalized = normalize_environment(temp_c, atemp_c, hum_pct, windspeed_kmh)

    st.divider()
    n1, n2, n3, n4 = st.columns(4)
    with n1:
        st.metric("temp chuẩn hóa", f"{normalized['temp']:.3f}")
    with n2:
        st.metric("atemp chuẩn hóa", f"{normalized['atemp']:.3f}")
    with n3:
        st.metric("hum chuẩn hóa", f"{normalized['hum']:.3f}")
    with n4:
        st.metric("windspeed chuẩn hóa", f"{normalized['windspeed']:.3f}")

    st.markdown("</div>", unsafe_allow_html=True)

    return {
        "season": SEASON_OPTIONS[season_label],
        "hr": hour,
        "holiday": 1 if holiday_label == "Có" else 0,
        "weekday": WEEKDAY_OPTIONS[weekday_label],
        "workingday": 1 if workingday_label == "Có" else 0,
        "weathersit": WEATHER_OPTIONS[weather_label],
        **normalized,
        "season_label": season_label,
        "weather_label": weather_label,
        "weekday_label": weekday_label,
        "holiday_label": holiday_label,
        "workingday_label": workingday_label,
        "temp_c": temp_c,
        "atemp_c": atemp_c,
        "hum_pct": hum_pct,
        "windspeed_kmh": windspeed_kmh,
    }


def render_prediction_card(model_name: str, model: object, raw_input: dict[str, int | float | str]) -> None:
    info = MODEL_DESCRIPTIONS[model_name]
    key = model_name.lower().replace(" ", "_")

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(
        f"""
        <span class="badge">{info["icon"]} {model_name}</span>
        <h3 style="margin:0.75rem 0 0.35rem;">{model_name}</h3>
        <p class="muted">{info["summary"]}</p>
        <p><strong>{info["strength"]}</strong></p>
        """,
        unsafe_allow_html=True,
    )

    metric_row = MODEL_METRICS[MODEL_METRICS["Mô hình"] == model_name].iloc[0]
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("MAE", f"{metric_row['MAE']:.2f}")
    with col2:
        st.metric("RMSE", f"{metric_row['RMSE']:.2f}")
    with col3:
        st.metric("R²", f"{metric_row['R² Score']:.3f}")

    if st.button(f"🔮 Dự đoán với {model_name}", key=f"predict_{key}", use_container_width=True, type="primary"):
        st.session_state[f"prediction_{key}"] = predict_with_model(model, raw_input)

    prediction = st.session_state.get(f"prediction_{key}")
    if prediction is not None:
        st.markdown(
            f"""
            <div class="big-result">
                <div class="muted">Số lượng xe dự đoán</div>
                <div class="big-result-value">{prediction:,}</div>
                <div><strong>{prediction:,} xe</strong></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)


def render_prediction_tab(models: dict[str, object]) -> None:
    left, right = st.columns([1, 1.1], gap="large")
    with left:
        raw_input = render_input_panel()

    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<p class="section-title">🧾 Tóm tắt kịch bản</p>', unsafe_allow_html=True)
        s1, s2, s3 = st.columns(3)
        with s1:
            st.metric("Giờ", f"{int(raw_input['hr']):02d}:00")
        with s2:
            st.metric("Mùa", str(raw_input["season_label"]).split(" ", 1)[-1])
        with s3:
            st.metric("Thời tiết", str(raw_input["weather_label"]).split(" ", 1)[-1])
        st.caption("Mỗi model có nút dự đoán riêng, kết quả được lưu độc lập trong session.")
        st.markdown("</div>", unsafe_allow_html=True)

        st.write("")
        for model_name in MODEL_FILES:
            if model_name in models:
                render_prediction_card(model_name, models[model_name], raw_input)
                st.write("")
            else:
                st.warning(f"⚠️ Chưa thể hiển thị `{model_name}` vì model chưa được load.")


def render_comparison_tab() -> None:
    st.markdown('<p class="section-title">📊 Bảng so sánh mô hình</p>', unsafe_allow_html=True)

    styled = MODEL_METRICS.style.format(
        {"MAE": "{:.2f}", "RMSE": "{:.2f}", "R² Score": "{:.3f}"}
    ).highlight_min(subset=["MAE", "RMSE"], color="#dcfce7").highlight_max(
        subset=["R² Score"], color="#dcfce7"
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)

    st.divider()
    col1, col2 = st.columns([1.2, 0.8], gap="large")
    with col1:
        chart_data = MODEL_METRICS.set_index("Mô hình")[["MAE", "RMSE", "R² Score"]]
        st.bar_chart(chart_data, use_container_width=True)

    with col2:
        st.markdown(
            """
            <div class="card">
                <span class="badge">🏆 Model tốt nhất</span>
                <h3>Random Forest</h3>
                <p class="muted">
                    Random Forest đạt MAE và RMSE thấp hơn, đồng thời R² Score cao hơn Linear Regression.
                    Điều này cho thấy mô hình giải thích được nhiều biến động nhu cầu thuê xe hơn.
                </p>
                <p>
                    Lý do chính: nhu cầu thuê xe phụ thuộc phi tuyến vào giờ cao điểm, thời tiết,
                    độ ẩm và ngày làm việc. Random Forest có thể học các tương tác phức tạp này
                    tốt hơn mô hình tuyến tính.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.info(
        "Insight: Linear Regression phù hợp làm baseline, nhưng Random Forest nên được chọn cho dự báo vận hành vì sai số thấp hơn và khả năng bắt pattern giờ cao điểm tốt hơn."
    )


def render_feature_importance_tab(models: dict[str, object]) -> None:
    rf_model = models.get("Random Forest")
    if rf_model is None:
        st.warning("⚠️ Không tìm thấy `random_forest.pkl`, chưa thể hiển thị Feature Importance.")
        return

    if not hasattr(rf_model, "feature_importances_"):
        st.warning("⚠️ Model Random Forest hiện tại không hỗ trợ `feature_importances_`.")
        return

    feature_names = get_feature_names(rf_model)
    importance_df = pd.DataFrame(
        {
            "Feature": feature_names,
            "Importance": rf_model.feature_importances_,
        }
    )
    importance_df["Nhóm feature"] = importance_df["Feature"].map(get_base_feature)

    grouped = (
        importance_df.groupby("Nhóm feature", as_index=False)["Importance"]
        .sum()
        .sort_values("Importance", ascending=False)
    )

    col1, col2 = st.columns([1.2, 0.8], gap="large")
    with col1:
        fig, ax = plt.subplots(figsize=(9, 5.4))
        top_grouped = grouped.head(10).sort_values("Importance", ascending=True)
        colors = ["#0ea5e9" if x != top_grouped["Importance"].max() else "#059669" for x in top_grouped["Importance"]]
        ax.barh(top_grouped["Nhóm feature"], top_grouped["Importance"], color=colors)
        ax.set_xlabel("Mức độ quan trọng")
        ax.set_ylabel("")
        ax.set_title("Feature Importance tổng hợp - Random Forest", fontweight="bold")
        ax.grid(axis="x", alpha=0.18)
        for spine in ["top", "right", "left"]:
            ax.spines[spine].set_visible(False)
        st.pyplot(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<span class="badge">🎯 Top feature</span>', unsafe_allow_html=True)
        for _, row in grouped.head(5).iterrows():
            st.metric(row["Nhóm feature"], f"{row['Importance']:.3f}")
        st.markdown("</div>", unsafe_allow_html=True)

    st.divider()
    st.markdown('<p class="section-title">📌 Business insight</p>', unsafe_allow_html=True)
    st.write(
        """
        `hr` thường là nhóm feature quan trọng nhất vì hành vi thuê xe thay đổi mạnh theo khung giờ:
        sáng và chiều là các giai đoạn cao điểm đi làm, đi học. `temp` và `atemp` phản ánh mức độ
        thoải mái khi di chuyển ngoài trời; thời tiết dễ chịu làm nhu cầu tăng. `weathersit`,
        `hum` và `windspeed` giúp mô hình nhận biết các điều kiện làm giảm nhu cầu như mưa,
        độ ẩm cao hoặc gió mạnh.
        """
    )

    top_raw = importance_df.sort_values("Importance", ascending=False).head(15)
    st.dataframe(
        top_raw.rename(columns={"Feature": "Feature gốc", "Importance": "Mức độ quan trọng"}),
        use_container_width=True,
        hide_index=True,
    )


def render_conclusion_tab() -> None:
    st.markdown('<p class="section-title">📋 Kết luận hệ thống</p>', unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown(
            """
            <div class="card">
                <span class="badge">🧭 Workflow Machine Learning</span>
                <h3>Quy trình thực hiện</h3>
                <p>1. Hiểu bài toán và dữ liệu Bike Sharing Dataset.</p>
                <p>2. Làm sạch dữ liệu, loại bỏ leakage và các cột không dùng: instant, dteday, casual, registered, yr, mnth.</p>
                <p>3. Phân tích EDA để tìm pattern theo giờ, thời tiết, mùa và ngày làm việc.</p>
                <p>4. One-hot encoding feature phân loại, train Linear Regression và Random Forest.</p>
                <p>5. Đánh giá bằng MAE, RMSE, R² Score và triển khai GUI bằng Streamlit.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div class="card">
                <span class="badge">💼 Business insight</span>
                <h3>Giá trị vận hành</h3>
                <p>Giờ cao điểm ảnh hưởng mạnh nhất đến nhu cầu thuê xe.</p>
                <p>Thời tiết xấu, độ ẩm cao và gió mạnh thường làm giảm số lượt thuê.</p>
                <p>Nhiệt độ dễ chịu giúp tăng nhu cầu di chuyển bằng xe đạp.</p>
                <p>Random Forest là model tối ưu nhất vì xử lý tốt quan hệ phi tuyến giữa các feature.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Model đề xuất", "Random Forest")
    with c2:
        st.metric("R² Score", "0.822")
    with c3:
        st.metric("Mục tiêu dự báo", "cnt")

    st.success(
        "Kết luận: hệ thống có thể hỗ trợ doanh nghiệp dự báo nhu cầu thuê xe theo giờ, tối ưu phân bổ xe tại trạm và chuẩn bị nguồn lực cho các khung giờ cao điểm."
    )


# =============================================================================
# MAIN APP
# =============================================================================

def main() -> None:
    inject_css()

    if "started" not in st.session_state:
        st.session_state.started = False

    if not st.session_state.started:
        render_welcome()
        return

    models, warnings = load_models()
    render_header(models)

    for warning in warnings:
        st.warning(f"⚠️ {warning}")

    if not models:
        st.error("Không load được model nào từ thư mục `models/`. Vui lòng kiểm tra file `.pkl` trước khi dự đoán.")
        return

    tab_predict, tab_compare, tab_importance, tab_conclusion = st.tabs(
        [
            "🧠 Dự đoán",
            "📊 So sánh mô hình",
            "📈 Feature Importance",
            "📋 Kết luận hệ thống",
        ]
    )

    with tab_predict:
        render_prediction_tab(models)

    with tab_compare:
        render_comparison_tab()

    with tab_importance:
        render_feature_importance_tab(models)

    with tab_conclusion:
        render_conclusion_tab()


if __name__ == "__main__":
    main()
