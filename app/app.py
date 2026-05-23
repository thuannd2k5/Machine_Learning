"""
================================================================================
HỆ THỐNG DỰ ĐOÁN SỐ LƯỢNG XE ĐẠP THUÊ BẰNG MACHINE LEARNING
================================================================================
Ứng dụng Streamlit để dự đoán nhu cầu thuê xe đạp theo giờ
Sử dụng Linear Regression và Random Forest models
================================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════
# CẤU HÌNH STREAMLIT
# ═══════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="🚲 Dự Đoán Xe Đạp Thuê",
    page_icon="🚲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ═══════════════════════════════════════════════════════════════════════════
# ĐỊNH NGHĨA ĐẦU VÀO
# ═══════════════════════════════════════════════════════════════════════════

# Các cột categorical (được sử dụng trong training)
CATEGORICAL_COLS = [
    "season",
    "yr",
    "mnth",
    "hr",
    "holiday",
    "weekday",
    "workingday",
    "weathersit"
]

# Các cột numeric
NUMERIC_COLS = [
    "temp",
    "atemp",
    "hum",
    "windspeed"
]

# Tên tiếng Việt cho các features
FEATURE_LABELS = {
    "season": "🍂 Mùa trong năm",
    "yr": "📅 Năm",
    "mnth": "📆 Tháng",
    "hr": "⏰ Giờ",
    "holiday": "🎉 Ngày lễ",
    "weekday": "📅 Thứ trong tuần",
    "workingday": "💼 Ngày làm việc",
    "weathersit": "🌤️ Tình trạng thời tiết",
    "temp": "🌡️ Nhiệt độ",
    "atemp": "🌡️ Nhiệt độ cảm nhận",
    "hum": "💧 Độ ẩm",
    "windspeed": "💨 Tốc độ gió"
}

# Giá trị tương ứng cho các dropdown
SEASON_MAP = {
    "🌸 Xuân": 1,
    "☀️ Hè": 2,
    "🍂 Thu": 3,
    "❄️ Đông": 4
}

YEAR_MAP = {
    "2011": 0,
    "2012": 1
}

MONTH_MAP = {
    "Tháng 1": 1, "Tháng 2": 2, "Tháng 3": 3, "Tháng 4": 4,
    "Tháng 5": 5, "Tháng 6": 6, "Tháng 7": 7, "Tháng 8": 8,
    "Tháng 9": 9, "Tháng 10": 10, "Tháng 11": 11, "Tháng 12": 12
}

WEEKDAY_MAP = {
    "Chủ Nhật": 0, "Thứ Hai": 1, "Thứ Ba": 2, "Thứ Tư": 3,
    "Thứ Năm": 4, "Thứ Sáu": 5, "Thứ Bảy": 6
}

WEATHER_MAP = {
    "☀️ Trời quang / Ít mây": 1,
    "🌥️ Sương mù / Nhiều mây": 2,
    "🌧️ Mưa nhẹ / Tuyết nhẹ": 3,
    "⛈️ Mưa lớn / Tuyết / Sương mù dày": 4
}

# ═══════════════════════════════════════════════════════════════════════════
# HÀM LOAD MODELS
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_resource
def load_models():
    """Load các models được train từ file .pkl"""
    models = {}
    
    # Đường dẫn đến folder models
    models_dir = Path(__file__).parent.parent / "models"
    
    # Load Linear Regression
    lr_path = models_dir / "linear_regression.pkl"
    if lr_path.exists():
        models["Linear Regression"] = joblib.load(lr_path)
    else:
        st.warning(f"⚠️ Không tìm thấy: {lr_path}")
    
    # Load Random Forest
    rf_path = models_dir / "random_forest.pkl"
    if rf_path.exists():
        models["Random Forest"] = joblib.load(rf_path)
    else:
        st.warning(f"⚠️ Không tìm thấy: {rf_path}")
    
    return models

# ═══════════════════════════════════════════════════════════════════════════
# HÀM TIỀN XỬ LÝ DỮ LIỆU
# ═══════════════════════════════════════════════════════════════════════════

def preprocess_input(input_data):
    """
    Tiền xử lý input data:
    - Tạo DataFrame
    - One-hot encoding
    - Sắp xếp cột theo đúng thứ tự training
    """
    # Tạo DataFrame từ input
    df_input = pd.DataFrame([input_data])
    
    # Chuyển đổi kiểu dữ liệu
    for col in CATEGORICAL_COLS:
        df_input[col] = df_input[col].astype("category")
    
    # One-hot encoding (drop_first=True như lúc training)
    df_encoded = pd.get_dummies(
        df_input,
        columns=CATEGORICAL_COLS,
        drop_first=True
    )
    
    return df_encoded

def get_feature_order():
    """
    Lấy thứ tự features như lúc training
    Tạo bằng cách one-hot encode toàn bộ dataset một lần
    """
    try:
        data_dir = Path(__file__).parent.parent / "data"
        df = pd.read_csv(data_dir / "hour.csv")
        
        # Drop các cột không cần
        df = df.drop(columns=["instant", "dteday", "casual", "registered"], errors="ignore")
        
        # Convert categorical
        for col in CATEGORICAL_COLS:
            df[col] = df[col].astype("category")
        
        # Xử lý weathersit = 4
        if 4 in df["weathersit"].values:
            df["weathersit"] = df["weathersit"].replace(4, 3)
            df["weathersit"] = df["weathersit"].cat.remove_unused_categories()
        
        # One-hot encoding
        df_encoded = pd.get_dummies(
            df,
            columns=CATEGORICAL_COLS,
            drop_first=True
        )
        
        # Trả về thứ tự features (trừ target)
        features = [col for col in df_encoded.columns if col != "cnt"]
        return features
    except Exception as e:
        st.error(f"Lỗi khi đọc dataset: {e}")
        return None

# ═══════════════════════════════════════════════════════════════════════════
# HÀM DMaps VÀ REVERSE MAPS
# ═══════════════════════════════════════════════════════════════════════════

def get_reverse_maps():
    """Tạo reverse mapping cho các features"""
    reverse_maps = {
        "season": {v: k for k, v in SEASON_MAP.items()},
        "yr": {v: k for k, v in YEAR_MAP.items()},
        "mnth": {v: k for k, v in MONTH_MAP.items()},
        "weekday": {v: k for k, v in WEEKDAY_MAP.items()},
        "weathersit": {v: k for k, v in WEATHER_MAP.items()}
    }
    return reverse_maps

# ═══════════════════════════════════════════════════════════════════════════
# HÀM HIỂN THỊ KẾT QUẢ
# ═══════════════════════════════════════════════════════════════════════════

def display_prediction_results(input_data, predictions, models):
    """Hiển thị kết quả dự đoán"""
    
    # Tiêu đề
    st.markdown("---")
    st.markdown("## 📊 KẾT QUẢ DỰ ĐOÁN")
    st.markdown("---")
    
    # Layout 3 cột cho 3 metric chính
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if "Linear Regression" in predictions:
            pred_lr = predictions["Linear Regression"][0]
            st.metric(
                label="📈 Linear Regression",
                value=f"{pred_lr:.0f} xe",
                delta=None
            )
    
    with col2:
        if "Random Forest" in predictions:
            pred_rf = predictions["Random Forest"][0]
            st.metric(
                label="🌲 Random Forest",
                value=f"{pred_rf:.0f} xe",
                delta=None
            )
    
    with col3:
        # Tính trung bình
        all_preds = [v[0] for v in predictions.values()]
        avg_pred = np.mean(all_preds)
        st.metric(
            label="📊 Dự đoán trung bình",
            value=f"{avg_pred:.0f} xe"
        )
    
    st.markdown("---")
    
    # Bảng chi tiết dự đoán
    st.subheader("📋 Chi tiết dự đoán từng model:")
    
    prediction_data = []
    for model_name, pred_value in predictions.items():
        prediction_data.append({
            "Model": model_name,
            "Dự đoán (xe)": f"{pred_value[0]:.2f}"
        })
    
    df_predictions = pd.DataFrame(prediction_data)
    st.dataframe(df_predictions, use_container_width=True)
    
    # Biểu đồ so sánh
    st.subheader("📈 Biểu đồ so sánh dự đoán:")
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    model_names = list(predictions.keys())
    pred_values = [predictions[m][0] for m in model_names]
    
    colors = ["#FF6B6B", "#4ECDC4"]
    bars = ax.bar(model_names, pred_values, color=colors, edgecolor="black", linewidth=2)
    
    # Thêm giá trị trên các thanh
    for bar, val in zip(bars, pred_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.0f}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_ylabel("Số lượng xe thuê", fontsize=11, fontweight="bold")
    ax.set_title("So sánh dự đoán từ các models", fontsize=13, fontweight="bold", pad=20)
    ax.set_ylim(0, max(pred_values) * 1.15)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Model tốt nhất
    best_model = max(predictions.keys(), key=lambda x: predictions[x][0])
    best_value = predictions[best_model][0]
    
    st.success(f"✅ **Model dự đoán cao nhất:** {best_model} ({best_value:.0f} xe)")

# ═══════════════════════════════════════════════════════════════════════════
# HÀM HIỂN THỊ SO SÁNH MÔ HÌNH
# ═══════════════════════════════════════════════════════════════════════════

def display_model_comparison():
    """Hiển thị bảng so sánh hiệu năng models"""
    st.markdown("---")
    st.markdown("## 📊 SO SÁNH HIỆU NĂNG MÔ HÌNH")
    st.markdown("---")
    
    # Dữ liệu so sánh (từ notebook)
    comparison_data = {
        "Model": ["Linear Regression", "Random Forest"],
        "MAE": [33.41, 18.43],
        "RMSE": [45.87, 25.14],
        "R² Score": [0.65, 0.84]
    }
    
    df_comparison = pd.DataFrame(comparison_data)
    
    # Hiển thị dưới dạng table đẹp
    st.dataframe(df_comparison, use_container_width=True, hide_index=True)
    
    # Nhận xét
    st.info(
        """
        💡 **Nhận xét:**
        - **Random Forest** vượt trội hơn với R² = 0.84 (tốt hơn 29%)
        - **Linear Regression** đơn giản nhưng MAE cao (33.41)
        - **Random Forest** có MAE thấp (18.43) - dự đoán chính xác hơn
        - Khuyến nghị: Sử dụng **Random Forest** để dự đoán
        """
    )

# ═══════════════════════════════════════════════════════════════════════════
# HÀM HIỂN THỊ FEATURE IMPORTANCE
# ═══════════════════════════════════════════════════════════════════════════

def display_feature_importance(rf_model):
    """Hiển thị feature importance của Random Forest"""
    st.markdown("---")
    st.markdown("## 🎯 TẦM QUAN TRỌNG CỦA CÁC FEATURES (Random Forest)")
    st.markdown("---")
    
    try:
        # Lấy feature names từ training data
        feature_names = get_feature_order()
        
        if feature_names is None:
            st.error("Không thể lấy danh sách features")
            return
        
        # Lấy importances
        importances = rf_model.feature_importances_
        
        # Tạo DataFrame
        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances
        }).sort_values("Importance", ascending=False).head(15)
        
        # Hiển thị bảng
        st.dataframe(importance_df, use_container_width=True, hide_index=True)
        
        # Biểu đồ bar
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.barh(range(len(importance_df)), importance_df["Importance"], color="#4ECDC4", edgecolor="black")
        ax.set_yticks(range(len(importance_df)))
        ax.set_yticklabels(importance_df["Feature"])
        ax.set_xlabel("Mức độ quan trọng", fontweight="bold")
        ax.set_title("Top 15 Features Quan Trọng Nhất", fontsize=13, fontweight="bold", pad=20)
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        st.pyplot(fig)
        
    except Exception as e:
        st.error(f"Lỗi khi hiển thị feature importance: {e}")

# ═══════════════════════════════════════════════════════════════════════════
# GIAO DIỆN CHÍNH
# ═══════════════════════════════════════════════════════════════════════════

def main():
    """Hàm chính - giao diện ứng dụng"""
    
    # ─────────────────────────────────────────────────────────────────────
    # HEADER
    # ─────────────────────────────────────────────────────────────────────
    st.markdown("""
        <div style="text-align: center; padding: 30px 0;">
            <h1 style="color: #FF6B6B; font-size: 3em;">🚲 HỆ THỐNG DỰ ĐOÁN XE ĐẠP THUÊ</h1>
            <h3 style="color: #4ECDC4; margin-top: -15px;">Dự Đoán Nhu Cầu Thuê Xe Bằng Machine Learning</h3>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ─────────────────────────────────────────────────────────────────────
    # MÔ TẢ NGẮN
    # ─────────────────────────────────────────────────────────────────────
    st.markdown("""
        <div style="background-color: #F0F8FF; padding: 20px; border-radius: 10px; border-left: 5px solid #4ECDC4;">
            <h4>📖 Giới Thiệu</h4>
            <p>
                Hệ thống này sử dụng các mô hình Machine Learning (Linear Regression và Random Forest) 
                để dự đoán số lượng xe đạp sẽ được thuê trong một giờ cụ thể dựa trên các điều kiện 
                thời tiết, thời gian, và các yếu tố môi trường khác.
            </p>
            <p>
                <strong>Mục đích:</strong> Giúp công ty xe đạp thuê dự báo nhu cầu, tối ưu hóa phân bố 
                xe, và cải thiện kế hoạch hoạt động.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ─────────────────────────────────────────────────────────────────────
    # LOAD MODELS
    # ─────────────────────────────────────────────────────────────────────
    models = load_models()
    
    if not models:
        st.error("❌ Lỗi: Không thể load bất kỳ model nào. Vui lòng kiểm tra folder 'models/'")
        return
    
    st.success(f"✅ Đã load thành công {len(models)} model")
    
    # ─────────────────────────────────────────────────────────────────────
    # SIDEBAR - INPUT NGƯỜI DÙNG
    # ─────────────────────────────────────────────────────────────────────
    st.sidebar.markdown("# 📋 NHẬP DỮ LIỆU")
    st.sidebar.markdown("---")
    
    with st.sidebar:
        # Categorical inputs
        st.subheader("🔖 Thông tin phân loại")
        
        season = st.selectbox(
            "🍂 Mùa trong năm",
            list(SEASON_MAP.keys())
        )
        
        yr = st.selectbox(
            "📅 Năm",
            list(YEAR_MAP.keys())
        )
        
        mnth = st.selectbox(
            "📆 Tháng",
            list(MONTH_MAP.keys())
        )
        
        hr = st.slider(
            "⏰ Giờ (0-23)",
            min_value=0,
            max_value=23,
            value=12,
            step=1
        )
        
        holiday = st.selectbox(
            "🎉 Có phải ngày lễ?",
            ["Không", "Có"]
        )
        
        weekday = st.selectbox(
            "📅 Thứ trong tuần",
            list(WEEKDAY_MAP.keys())
        )
        
        workingday = st.selectbox(
            "💼 Ngày làm việc?",
            ["Không (Cuối tuần/Lễ)", "Có"]
        )
        
        weathersit = st.selectbox(
            "🌤️ Tình trạng thời tiết",
            list(WEATHER_MAP.keys())
        )
        
        st.divider()
        
        # Numeric inputs
        st.subheader("📊 Thông tin liên tục")
        
        temp = st.slider(
            "🌡️ Nhiệt độ (0-1 chuẩn hóa)",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05
        )
        
        atemp = st.slider(
            "🌡️ Nhiệt độ cảm nhận (0-1 chuẩn hóa)",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05
        )
        
        hum = st.slider(
            "💧 Độ ẩm (0-1 chuẩn hóa)",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05
        )
        
        windspeed = st.slider(
            "💨 Tốc độ gió (0-1 chuẩn hóa)",
            min_value=0.0,
            max_value=1.0,
            value=0.2,
            step=0.05
        )
    
    # ─────────────────────────────────────────────────────────────────────
    # NỘI DUNG CHÍNH
    # ─────────────────────────────────────────────────────────────────────
    
    # Nút dự đoán
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        predict_button = st.button(
            "🔮 DÀNH DỰ ĐOÁN",
            use_container_width=True,
            type="primary"
        )
    
    if predict_button:
        # Chuẩn bị input data
        input_data = {
            "season": SEASON_MAP[season],
            "yr": YEAR_MAP[yr],
            "mnth": MONTH_MAP[mnth],
            "hr": hr,
            "holiday": 1 if holiday == "Có" else 0,
            "weekday": WEEKDAY_MAP[weekday],
            "workingday": 1 if workingday == "Có" else 0,
            "weathersit": WEATHER_MAP[weathersit],
            "temp": temp,
            "atemp": atemp,
            "hum": hum,
            "windspeed": windspeed
        }
        
        # Tiền xử lý
        with st.spinner("⏳ Đang xử lý dữ liệu..."):
            try:
                X_input = preprocess_input(input_data)
                
                # Sắp xếp cột theo đúng thứ tự
                feature_order = get_feature_order()
                if feature_order:
                    # Thêm các cột bị thiếu với giá trị 0
                    for col in feature_order:
                        if col not in X_input.columns:
                            X_input[col] = 0
                    
                    # Sắp xếp lại cột
                    X_input = X_input[feature_order]
                
                # Dự đoán
                predictions = {}
                for model_name, model in models.items():
                    pred = model.predict(X_input)
                    predictions[model_name] = pred
                
                # Hiển thị input
                st.markdown("## 📥 Dữ Liệu Nhập Vào")
                st.markdown("---")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Thông tin phân loại:**")
                    st.write(f"- Mùa: {season}")
                    st.write(f"- Năm: {yr}")
                    st.write(f"- Tháng: {mnth}")
                    st.write(f"- Giờ: {hr}:00")
                    st.write(f"- Ngày lễ: {holiday}")
                    st.write(f"- Thứ: {weekday}")
                
                with col2:
                    st.write("**Điều kiện thời tiết & môi trường:**")
                    st.write(f"- Thời tiết: {weathersit}")
                    st.write(f"- Ngày làm việc: {workingday}")
                    st.write(f"- Nhiệt độ: {temp:.2f}")
                    st.write(f"- Cảm nhận nhiệt độ: {atemp:.2f}")
                    st.write(f"- Độ ẩm: {hum:.2f}")
                    st.write(f"- Tốc độ gió: {windspeed:.2f}")
                
                # Hiển thị kết quả dự đoán
                display_prediction_results(input_data, predictions, models)
                
                # Hiển thị so sánh model
                display_model_comparison()
                
                # Hiển thị feature importance (nếu có Random Forest)
                if "Random Forest" in models:
                    display_feature_importance(models["Random Forest"])
                
            except Exception as e:
                st.error(f"❌ Lỗi trong quá trình dự đoán: {str(e)}")
                st.error("Vui lòng kiểm tra lại dữ liệu hoặc models")
    
    # ─────────────────────────────────────────────────────────────────────
    # FOOTER - KẾT LUẬN
    # ─────────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("""
        <div style="background-color: #FFE5E5; padding: 20px; border-radius: 10px; border-left: 5px solid #FF6B6B;">
            <h4>📌 Kết Luận</h4>
            <p>
                <strong>Dự án:</strong> Bike Rental Demand Prediction Using Machine Learning
            </p>
            <p>
                <strong>Mục tiêu:</strong> Dự đoán nhu cầu thuê xe đạp theo giờ để tối ưu hóa 
                hoạt động kinh doanh và phân bố tài nguyên hiệu quả.
            </p>
            <p>
                <strong>Kỹ thuật:</strong> Sử dụng Linear Regression và Random Forest Regression 
                để so sánh hiệu năng và chọn mô hình tốt nhất.
            </p>
            <p>
                <strong>Kết quả:</strong> Random Forest cho kết quả dự đoán chính xác hơn 
                với R² Score = 0.84 và MAE = 18.43
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("""
        <div style="text-align: center; padding: 20px; color: #666;">
            <p>🚲 <strong>Bike Rental Demand Prediction System</strong> 🚲</p>
            <p><em>Powered by Machine Learning | Streamlit</em></p>
        </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# CHẠY ỨNG DỤNG
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    main()
