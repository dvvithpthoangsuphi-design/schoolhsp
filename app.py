import streamlit as st
import pandas as pd
import os
import yaml
import json
from io import StringIO, BytesIO

# Load config from file or environment variable
config_path = 'auth_config.yaml'
config = None

if os.path.exists(config_path):
    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
elif 'AUTH_CONFIG' in os.environ:
    config = yaml.safe_load(os.environ['AUTH_CONFIG'])
else:
    st.error("File YAML không tồn tại hoặc biến môi trường AUTH_CONFIG không được thiết lập!")

# Initialize session state
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

# Main area login (only show if not authenticated)
if not st.session_state.authenticated and config and 'credentials' in config and 'usernames' in config['credentials']:
    st.title("Đăng Nhập")
    username = st.text_input("Tên người dùng")
    password = st.text_input("Mật khẩu", type="password")
    if st.button("Đăng nhập"):
        if username in config['credentials']['usernames'] and config['credentials']['usernames'][username]['password'] == password:
            st.session_state.authenticated = True
            st.success("Đăng nhập thành công!")
            st.experimental_rerun()
        else:
            st.error("Tên người dùng hoặc mật khẩu không đúng!")
elif not config:
    st.error("Cấu hình không hợp lệ hoặc file YAML bị lỗi!")

# Sidebar for navigation
st.sidebar.title("Menu")

# Authentication in sidebar
if not st.session_state.authenticated and config and 'credentials' in config and 'usernames' in config['credentials']:
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")
    if st.sidebar.button("Login"):
        if username in config['credentials']['usernames'] and config['credentials']['usernames'][username]['password'] == password:
            st.session_state.authenticated = True
            st.sidebar.success(f"Welcome, {username}!")
            st.experimental_rerun()
        else:
            st.sidebar.error("Invalid username or password")
elif not config:
    st.sidebar.error("Cấu hình không khả dụng để đăng nhập!")

# Logout
if st.session_state.authenticated:
    st.sidebar.button("Logout", on_click=lambda: setattr(st.session_state, 'authenticated', False))

# Main content only if authenticated
if st.session_state.authenticated:
    st.title("AI Dự Báo Điểm Học Sinh")

    # Upload file with dynamic table structure support
    st.header("Tải Lên và Phân Tích Dữ Liệu Không Cấu Trúc")
    uploaded_file = st.file_uploader("Chọn file (CSV, Excel, JSON)", type=["csv", "xlsx", "xls", "json"])
    if uploaded_file is not None:
        file_extension = uploaded_file.name.split('.')[-1].lower()
        try:
            # Xử lý file theo định dạng với cấu trúc động
            if file_extension in ['csv']:
                df = pd.read_csv(uploaded_file, sep=None, engine='python', dtype=str, on_bad_lines='skip')
                st.success(f"Đã đọc thành công file CSV với {len(df)} hàng và {len(df.columns)} cột!")
            elif file_extension in ['xlsx', 'xls']:
                df = pd.read_excel(uploaded_file, engine='openpyxl' if file_extension == 'xlsx' else 'xlrd', dtype=str)
                st.success(f"Đã đọc thành công file Excel với {len(df)} hàng và {len(df.columns)} cột!")
            elif file_extension in ['json']:
                json_data = json.load(uploaded_file)
                df = pd.json_normalize(json_data) if isinstance(json_data, list) else pd.DataFrame([json_data])
                st.success(f"Đã đọc thành công file JSON với {len(df)} hàng và {len(df.columns)} cột!")
            else:
                st.error("Định dạng file không được hỗ trợ!")
                df = None

            if df is not None:
                # Hiển thị dữ liệu đầu
                st.write("**Dữ liệu đầu tiên (5 hàng):**")
                st.dataframe(df.head())

                # Tự động phân tích kiểu dữ liệu
                st.subheader("Phân Tích Tự Động Cấu Trúc Bảng")
                # Chuyển đổi kiểu dữ liệu tự động
                for col in df.columns:
                    if df[col].str.match(r'^-?\d*\.?\d+$').all():  # Kiểm tra số
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    elif pd.to_datetime(df[col], errors='coerce', infer_datetime_format=True).notna().all():
                        df[col] = pd.to_datetime(df[col], infer_datetime_format=True, errors='coerce')
                    # Nếu không phải số hoặc ngày, để nguyên là object

                # Hiển thị kiểu dữ liệu sau khi phân tích
                st.write("**Kiểu dữ liệu tự động phát hiện:**")
                st.write(df.dtypes)

                # Xử lý dữ liệu thiếu
                missing_data = df.isnull().sum()
                if missing_data.sum() > 0:
                    st.write("**Dữ liệu thiếu (số lượng và %):**")
                    missing_df = pd.DataFrame({'Missing Count': missing_data, 'Percentage': (missing_data / len(df)) * 100})
                    st.write(missing_df[missing_df['Missing Count'] > 0])
                    if st.button("Điền dữ liệu thiếu bằng trung bình (cột số)"):
                        numeric_cols = df.select_dtypes(include=['number']).columns
                        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
                        st.success("Đã điền dữ liệu thiếu!")
                        st.write("Dữ liệu sau khi điền (5 hàng):")
                        st.dataframe(df.head())
                else:
                    st.success("Không có dữ liệu thiếu!")

                # Thống kê mô tả cho cột số
                numeric_cols = df.select_dtypes(include=['number']).columns
                if len(numeric_cols) > 0:
                    st.write("**Thống kê mô tả (cột số):**")
                    st.write(df[numeric_cols].describe())

                # Thống kê phân loại cho cột văn bản
                categorical_cols = df.select_dtypes(include=['object']).columns
                if len(categorical_cols) > 0:
                    st.subheader("Thống Kê Phân Loại (Categorical)")
                    for col in categorical_cols:
                        if df[col].nunique() < 20:
                            st.write(f"**Cột '{col}' (số giá trị duy nhất: {df[col].nunique()}):**")
                            st.write(df[col].value_counts().head(10))

                # Biểu đồ phân bố cho cột số
                if len(numeric_cols) > 0:
                    st.subheader("Biểu Đồ Phân Bố")
                    for col in numeric_cols[:4]:  # Giới hạn 4 cột
                        fig = px.histogram(df, x=col, title=f"Phân Bố Cột '{col}'", nbins=20)
                        st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Lỗi khi xử lý file: {str(e)}. Vui lòng kiểm tra định dạng hoặc nội dung file.")
    else:
        st.info("Vui lòng tải lên file CSV, Excel, hoặc JSON để phân tích.")