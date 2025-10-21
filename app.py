import streamlit as st
import pandas as pd
import os
import yaml
import json
import plotly.express as px
from io import StringIO, BytesIO
from sklearn.linear_model import LinearRegression
import numpy as np
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

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
if 'data_store' not in st.session_state:
    st.session_state.data_store = {}
if 'selected_dataset' not in st.session_state:
    st.session_state.selected_dataset = None

# Google Drive setup
SCOPES = ['https://www.googleapis.com/auth/drive']
creds = Credentials.from_service_account_file('credentials.json', scopes=SCOPES)
drive_service = build('drive', 'v3', credentials=creds)
FOLDER_ID = 'your_folder_id_here'  # Thay bằng ID thư mục Google Drive

def upload_to_drive(file, file_name):
    file_metadata = {'name': file_name, 'parents': [FOLDER_ID]}
    media = MediaFileUpload(file, mimetype='application/vnd.google-apps.spreadsheet')
    file = drive_service.files().create(body=file_metadata, media_body=media, fields='id').execute()
    st.success(f"Đã tải '{file_name}' lên Google Drive với ID: {file.get('id')}")

def download_from_drive(file_id):
    request = drive_service.files().get_media(fileId=file_id)
    fh = BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while done is False:
        status, done = downloader.next_chunk()
    fh.seek(0)
    return pd.read_excel(fh)

# Main area login
if not st.session_state.authenticated and config and 'credentials' in config and 'usernames' in config['credentials']:
    st.title("Đăng Nhập")
    username = st.text_input("Tên người dùng")
    password = st.text_input("Mật khẩu", type="password")
    if st.button("Đăng nhập"):
        if username in config['credentials']['usernames'] and config['credentials']['usernames'][username]['password'] == password:
            st.session_state.authenticated = True
            st.success("Đăng nhập thành công!")
            st.rerun()
        else:
            st.error("Tên người dùng hoặc mật khẩu không đúng!")
elif not config:
    st.error("Cấu hình không hợp lệ hoặc file YAML bị lỗi!")

# Sidebar for navigation
st.sidebar.title("Menu")
if not st.session_state.authenticated and config and 'credentials' in config and 'usernames' in config['credentials']:
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")
    if st.sidebar.button("Login"):
        if username in config['credentials']['usernames'] and config['credentials']['usernames'][username]['password'] == password:
            st.session_state.authenticated = True
            st.sidebar.success(f"Welcome, {username}!")
            st.rerun()
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

    # Banner
    st.markdown(
        """
        <style>
        .banner {
            background: linear-gradient(90deg, #2c3e50, #3498db);
            color: white;
            padding: 20px;
            text-align: center;
            font-size: 28px;
            font-weight: bold;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        }
        </style>
        <div class="banner">
            Chào mừng đến với Hệ Thống AI Dự Báo Điểm Học Sinh
        </div>
        """,
        unsafe_allow_html=True
    )

    # Data Store Management
    st.header("Quản Lý Kho Dữ Liệu")
    if not st.session_state.data_store:
        st.warning("Kho dữ liệu trống. Vui lòng tải file để bắt đầu.")
    else:
        st.write("**Danh sách dữ liệu trong kho:**")
        st.session_state.selected_dataset = st.selectbox("Chọn tập dữ liệu để phân tích", list(st.session_state.data_store.keys()), index=0 if st.session_state.selected_dataset in st.session_state.data_store else 0)

    # Upload new file to data store and Google Drive
    st.subheader("Tải Lên Dữ Liệu Mới")
    uploaded_file = st.file_uploader("Chọn file (CSV, Excel, JSON)", type=["csv", "xlsx", "xls", "json"], help="Kéo và thả file lên đây (tối đa 200MB)")
    if uploaded_file is not None:
        file_name = uploaded_file.name
        file_extension = file_name.split('.')[-1].lower()
        try:
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
                st.session_state.data_store[file_name] = df
                st.success(f"Đã thêm '{file_name}' vào kho dữ liệu!")
                st.session_state.selected_dataset = file_name
                # Upload to Google Drive
                with open(file_name, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                upload_to_drive(file_name, file_name)
                os.remove(file_name)  # Xóa file tạm
        except ValueError as ve:
            st.error(f"Lỗi định dạng file Excel: {str(ve)}. Vui lòng kiểm tra hoặc convert sang .xlsx.")
        except Exception as e:
            st.error(f"Lỗi khi xử lý file: {str(e)}. Vui lòng kiểm tra định dạng hoặc nội dung file.")

    # Load all files from Google Drive folder
    query = f"'{FOLDER_ID}' in parents"
    results = drive_service.files().list(q=query, fields="files(id, name)").execute()
    drive_files = results.get('files', [])
    if drive_files:
        st.subheader("Dữ Liệu Từ Google Drive")
        for file in drive_files:
            if file['name'] not in st.session_state.data_store:
                df = download_from_drive(file['id'])
                st.session_state.data_store[file['name']] = df
                st.success(f"Đã tải '{file['name']}' từ Google Drive vào kho dữ liệu!")

    # Analysis section
    if st.session_state.data_store and st.session_state.selected_dataset:
        df = st.session_state.data_store[st.session_state.selected_dataset]

        # Hiển thị dữ liệu
        st.subheader("Dữ liệu (tùy chọn hiển thị)")
        rows_to_show = st.slider("Số hàng để hiển thị", min_value=5, max_value=len(df), value=5)
        st.dataframe(df.head(rows_to_show), use_container_width=True, hide_index=True)

        # Phân tích kiểu dữ liệu
        st.subheader("Phân Tích Tự Động Cấu Trúc Bảng")
        numeric_cols = []
        for col in df.columns:
            try:
                if df[col].str.match(r'^-?\d*\.?\d+$').all() or df[col].str.replace('.', '', regex=False).str.isnumeric().all():
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    numeric_cols.append(col)
                elif pd.to_datetime(df[col], errors='coerce', infer_datetime_format=True).notna().all():
                    df[col] = pd.to_datetime(df[col], infer_datetime_format=True, errors='coerce')
                else:
                    df[col] = df[col].astype(str)
                    if df[col].str.contains(r'[^a-zA-Z0-9\s]', na=False).any():
                        st.warning(f"Cột '{col}' chứa ký tự đặc biệt, đã chuyển thành text.")
            except Exception as e:
                st.warning(f"Không thể xử lý cột '{col}': {str(e)}. Đã chuyển thành text.")
                df[col] = df[col].astype(str)

        st.write("**Kiểu dữ liệu tự động phát hiện:**")
        st.dataframe(pd.DataFrame({'Column': df.columns, 'Data Type': df.dtypes}), use_container_width=True)

        # Xử lý dữ liệu thiếu
        missing_data = df.isnull().sum()
        if missing_data.sum() > 0:
            st.write("**Dữ liệu thiếu (số lượng và %):**")
            missing_df = pd.DataFrame({'Missing Count': missing_data, 'Percentage': (missing_data / len(df)) * 100})
            st.dataframe(missing_df[missing_df['Missing Count'] > 0], use_container_width=True)
            if st.button("Điền dữ liệu thiếu bằng trung bình (cột số)"):
                if numeric_cols:
                    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
                    st.session_state.data_store[st.session_state.selected_dataset] = df
                    st.success("Đã điền dữ liệu thiếu!")
                    st.dataframe(df.head(rows_to_show), use_container_width=True, hide_index=True)
                else:
                    st.warning("Không có cột số để điền dữ liệu thiếu!")
        else:
            st.success("Không có dữ liệu thiếu!")

        # Thống kê mô tả
        if numeric_cols:
            st.write("**Thống kê mô tả (tất cả cột số):**")
            st.dataframe(df[numeric_cols].describe(), use_container_width=True)

        # Thống kê phân loại
        categorical_cols = df.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            st.subheader("Thống Kê Phân Loại (Categorical)")
            for col in categorical_cols:
                st.write(f"**Cột '{col}' (số giá trị duy nhất: {df[col].nunique()}):**")
                st.dataframe(pd.DataFrame({'Value': df[col].value_counts().index, 'Count': df[col].value_counts().values}), use_container_width=True)

        # Biểu đồ phân bố
        if numeric_cols:
            st.subheader("Biểu Đồ Phân Bố")
            for col in numeric_cols:
                fig = px.histogram(df, x=col, title=f"Phân Bố Cột '{col}'", nbins=20, color_discrete_sequence=['#3498db'])
                st.plotly_chart(fig, use_container_width=True)

        # Phân tích tương quan
        if len(numeric_cols) > 1:
            st.subheader("Phân Tích Tương Quan Giữa Các Cột Điểm")
            correlation_matrix = df[numeric_cols].corr()
            fig = px.imshow(correlation_matrix, text_auto=True, aspect="auto", color_continuous_scale='RdBu')
            fig.update_layout(title="Ma trận Tương Quan")
            st.plotly_chart(fig, use_container_width=True)

        # Dự đoán và cảnh báo với học hỏi
        if numeric_cols and len(df) > 1:
            st.subheader("Dự Đoán và Cảnh Báo Tình Hình Học Tập")
            selected_col = st.selectbox("Chọn cột điểm để dự đoán", numeric_cols)
            if 'Ngày thi' in df.columns:
                X = (df['Ngày thi'] - df['Ngày thi'].min()).dt.days.values.reshape(-1, 1)
            else:
                X = np.arange(len(df)).reshape(-1, 1)
            y = df[selected_col].values.reshape(-1, 1)
            model = LinearRegression()
            model.fit(X, y)
            next_index = len(df) if 'Ngày thi' not in df.columns else (pd.to_datetime('2025-12-31') - df['Ngày thi'].min()).days
            predicted_score = model.predict([[next_index]])[0][0]
            st.write(f"Dự đoán điểm {selected_col} cho kỳ tiếp theo: {predicted_score:.2f}")

            threshold = 5.0
            if df[selected_col].min() < threshold or predicted_score < threshold:
                st.warning(f"Cảnh báo: Điểm {selected_col} có giá trị thấp nhất {df[selected_col].min():.2f} hoặc dự đoán {predicted_score:.2f} dưới {threshold}. Cần hỗ trợ học sinh!")
            elif df[selected_col].mean() < 6.5:
                st.warning(f"Cảnh báo: Điểm trung bình {selected_col} là {df[selected_col].mean():.2f}, cần chú ý cải thiện!")

    # Footer
    st.markdown(
        """
        <style>
        .footer {
            background: linear-gradient(90deg, #3498db, #2c3e50);
            color: white;
            padding: 10px;
            text-align: center;
            font-size: 14px;
            position: fixed;
            bottom: 0;
            width: 100%;
            border-radius: 5px 5px 0 0;
            box-shadow: 0 -4px 8px rgba(0, 0, 0, 0.2);
        }
        </style>
        <div class="footer">
            © 2025 AI Dự Báo Điểm Học Sinh 
        </div>
        """,
        unsafe_allow_html=True
    )