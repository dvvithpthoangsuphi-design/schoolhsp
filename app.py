import streamlit as st
import pandas as pd
import os
import yaml
import json
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.linear_model import LinearRegression
import requests
import time
from io import BytesIO
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload

# ===================== CONFIG =====================
st.set_page_config(page_title="AI Dự Báo Điểm", layout="wide")

# Load config
config = yaml.safe_load(os.environ.get('AUTH_CONFIG', '{}'))

# Session state
for key in ['authenticated', 'data_store', 'selected_dataset']:
    if key not in st.session_state:
        st.session_state[key] = False if key == 'authenticated' else {} if key == 'data_store' else None

# ===================== SERVICE ACCOUNT (RENDER ONLY - ENV VAR) =====================
SCOPES = ['https://www.googleapis.com/auth/drive']
SERVICE_ACCOUNT_JSON = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')

@st.cache_resource
def get_drive_service():
    if not SERVICE_ACCOUNT_JSON:
        st.error("Thiếu biến môi trường GOOGLE_APPLICATION_CREDENTIALS!")
        st.info("Vào Render → Settings → Environment Variables → Thêm key: GOOGLE_APPLICATION_CREDENTIALS")
        return None
    try:
        creds_info = json.loads(SERVICE_ACCOUNT_JSON)
        creds = Credentials.from_service_account_info(creds_info, scopes=SCOPES)
        service = build('drive', 'v3', credentials=creds)
        st.success("Kết nối Google Drive thành công!")
        return service
    except Exception as e:
        st.error(f"Lỗi Service Account: {e}")
        st.info("Kiểm tra JSON trong biến môi trường có đúng không?")
        return None

drive_service = get_drive_service()
if not drive_service:
    st.stop()

# ID THƯ MỤC TRONG SHARED DRIVE
FOLDER_ID = '1K6Z-huJcdphdM42o2NL3kvu6KY7asD_u'

# ===================== ĐĂNG NHẬP =====================
if not st.session_state.authenticated:
    st.title("Đăng Nhập")
    username = st.text_input("Tên người dùng")
    password = st.text_input("Mật khẩu", type="password")
    if st.button("Đăng nhập"):
        if config.get('credentials', {}).get('usernames', {}).get(username, {}).get('password') == password:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Sai thông tin!")
else:
    if st.sidebar.button("Đăng xuất"):
        st.session_state.authenticated = False
        st.rerun()

# ===================== GIAO DIỆN CHÍNH =====================
if st.session_state.authenticated:
    st.title("AI Dự Báo Điểm Học Sinh")
    st.markdown("<div style='background:linear-gradient(90deg,#2c3e50,#3498db);color:white;padding:20px;text-align:center;border-radius:10px'>HỆ THỐNG HOẠT ĐỘNG TRÊN RENDER</div>", unsafe_allow_html=True)

    # ===================== UPLOAD FILE =====================
    st.subheader("Tải Lên Dữ Liệu Mới")
    uploaded_file = st.file_uploader("Chọn file", type=["csv", "xlsx", "xls", "json"])
    if uploaded_file:
        file_name = uploaded_file.name
        try:
            # SỬA LỖI: if file_scratch = → if file_name.endswith
            if file_name.endswith('.csv'):
                df = pd.read_csv(uploaded_file, dtype=str)
            else:
                df = pd.read_excel(uploaded_file, engine='openpyxl', dtype=str)
            st.success(f"Đọc thành công {len(df)} hàng!")

            st.session_state.data_store[file_name] = df
            st.session_state.selected_dataset = file_name

            # Upload to Drive
            temp_path = f"/tmp/{file_name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            try:
                file_metadata = {'name': file_name, 'parents': [FOLDER_ID]}
                mimetype = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' if file_name.endswith('.xlsx') else 'text/csv'
                media = MediaFileUpload(temp_path, mimetype=mimetype, resumable=True)
                drive_service.files().create(body=file_metadata, media_body=media, fields='id').execute()
                st.success("Đã upload lên Google Drive!")
            except Exception as e:
                st.error(f"Upload lỗi: {e}")
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
        except Exception as e:
            st.error(f"Lỗi đọc file: {e}")

    # ===================== TẢI TỪ DRIVE =====================
    try:
        results = drive_service.files().list(q=f"'{FOLDER_ID}' in parents", fields="files(id, name)").execute()
        drive_files = results.get('files', [])
        if drive_files:
            st.subheader("Dữ Liệu Từ Google Drive")
            for file in drive_files:
                if file['name'] not in st.session_state.data_store:
                    request = drive_service.files().get_media(fileId=file['id'])
                    fh = BytesIO()
                    downloader = MediaIoBaseDownload(fh, request)
                    done = False
                    while not done:
                        status, done = downloader.next_chunk()
                    fh.seek(0)
                    df = pd.read_excel(fh) if file['name'].endswith('.xlsx') else pd.read_csv(fh)
                    st.session_state.data_store[file['name']] = df
                    st.success(f"Đã tải {file['name']}")
    except Exception as e:
        st.error(f"Lỗi tải: {e}")

    # ===================== PHÂN TÍCH + BIỂU ĐỒ =====================
    if st.session_state.selected_dataset:
        df = st.session_state.data_store[st.session_state.selected_dataset]

        # Tìm cột tên
        name_col = next((col for col in df.columns if 'tên' in str(col).lower()), None)
        if name_col:
            df = df.rename(columns={name_col: 'Họ tên'})
        else:
            st.error("Không tìm thấy cột tên!")
            st.stop()

        point_cols = [col for col in df.columns if col not in ['Họ tên', 'Lớp']]
        for col in point_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df['ĐTB'] = df[point_cols].mean(axis=1).round(2)
        df = df.sort_values('ĐTB', ascending=False)

        st.subheader("Bảng Điểm Học Sinh")
        st.dataframe(df[['Họ tên', 'Lớp'] + point_cols + ['ĐTB']], use_container_width=True)

        # BIỂU ĐỒ 1: TOP 3
        st.subheader("Top 3 Học Sinh")
        top3 = df.head(3)
        st.dataframe(top3[['Họ tên', 'Lớp', 'ĐTB']], use_container_width=True)
        fig_top = px.bar(top3.melt(id_vars='Họ tên', value_vars=point_cols), x='Họ tên', y='value', color='variable', title="Top 3 So Sánh Môn")
        st.plotly_chart(fig_top, use_container_width=True)

        # BIỂU ĐỒ 2: PHÂN BỐ ĐTB
        st.subheader("Phân Bố ĐTB")
        fig_hist = px.histogram(df, x='ĐTB', nbins=15, title="Phân Bố Điểm Trung Bình")
        fig_hist.add_vline(x=df['ĐTB'].mean(), line_dash="dash", line_color="red", annotation_text=f"TB: {df['ĐTB'].mean():.2f}")
        st.plotly_chart(fig_hist, use_container_width=True)

        # BIỂU ĐỒ 3: DỰ BÁO AI
        st.subheader("Dự Báo AI Kỳ Tới")
        pred_data = []
        for col in point_cols:
            scores = df[col].dropna()
            if len(scores) >= 2:
                X = np.arange(len(scores)).reshape(-1, 1)
                model = LinearRegression().fit(X, scores.values.reshape(-1, 1))
                pred = model.predict([[len(scores)]])[0][0]
            else:
                pred = scores.mean()
            pred_data.append({"Môn": col, "Hiện tại": scores.mean().round(2), "Dự báo": round(pred, 2)})
        pred_df = pd.DataFrame(pred_data)
        st.dataframe(pred_df, use_container_width=True)
        fig_pred = px.line(pred_df.melt(id_vars='Môn'), x='Môn', y='value', color='variable', title="Xu Hướng Điểm")
        st.plotly_chart(fig_pred, use_container_width=True)

        # BIỂU ĐỒ 4: TƯƠNG QUAN
        if len(point_cols) > 1:
            st.subheader("Tương Quan Giữa Các Môn")
            corr = df[point_cols].corr()
            fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu')
            st.plotly_chart(fig_corr, use_container_width=True)

    # ===================== GỬI ZALO =====================
    st.markdown("---")
    st.subheader("Gửi Báo Cáo Qua Zalo")
    if st.button("GỬI CHO TỪNG MẸ"):
        st.info("Chức năng đang phát triển...")

    # ===================== FOOTER =====================
    st.markdown("<div style='position:fixed;bottom:0;width:100%;background:#2c3e50;color:white;padding:10px;text-align:center'>© 2025 AI Dự Báo Điểm</div>", unsafe_allow_html=True)