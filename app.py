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
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload

# ===================== CONFIG & AUTH =====================
st.set_page_config(page_title="AI Dự Báo Điểm", layout="wide")

# Load config
config_path = 'auth_config.yaml'
config = None
if os.path.exists(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
elif 'AUTH_CONFIG' in os.environ:
    config = yaml.safe_load(os.environ['AUTH_CONFIG'])

# Session state
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'data_store' not in st.session_state:
    st.session_state.data_store = {}
if 'selected_dataset' not in st.session_state:
    st.session_state.selected_dataset = None
if 'drive_service' not in st.session_state:
    st.session_state.drive_service = None

# ===================== OAUTH 2.0 SETUP =====================
SCOPES = ['https://www.googleapis.com/auth/drive']
CLIENT_CONFIG = {
    "installed": {
        "client_id": os.environ.get("GOOGLE_CLIENT_ID"),
        "client_secret": os.environ.get("GOOGLE_CLIENT_SECRET"),
        "redirect_uris": ["http://localhost:8501"],
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token"
    }
}

def get_drive_service():
    if st.session_state.drive_service:
        return st.session_state.drive_service
    
    creds = None
    if 'google_token' in st.session_state:
        creds = Credentials.from_authorized_user_info(st.session_state.google_token, SCOPES)
    
    if not creds or not creds.valid:
        flow = InstalledAppFlow.from_client_config(CLIENT_CONFIG, SCOPES)
        creds = flow.run_local_server(port=8501)
        st.session_state.google_token = creds.to_json()
    
    service = build('drive', 'v3', credentials=creds)
    st.session_state.drive_service = service
    return service

# ===================== ĐĂNG NHẬP =====================
if not st.session_state.authenticated:
    st.title("Đăng Nhập")
    username = st.text_input("Tên người dùng")
    password = st.text_input("Mật khẩu", type="password")
    if st.button("Đăng nhập"):
        if config and username in config['credentials']['usernames']:
            if config['credentials']['usernames'][username]['password'] == password:
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Sai mật khẩu!")
        else:
            st.error("Sai tên đăng nhập!")
else:
    if st.sidebar.button("Đăng xuất"):
        st.session_state.authenticated = False
        st.rerun()

# ===================== GIAO DIỆN CHÍNH =====================
if st.session_state.authenticated:
    st.title("AI Dự Báo Điểm Học Sinh")
    st.markdown("<div style='background:linear-gradient(90deg,#2c3e50,#3498db);color:white;padding:20px;text-align:center;border-radius:10px;margin-bottom:20px'>Chào mừng đến với Hệ Thống AI Dự Báo Điểm Học Sinh</div>", unsafe_allow_html=True)

    try:
        drive_service = get_drive_service()
        FOLDER_ID = '1K6Z-huJcdphdM42o2NL3kvu6KY7asD_u'
    except Exception as e:
        st.error(f"Lỗi Google Drive: {e}")
        drive_service = None
        FOLDER_ID = None

    # ===================== UPLOAD FILE =====================
    st.subheader("Tải Lên Dữ Liệu Mới")
    uploaded_file = st.file_uploader("Chọn file", type=["csv", "xlsx", "xls", "json"])
    if uploaded_file and drive_service:
        file_name = uploaded_file.name
        try:
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
                media = MediaFileUpload(temp_path, resumable=True)
                file = drive_service.files().create(body=file_metadata, media_body=media, fields='id').execute()
                st.success(f"Đã upload lên Google Drive!")
            except Exception as e:
                st.error(f"Upload lỗi: {e}")
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
        except Exception as e:
            st.error(f"Lỗi đọc file: {e}")

    # ===================== TẢI TỪ DRIVE =====================
    drive_files = []
    if drive_service:
        try:
            results = drive_service.files().list(q=f"'{FOLDER_ID}' in parents", fields="files(id, name)").execute()
            drive_files = results.get('files', [])
            if drive_files:
                st.subheader("Dữ Liệu Từ Google Drive")
                for file in drive_files:
                    if file['name'] not in st.session_state.data_store:
                        try:
                            df = download_from_drive(drive_service, file['id'])
                            st.session_state.data_store[file['name']] = df
                            st.success(f"Đã tải '{file['name']}'")
                            st.session_state.selected_dataset = file['name']
                        except Exception as e:
                            st.error(f"Lỗi tải: {e}")
        except Exception as e:
            st.error(f"Lỗi truy cập Drive: {e}")

    # ===================== PHÂN TÍCH =====================
    if st.session_state.data_store and st.session_state.selected_dataset:
        df = st.session_state.data_store[st.session_state.selected_dataset]
        
        st.subheader("Bảng Điểm Học Sinh")
        if 'Họ tên' in df.columns or any('tên' in str(col).lower() for col in df.columns):
            # Tìm cột tên
            name_col = next((col for col in df.columns if 'tên' in str(col).lower()), None)
            df = df.rename(columns={name_col: 'Họ tên'}) if name_col else df
            
            point_cols = [col for col in df.columns if col not in ['Họ tên', 'Lớp']]
            for col in point_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df['ĐTB'] = df[point_cols].mean(axis=1).round(2)
            df = df.sort_values('ĐTB', ascending=False)

            display_df = df[['Họ tên', 'Lớp'] + point_cols].dropna(subset=['Họ tên'])
            st.dataframe(display_df, use_container_width=True, hide_index=True)

            # Top 3
            st.subheader("Top 3 Học Sinh")
            top3 = df.head(3)
            st.dataframe(top3[['Họ tên', 'Lớp', 'ĐTB']], use_container_width=True, hide_index=True)

            # Dự báo
            st.subheader("Dự Báo AI")
            pred_data = []
            for col in point_cols:
                mean_val = df[col].mean()
                pred = mean_val * 1.05 if mean_val > 0 else 0
                pred_data.append({"Môn": col, "Hiện tại": round(mean_val, 2), "Dự báo": round(pred, 2)})
            st.dataframe(pd.DataFrame(pred_data), use_container_width=True, hide_index=True)

        else:
            st.error("Không tìm thấy cột tên học sinh.")
            st.dataframe(df.head(5))

    # ===================== GỬI ZALO =====================
    st.markdown("---")
    st.subheader("Gửi Báo Cáo Qua Zalo")
    
    def gui_bao_cao_zalo_tu_dong():
        token = os.environ.get('ZALO_OA_TOKEN')
        if not token:
            st.error("Thiếu ZALO_OA_TOKEN!")
            return
        st.info("Chức năng đang phát triển...")

    if st.button("GỬI CHO TỪNG MẸ"):
        with st.spinner("Đang xử lý..."):
            gui_bao_cao_zalo_tu_dong()

    # ===================== FOOTER =====================
    st.markdown("""
    <div style='position:fixed;bottom:0;width:100%;background:linear-gradient(90deg,#3498db,#2c3e50);color:white;padding:10px;text-align:center'>
    © 2025 AI Dự Báo Điểm Học Sinh
    </div>
    """, unsafe_allow_html=True)