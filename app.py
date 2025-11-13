# app.py
import streamlit as st
import pandas as pd
import json
import os
import yaml
import plotly.express as px
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
import requests
import time
from io import BytesIO
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload

# ===================== CONFIG =====================
st.set_page_config(page_title="AI Dự Báo Điểm", layout="wide")

# Load config.yaml
if not os.path.exists('config.yaml'):
    st.error("Không tìm thấy config.yaml!")
    st.stop()
with open('config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# Session state
for key in ['authenticated', 'drive_service', 'folder_ids', 'ai2_result', 'username']:
    if key not in st.session_state:
        st.session_state[key] = False if key in ['authenticated'] else None

# ===================== GOOGLE DRIVE =====================
SCOPES = ['https://www.googleapis.com/auth/drive']
SERVICE_ACCOUNT_JSON = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')

@st.cache_resource
def get_drive_service():
    if not SERVICE_ACCOUNT_JSON:
        st.error("Thiếu GOOGLE_APPLICATION_CREDENTIALS trong Environment!")
        return None
    try:
        creds_info = json.loads(SERVICE_ACCOUNT_JSON)
        creds = Credentials.from_service_account_info(creds_info, scopes=SCOPES)
        service = build('drive', 'v3', credentials=creds)
        st.success("Kết nối Google Drive thành công!")
        return service
    except Exception as e:
        st.error(f"Lỗi Service Account: {e}")
        return None

drive_service = get_drive_service()
if not drive_service:
    st.stop()

# ===================== FOLDERS =====================
FOLDER_STRUCTURE = {
    "RAW_DATA": "1K6Z-huJcdphdM42o2NL3kvu6KY7asD_u",
    "AI1_OUTPUT": None, "AI2_OUTPUT": None, "AI3_REPORTS": None
}

def ensure_folders():
    if st.session_state.folder_ids:
        return st.session_state.folder_ids
    folders = {}
    names = ["AI1_Cleaned_Data", "AI2_Analysis", "AI3_Reports"]
    parent = FOLDER_STRUCTURE["RAW_DATA"]
    for name in names:
        q = f"name='{name}' and mimeType='application/vnd.google-apps.folder' and '{parent}' in parents and trashed=false"
        res = drive_service.files().list(q=q, fields="files(id)").execute()
        files = res.get('files', [])
        if files:
            folder_id = files[0]['id']
        else:
            file = drive_service.files().create(
                body={'name': name, 'mimeType': 'application/vnd.google-apps.folder', 'parents': [parent]},
                fields='id'
            ).execute()
            folder_id = file['id']
        folders[name.replace(" ", "_").upper()] = folder_id
    FOLDER_STRUCTURE.update(folders)
    st.session_state.folder_ids = FOLDER_STRUCTURE
    st.info(f"Thư mục đã sẵn sàng: AI1_OUTPUT = {folders['AI1_CLEANED_DATA']}")
    return FOLDER_STRUCTURE

# ===================== LOGIN =====================
if not st.session_state.authenticated:
    st.title("Đăng Nhập Hệ Thống")
    username = st.text_input("Tên đăng nhập")
    password = st.text_input("Mật khẩu", type="password")
    if st.button("Đăng nhập"):
        user = config['credentials']['usernames'].get(username)
        if user and user['password'] == password:
            st.session_state.authenticated = True
            st.session_state.username = username
            st.rerun()
        else:
            st.error("Sai tên đăng nhập hoặc mật khẩu!")
else:
    st.sidebar.write(f"Xin chào: **{config['credentials']['usernames'][st.session_state.username]['name']}**")
    if st.sidebar.button("Đăng xuất"):
        st.session_state.clear()
        st.rerun()

# ===================== MAIN =====================
if st.session_state.authenticated:
    st.title("AI Xử Lý Điểm Học Sinh")
    st.markdown("**Tự động: Xử lý → Phân tích → Biểu đồ → Gửi Zalo**")

    folders = ensure_folders()

    # AI 1: Xử lý dữ liệu
        # AI 1: Xử lý dữ liệu (HỖ TRỢ CẢ GOOGLE SHEETS & FILE THẬT)
        # AI 1: Xử lý dữ liệu (TÌM CỘT THÔNG MINH + BÁO LỖI RÕ RÀNG)
    def run_ai1():
        with st.spinner("AI 1: Đang xử lý file mới nhất..."):
            try:
                # LẤY FILE MỚI NHẤT
                res = drive_service.files().list(
                    q=f"'{folders['RAW_DATA']}' in parents and trashed=false",
                    fields="files(id, name, mimeType, modifiedTime)",
                    orderBy="modifiedTime desc"
                ).execute()
                files = res.get('files', [])
                if not files:
                    st.warning("Không có file nào trong thư mục RAW_DATA!")
                    return False

                file = files[0]
                file_id = file['id']
                file_name = file['name']
                mime_type = file['mimeType']

                fh = BytesIO()

                # === HỖ TRỢ GOOGLE SHEETS ===
                if mime_type == 'application/vnd.google-apps.spreadsheet':
                    st.info("Phát hiện Google Sheets → Export thành CSV...")
                    request = drive_service.files().export_media(fileId=file_id, mimeType='text/csv')
                    file_name = file_name.rsplit('.', 1)[0] + '.csv'
                else:
                    request = drive_service.files().get_media(fileId=file_id)

                downloader = MediaIoBaseDownload(fh, request)
                done = False
                while not done:
                    status, done = downloader.next_chunk()

                fh.seek(0)

                # === ĐỌC FILE ===
                if file_name.endswith('.csv'):
                    df = pd.read_csv(fh, dtype=str, encoding='utf-8-sig')
                else:
                    df = pd.read_excel(fh, engine='openpyxl', dtype=str)

                if df.empty:
                    st.error("File rỗng!")
                    return False

                # === TÌM CỘT THÔNG MINH ===
                name_col = class_col = None
                for col in df.columns:
                    c = str(col).lower().strip()
                    if any(x in c for x in ['họ tên', 'tên', 'name', 'họ']):
                        name_col = col
                    if any(x in c for x in ['lớp', 'class', 'lop']):
                        class_col = col

                if not name_col:
                    st.error(f"**Không tìm thấy cột Họ tên!**\nCác cột có: {list(df.columns[:5])}...")
                    return False
                if not class_col:
                    st.error(f"**Không tìm thấy cột Lớp!**\nCác cột có: {list(df.columns[:5])}...")
                    return False

                df = df.rename(columns={name_col: 'Họ tên', class_col: 'Lớp'})
                point_cols = [c for c in df.columns if c not in ['Họ tên', 'Lớp', 'Zalo_ID']]

                for c in point_cols:
                    df[c] = pd.to_numeric(df[c].str.strip(), errors='coerce')
                df['ĐTB'] = df[point_cols].mean(axis=1).round(2)

                # === LƯU CLEANED.JSON ===
                records = []
                for _, r in df.iterrows():
                    record = {
                        "Họ tên": str(r['Họ tên']).strip(),
                        "Lớp": str(r['Lớp']).strip(),
                        "Zalo_ID": str(r.get('Zalo_ID', '')).strip(),
                        "ĐTB": float(r['ĐTB']) if pd.notna(r['ĐTB']) else None,
                        "Môn": {c: float(r[c]) if pd.notna(r[c]) else None for c in point_cols}
                    }
                    records.append(record)

                output = json.dumps(records, ensure_ascii=False, indent=2).encode('utf-8')
                media = MediaFileUpload(BytesIO(output), mimetype='application/json')
                drive_service.files().create(
                    body={'name': f"cleaned_{int(time.time())}.json", 'parents': [folders['AI1_OUTPUT']]},
                    media_body=media
                ).execute()

                st.success(f"AI 1: Đã xử lý `{file_name}` → {len(records)} học sinh")
                return True

            except Exception as e:
                st.error(f"**Lỗi AI1:** {str(e)}")
                return False

                            # === SAU KHI LƯU JSON THÀNH CÔNG ===
                st.success(f"AI 1: Đã xử lý `{file_name}` → {len(records)} học sinh")
                
                # ĐÁNH DẤU: AI1 ĐÃ CHẠY XONG
                st.session_state.ai1_done = True
                return True
    # AI 2: Phân tích
        # AI 2: Phân tích
        # AI 2: Phân tích
    def run_ai2():
        # KIỂM TRA AI1 ĐÃ CHẠY CHƯA
        if not st.session_state.get('ai1_done', False):
            st.error("**LỖI: Chưa chạy AI 1!**\n"
                     "Vui lòng nhấn **AI 1** hoặc **TOÀN BỘ** trước.")
            return False

        if not st.session_state.get('folder_ids') or not st.session_state.folder_ids.get('AI1_OUTPUT'):
            st.error("Thư mục AI1_OUTPUT chưa được tạo!")
            return False

        if not st.session_state.get('folder_ids') or not st.session_state.folder_ids.get('AI1_OUTPUT'):
            st.error("Chưa tạo thư mục AI1_OUTPUT! Chạy AI1 trước.")
            return False

        try:
            res = drive_service.files().list(
                q=f"'{folders['AI1_OUTPUT']}' in parents and mimeType='application/json'",
                orderBy="modifiedTime desc"
            ).execute()
            files = res.get('files', [])
            if not files:
                st.error("Chưa có file cleaned.json! Chạy AI1 trước.")
                return False

            # ĐỌC FILE MỚI NHẤT
            file_id = files[0]['id']
            fh = BytesIO()
            downloader = MediaIoBaseDownload(fh, drive_service.files().get_media(fileId=file_id))
            done = False
            while not done: status, done = downloader.next_chunk()
            fh.seek(0)
            df = pd.read_json(fh)

            # BỔ SUNG CỘT
            df['Xếp hạng lớp'] = df.groupby('Lớp')['ĐTB'].rank(ascending=False, method='min').astype(int)
            df['Xếp hạng trường'] = df['ĐTB'].rank(ascending=False, method='min').astype(int)
            df['Đánh giá'] = df['ĐTB'].apply(lambda x: "Giỏi" if x >= 8 else "Khá" if x >= 6.5 else "Trung bình" if x >= 5 else "Yếu")
            df['Cảnh báo'] = df['ĐTB'].apply(lambda x: "Nguy cơ" if x < 5 else "Ổn định")

            # LƯU AI2
            output = df.to_json(orient="records", force_ascii=False, indent=2).encode('utf-8')
            media = MediaFileUpload(BytesIO(output), mimetype='application/json')
            drive_service.files().create(
                body={'name': f"analysis_{int(time.time())}.json", 'parents': [folders['AI2_OUTPUT']]},
                media_body=media
            ).execute()

            st.session_state.ai2_result = df.to_dict("records")
            st.success("AI 2: Phân tích hoàn tất!")
            return True

        except Exception as e:
            st.error(f"Lỗi AI2: {str(e)}")
            return False

    # AI 3: Biểu đồ (hiển thị trên app) + Gửi Zalo
        # AI 3: Biểu đồ + Gửi Zalo (KIỂM TRA CỘT AN TOÀN)
    def run_ai3():
        # KIỂM TRA AI2 ĐÃ CHẠY CHƯA
        if 'ai2_result' not in st.session_state or not st.session_state.ai2_result:
            st.error("**LỖI: Chưa có kết quả từ AI 2!**\n"
                     "Vui lòng chạy **AI 1 → AI 2** trước.")
            return False
            
        if 'ai2_result' not in st.session_state:
            st.error("Chưa có dữ liệu từ AI 2! Chạy AI1 → AI2 trước.")
            return False

        df = pd.DataFrame(st.session_state.ai2_result)
        if df.empty:
            st.error("Dữ liệu AI2 rỗng!")
            return False

        # KIỂM TRA CỘT
        if 'Lớp' not in df.columns or 'ĐTB' not in df.columns:
            st.error(f"Thiếu cột: Lớp={'✓' if 'Lớp' in df.columns else '✗'}, ĐTB={'✓' if 'ĐTB' in df.columns else '✗'}")
            return False

        # BIỂU ĐỒ 1
        try:
            class_avg = df.groupby('Lớp')['ĐTB'].mean().round(2).reset_index()
            fig1 = px.bar(class_avg, x='Lớp', y='ĐTB', title="ĐTB theo lớp")
            st.plotly_chart(fig1, use_container_width=True)
        except:
            st.warning("Không vẽ được biểu đồ lớp.")

        # BIỂU ĐỒ 2
        try:
            fig2 = px.pie(df['Đánh giá'].value_counts(), names=df['Đánh giá'].value_counts().index, title="Học lực")
            st.plotly_chart(fig2, use_container_width=True)
        except:
            st.warning("Không vẽ được biểu đồ học lực.")

        st.success("AI 3: Hoàn tất!")

        # === GỬI ZALO (CHỈ GỬI NẾU CÓ ĐỦ CỘT) ===
        zalo_token = os.getenv('ZALO_OA_TOKEN')
        if zalo_token and not missing_zalo:
            if st.button("Gửi Báo Cáo Qua Zalo", type="primary"):
                with st.spinner("Đang gửi tin nhắn..."):
                    success = 0
                    failed = 0
                    for _, r in df.iterrows():
                        zid = str(r.get('Zalo_ID', '')).strip()
                        if not zid or zid in ['nan', 'None', '']:
                            failed += 1
                            continue

                        msg = f"**BÁO CÁO HỌC TẬP**\n"
                        msg += f"Họ tên: *{r['Họ tên']}*\n"
                        msg += f"Lớp: {r['Lớp']}\n"
                        msg += f"ĐTB: *{r['ĐTB']}* ({r['Đánh giá']})\n"
                        msg += f"Dự báo: *{r.get('Dự báo', 'N/A')}*\n"
                        msg += f"Xếp hạng lớp: *{r.get('Xếp hạng lớp', 'N/A')}*\n"
                        msg += f"Cảnh báo: {r.get('Cảnh báo', 'Ổn định')}"

                        payload = {
                            "recipient": {"user_id": zid},
                            "message": {"text": msg}
                        }
                        try:
                            response = requests.post(
                                "https://openapi.zalo.me/v2.0/oa/message/cs",
                                headers={
                                    "access_token": zalo_token,
                                    "Content-Type": "application/json"
                                },
                                json=payload,
                                timeout=10
                            )
                            if response.status_code == 200:
                                success += 1
                            else:
                                failed += 1
                        except:
                            failed += 1
                        time.sleep(0.8)  # Tránh spam

                    st.success(f"**GỬI ZALO HOÀN TẤT!**\n"
                               f"Thành công: {success}\n"
                               f"Thất bại: {failed}")
        elif zalo_token and missing_zalo:
            st.warning(f"**Không thể gửi Zalo:** Thiếu cột `{', '.join(missing_zalo)}`")
        else:
            st.info("**Zalo chưa được cấu hình.** Thiết lập `ZALO_OA_TOKEN` trong Environment.")

        return True
    # NÚT CHẠY
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.button("AI 1: Xử lý", on_click=run_ai1)
    with col2: st.button("AI 2: Phân tích", on_click=run_ai2)
    with col3: st.button("AI 3: Báo cáo", on_click=run_ai3)
    with col6:
        if st.button("TOÀN BỘ"):
            if run_ai1():
                time.sleep(3)
                if run_ai2():
                    time.sleep(2)
                    run_ai3()

    # Hiển thị kết quả
   if 'ai2_result' in st.session_state and st.session_state.ai2_result:
        df = pd.DataFrame(st.session_state.ai2_result)
        st.subheader("Kết quả phân tích")

        # Chỉ lấy cột có sẵn
        cols = ['Họ tên', 'Lớp', 'ĐTB', 'Đánh giá', 'Dự báo', 'Cảnh báo']
        available = [c for c in cols if c in df.columns]

        if available:
            st.dataframe(df[available], use_container_width=True)
        else:
            st.error("Không có dữ liệu hợp lệ để hiển thị!")
    else:
        st.info("Chưa có kết quả. Vui lòng chạy **AI 1 → AI 2 → AI 3**")

    st.markdown(
        "<div style='position:fixed;bottom:0;width:100%;background:#2c3e50;color:white;padding:10px;text-align:center'>"
        "© 2025 AI Dự Báo Điểm - Giáo viên Tin học</div>",
        unsafe_allow_html=True
    )