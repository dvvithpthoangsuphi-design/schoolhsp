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
        with st.spinner("AI 1: Đang xử lý file điểm mới nhất..."):
            # === LẤY FILE MỚI NHẤT ===
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

            # === TẢI FILE (HỖ TRỢ CẢ GOOGLE SHEETS & FILE THẬT) ===
            fh = BytesIO()
            try:
                if mime_type == 'application/vnd.google-apps.spreadsheet':
                    # Google Sheets → Export CSV
                    request = drive_service.files().export_media(fileId=file_id, mimeType='text/csv')
                    file_name = file_name.rsplit('.', 1)[0] + '.csv'
                else:
                    # File thật (.xlsx, .csv)
                    request = drive_service.files().get_media(fileId=file_id)

                downloader = MediaIoBaseDownload(fh, request)
                done = False
                while not done:
                    status, done = downloader.next_chunk()
            except Exception as e:
                st.error(f"Lỗi tải file: {e}")
                return False

            fh.seek(0)

            # === ĐỌC DỮ LIỆU ===
            try:
                if file_name.endswith('.csv'):
                    df = pd.read_csv(fh, dtype=str, encoding='utf-8-sig')
                else:
                    df = pd.read_excel(fh, engine='openpyxl', dtype=str)
            except Exception as e:
                st.error(f"Lỗi đọc file: {e}")
                return False

            if df.empty:
                st.error("File rỗng hoặc không có dữ liệu!")
                return False

            # === TÌM CỘT HỌ TÊN & LỚP (THÔNG MINH) ===
            name_col = None
            class_col = None
            possible_name_keywords = ['họ tên', 'họ và tên', 'họ', 'tên', 'họtên', 'name', 'student', 'học sinh']
            possible_class_keywords = ['lớp', 'class', 'lop', 'grade']

            for col in df.columns:
                col_clean = str(col).strip().lower().replace(' ', '').replace('_', '')
                if any(keyword in col_clean for keyword in possible_name_keywords):
                    name_col = col
                if any(keyword in col_clean for keyword in possible_class_keywords):
                    class_col = col

            # === BÁO LỖI RÕ RÀNG NẾU THIẾU CỘT ===
            if not name_col:
                st.error("**LỖI: Không tìm thấy cột Họ tên!**\n"
                         "Các cột có trong file:\n" +
                         "\n".join([f"- `{c}`" for c in df.columns[:10]]) +
                         ("\n... (và nhiều cột khác)" if len(df.columns) > 10 else ""))
                return False

            if not class_col:
                st.error("**LỖI: Không tìm thấy cột Lớp!**\n"
                         "Các cột có trong file:\n" +
                         "\n".join([f"- `{c}`" for c in df.columns[:10]]) +
                         ("\n... (và nhiều cột khác)" if len(df.columns) > 10 else ""))
                return False

            # === ĐỔI TÊN CỘT ===
            df = df.rename(columns={name_col: 'Họ tên', class_col: 'Lớp'})

            # === XỬ LÝ ĐIỂM ===
            point_cols = [c for c in df.columns if c not in ['Họ tên', 'Lớp', 'Zalo_ID']]
            if not point_cols:
                st.warning("Không tìm thấy cột điểm nào!")
                return False

            for c in point_cols:
                df[c] = pd.to_numeric(df[c].str.strip(), errors='coerce')

            df['ĐTB'] = df[point_cols].mean(axis=1).round(2)

            # === CHUẨN HÓA DỮ LIỆU ===
            records = []
            for _, r in df.iterrows():
                record = {
                    "Họ tên": str(r['Họ tên']).strip(),
                    "Lớp": str(r['Lớp']).strip(),
                    "Zalo_ID": str(r.get('Zalo_ID', '')).strip(),
                    "ĐTB": float(r['ĐTB']) if pd.notna(r['ĐTB']) else None,
                    "Môn": {}
                }
                for c in point_cols:
                    val = r[c]
                    record["Môn"][c] = float(val) if pd.notna(val) else None
                records.append(record)

            # === LƯU FILE JSON ===
            output = json.dumps(records, ensure_ascii=False, indent=2).encode('utf-8')
            media = MediaFileUpload(BytesIO(output), mimetype='application/json')
            try:
                drive_service.files().create(
                    body={
                        'name': f"cleaned_{int(time.time())}.json",
                        'parents': [folders['AI1_OUTPUT']]
                    },
                    media_body=media
                ).execute()
                st.success(f"**AI 1 HOÀN TẤT!**\n"
                           f"Đã xử lý: `{file_name}`\n"
                           f"Học sinh: {len(records)}\n"
                           f"Cột điểm: {len(point_cols)}")
                return True
            except Exception as e:
                st.error(f"Lỗi lưu file: {e}")
                return False
    # AI 2: Phân tích
        # AI 2: Phân tích
        # AI 2: Phân tích
    def run_ai2():
        with st.spinner("AI 2: Đang phân tích và dự báo..."):
            # ĐẢM BẢO THƯ MỤC AI1_OUTPUT ĐÃ TỒN TẠI
            if not st.session_state.folder_ids or not st.session_state.folder_ids.get('AI1_OUTPUT'):
                st.error("Chưa chạy AI 1 hoặc thư mục AI1_OUTPUT chưa được tạo!")
                return False

            folder_id = st.session_state.folder_ids['AI1_OUTPUT']
            res = drive_service.files().list(
                q=f"'{folder_id}' in parents and mimeType='application/json'",
                orderBy="modifiedTime desc",
                fields="files(id, name)"
            ).execute()
            files = res.get('files', [])
            if not files:
                st.error("Không tìm thấy file cleaned.json trong AI1_OUTPUT!")
                return False

            file_id = files[0]['id']
            fh = BytesIO()
            downloader = MediaIoBaseDownload(fh, drive_service.files().get_media(fileId=file_id))
            done = False
            while not done:
                status, done = downloader.next_chunk()
            fh.seek(0)
            data = json.load(fh)
            df = pd.DataFrame(data)

            # Xếp hạng
            df['Xếp hạng lớp'] = df.groupby('Lớp')['ĐTB'].rank(ascending=False, method='min').astype(int)
            df['Xếp hạng trường'] = df['ĐTB'].rank(ascending=False, method='min').astype(int)

            # Dự báo
            preds = []
            for col in df.iloc[0]['Môn']:
                s = df[col].dropna()
                if len(s) >= 3:
                    model = LinearRegression().fit(np.arange(len(s)).reshape(-1,1), s)
                    pred = model.predict([[len(s)]])[0]
                else:
                    pred = s.mean() if not s.empty else 6.0
                preds.append(pred)
            df['Dự báo'] = round(np.mean(preds), 2)

            # Đánh giá + Cảnh báo
            df['Đánh giá'] = df['ĐTB'].apply(lambda x: "Giỏi" if x >= 8 else "Khá" if x >= 6.5 else "Trung bình" if x >= 5 else "Yếu")
            df['Cảnh báo'] = df.apply(lambda r: "Nguy cơ học lực yếu" if r['ĐTB'] < 5 else "Cảnh báo giảm điểm" if r['Dự báo'] < r['ĐTB'] else "Ổn định", axis=1)

            # Lưu kết quả
            output = df.to_json(orient="records", force_ascii=False, indent=2).encode('utf-8')
            media = MediaFileUpload(BytesIO(output), mimetype='application/json')
            drive_service.files().create(
                body={'name': f"analysis_{int(time.time())}.json", 'parents': [st.session_state.folder_ids['AI2_OUTPUT']]},
                media_body=media
            ).execute()

            st.session_state.ai2_result = df.to_dict("records")
            st.success("AI 2: Phân tích hoàn tất!")
            return True

    # AI 3: Biểu đồ (hiển thị trên app) + Gửi Zalo
    def run_ai3():
        if 'ai2_result' not in st.session_state:
            st.error("Chưa có kết quả từ AI 2!")
            return
        df = pd.DataFrame(st.session_state.ai2_result)
        
        # Biểu đồ 1: ĐTB theo lớp (hiển thị trực tiếp, không lưu PNG)
        fig1 = px.bar(df.groupby('Lớp')['ĐTB'].mean().reset_index(), x='Lớp', y='ĐTB', title="ĐTB Trung bình theo lớp")
        st.plotly_chart(fig1, use_container_width=True)
        
        # Biểu đồ 2: Phân bố đánh giá
        fig2 = px.pie(df['Đánh giá'].value_counts().reset_index(), names='Đánh giá', values='count', title="Phân bố học lực")
        st.plotly_chart(fig2, use_container_width=True)
        
        st.success("AI 3: Biểu đồ đã hiển thị! (Lưu thủ công nếu cần)")

        # Gửi Zalo
        zalo_token = os.environ.get('ZALO_OA_TOKEN')
        if zalo_token and st.button("Gửi Báo Cáo Qua Zalo"):
            success = 0
            for _, r in df.iterrows():
                if not r['Zalo_ID'] or r['Zalo_ID'] == '': continue
                msg = f"**BÁO CÁO HỌC TẬP**\n"
                msg += f"Họ tên: *{r['Họ tên']}*\n"
                msg += f"Lớp: {r['Lớp']}\n"
                msg += f"ĐTB: *{r['ĐTB']}* ({r['Đánh giá']})\n"
                msg += f"Dự báo kỳ tới: *{r['Dự báo']}*\n"
                msg += f"Xếp hạng lớp: *{r['Xếp hạng lớp']}*\n"
                msg += f"Cảnh báo: {r['Cảnh báo']}"
                payload = {"recipient": {"user_id": str(r['Zalo_ID'])}, "message": {"text": msg}}
                try:
                    r = requests.post(
                        "https://openapi.zalo.me/v2.0/oa/message/cs",
                        headers={"access_token": zalo_token, "Content-Type": "application/json"},
                        json=payload
                    )
                    if r.status_code == 200:
                        success += 1
                except:
                    pass
                time.sleep(0.8)
            st.success(f"Đã gửi thành công cho {success} phụ huynh!")

    # NÚT CHẠY
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.button("AI 1: Xử lý", on_click=run_ai1)
    with col2: st.button("AI 2: Phân tích", on_click=run_ai2)
    with col3: st.button("AI 3: Báo cáo", on_click=run_ai3)
    with col4:
        if st.button("CHẠY TOÀN BỘ"):
            if run_ai1(): time.sleep(2)
            if run_ai2(): time.sleep(2)
            run_ai3()

    # Hiển thị kết quả
    if 'ai2_result' in st.session_state:
        df = pd.DataFrame(st.session_state.ai2_result)
        st.subheader("Kết quả phân tích")
        st.dataframe(df[['Họ tên', 'Lớp', 'ĐTB', 'Đánh giá', 'Dự báo', 'Cảnh báo']], use_container_width=True)

    st.markdown(
        "<div style='position:fixed;bottom:0;width:100%;background:#2c3e50;color:white;padding:10px;text-align:center'>"
        "© 2025 AI Dự Báo Điểm - Giáo viên Tin học</div>",
        unsafe_allow_html=True
    )