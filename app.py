import requests
from PIL import Image
from io import BytesIO
import glob
import os
import streamlit as st
import pandas as pd
import yaml
import json
import plotly.express as px
import plotly.graph_objects as go  # ← THÊM DÒNG NÀY
from io import StringIO, BytesIO
from sklearn.linear_model import LinearRegression
import numpy as np
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
import time  # ← THÊM ĐỂ DÙNG time.sleep()

# Load config
config_path = 'auth_config.yaml'
config = None
if os.path.exists(config_path):
    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
elif 'AUTH_CONFIG' in os.environ:
    config = yaml.safe_load(os.environ['AUTH_CONFIG'])
else:
    st.error("File YAML không tồn tại hoặc biến môi trường AUTH_CONFIG không được thiết lập!")

# Session state
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'data_store' not in st.session_state:
    st.session_state.data_store = {}
if 'selected_dataset' not in st.session_state:
    st.session_state.selected_dataset = None

# Google Drive setup
SCOPES = ['https://www.googleapis.com/auth/drive']
creds_json = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
if not creds_json:
    raise RuntimeError("LỖI: Thiếu GOOGLE_APPLICATION_CREDENTIALS!")
try:
    creds_info = json.loads(creds_json)
    creds = Credentials.from_service_account_info(creds_info, scopes=SCOPES)
    drive_service = build('drive', 'v3', credentials=creds)
except Exception as e:
    raise RuntimeError(f"Lỗi Google Drive: {str(e)}")

FOLDER_ID = '1K6Z-huJcdphdM42o2NL3kvu6KY7asD_u'

# === UPLOAD TO DRIVE (SỬA MIME + TEMP FILE) ===
def upload_to_drive(temp_file_path, file_name):
    file_metadata = {'name': file_name, 'parents': [FOLDER_ID]}
    media = MediaFileUpload(
        temp_file_path,
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        resumable=True
    )
    file = drive_service.files().create(body=file_metadata, media_body=media, fields='id').execute()
    st.success(f"Đã tải '{file_name}' lên Google Drive!")
    return file.get('id')

# === DOWNLOAD + AI THÍCH NGHI (SỬA pd.to_numeric) ===
def download_from_drive(file_id):
    request = drive_service.files().get_media(fileId=file_id)
    fh = BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        status, done = downloader.next_chunk()
    fh.seek(0)

    df_raw = pd.read_excel(fh, engine='openpyxl', header=None)
    fh.seek(0)

    header_row = None
    name_keywords = ["họ tên", "họ và tên", "tên học sinh"]
    class_keywords = ["lớp"]
    point_keywords = ["toán", "văn", "lý", "hóa", "sinh", "sử", "địa", "gdcd", "tin", "anh", "đtb"]

    for idx, row in df_raw.iterrows():
        row_lower = [str(cell).strip().lower() if pd.notna(cell) else "" for cell in row]
        if any(kw in " ".join(row_lower) for kw in name_keywords):
            point_count = sum(any(pkw in cell for pkw in point_keywords) for cell in row_lower)
            if point_count >= 3:
                header_row = idx
                break

    if header_row is None:
        st.warning("Không tìm thấy dòng tiêu đề.")
        return pd.read_excel(fh, engine='openpyxl')

    df = pd.read_excel(fh, engine='openpyxl', header=header_row)
    df = df.dropna(how='all').dropna(how='all', axis=1).reset_index(drop=True)

    cols_to_keep = []
    new_names = {}
    for col in df.columns:
        col_lower = str(col).strip().lower()
        if any(kw in col_lower for kw in name_keywords):
            cols_to_keep.append(col); new_names[col] = "Họ tên"
        elif any(kw in col_lower for kw in class_keywords):
            cols_to_keep.append(col); new_names[col] = "Lớp"
        elif any(pkw in col_lower for pkw in point_keywords):
            cols_to_keep.append(col)
            if "toán" in col_lower: new_names[col] = "Toán"
            elif "văn" in col_lower: new_names[col] = "Văn"
            elif "lý" in col_lower: new_names[col] = "Lý"
            elif "hóa" in col_lower: new_names[col] = "Hóa"
            elif "sinh" in col_lower: new_names[col] = "Sinh"
            elif "sử" in col_lower: new_names[col] = "Sử"
            elif "địa" in col_lower: new_names[col] = "Địa"
            elif "gdcd" in col_lower: new_names[col] = "GDCD"
            elif "tin" in col_lower: new_names[col] = "Tin"
            elif "anh" in col_lower: new_names[col] = "Anh"
            elif "đtb" in col_lower: new_names[col] = "ĐTB"

    df = df[cols_to_keep].rename(columns=new_names)
    df = df[df['Họ tên'].notna() & (df['Họ tên'].str.strip() != "")]
    
    point_cols = [col for col in df.columns if col not in ['Họ tên', 'Lớp']]
    for col in point_cols:
        df[col] = df[col].astype(str).str.strip()  # ← ÉP CHUỖI TRƯỚC
        df[col] = pd.to_numeric(df[col], errors='coerce')  # ← AN TOÀN

    return df.reset_index(drop=True)

# === ĐĂNG NHẬP ===
if not st.session_state.authenticated and config and 'credentials' in config:
    st.title("Đăng Nhập")
    username = st.text_input("Tên người dùng")
    password = st.text_input("Mật khẩu", type="password")
    if st.button("Đăng nhập"):
        if username in config['credentials']['usernames'] and config['credentials']['usernames'][username]['password'] == password:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Sai thông tin!")
else:
    st.sidebar.title("Menu")
    if st.session_state.authenticated:
        st.sidebar.button("Logout", on_click=lambda: setattr(st.session_state, 'authenticated', False))

if st.session_state.authenticated:
    st.title("AI Dự Báo Điểm Học Sinh")
    st.markdown("<div class='banner'>Chào mừng đến với Hệ Thống AI Dự Báo Điểm Học Sinh</div>", unsafe_allow_html=True)

    # === KHO DỮ LIỆU ===
    st.header("Quản Lý Kho Dữ Liệu")
    if not st.session_state.data_store:
        st.warning("Kho dữ liệu trống.")
    else:
        st.session_state.selected_dataset = st.selectbox("Chọn dữ liệu", list(st.session_state.data_store.keys()))

    # === TẢI FILE ===
    st.subheader("Tải Lên Dữ Liệu Mới")
    uploaded_file = st.file_uploader("Chọn file", type=["csv", "xlsx", "xls", "json"])
    if uploaded_file:
        file_name = uploaded_file.name
        try:
            if file_name.endswith('.csv'):
                df = pd.read_csv(uploaded_file, dtype=str)
            elif file_name.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(uploaded_file, engine='openpyxl', dtype=str)
            st.success(f"Đọc thành công {len(df)} hàng!")
            st.session_state.data_store[file_name] = df
            st.session_state.selected_dataset = file_name

            # === UPLOAD VỚI TEMP FILE ===
            temp_path = f"/tmp/{file_name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            try:
                upload_to_drive(temp_path, file_name)
            except Exception as e:
                st.error(f"Upload lỗi: {e}")
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
        except Exception as e:
            st.error(f"Lỗi: {e}")

    # === TẢI TỪ DRIVE ===
    query = f"'{FOLDER_ID}' in parents"
    results = drive_service.files().list(q=query, fields="files(id, name)").execute()
    drive_files = results.get('files', [])
    if drive_files:
        st.subheader("Dữ Liệu Từ Google Drive")
        for file in drive_files:
            if file['name'] not in st.session_state.data_store:
                try:
                    df = download_from_drive(file['id'])
                    st.session_state.data_store[file['name']] = df
                    st.success(f"Đã tải '{file['name']}'")
                    st.session_state.selected_dataset = file['name']
                except Exception as e:
                    st.error(f"Lỗi: {e}")

    # === PHÂN TÍCH ===
    if st.session_state.data_store and st.session_state.selected_dataset:
        df = st.session_state.data_store[st.session_state.selected_dataset]

        st.subheader("Bảng Điểm Học Sinh")
        if 'Họ tên' in df.columns:
            point_cols = [c for c in df.columns if c not in ['Họ tên', 'Lớp']]
            display_df = df[['Họ tên', 'Lớp'] + point_cols].dropna(subset=['Họ tên'])
            st.dataframe(display_df, use_container_width=True, hide_index=True)

            if point_cols:
                df['ĐTB'] = df[point_cols].mean(axis=1).round(2)
                df = df.sort_values('ĐTB', ascending=False)

                # Top 3
                st.subheader("Top 3 Học Sinh Xuất Sắc")
                top3 = df.head(3)[['Họ tên', 'Lớp', 'ĐTB'] + point_cols]
                st.dataframe(top3, use_container_width=True, hide_index=True)
                fig = px.bar(top3.melt(id_vars='Họ tên', value_vars=point_cols + ['ĐTB']), x='Họ tên', y='value', color='variable', barmode='group')
                st.plotly_chart(fig, use_container_width=True)

                # Phân bố
                st.subheader("Phân Bố ĐTB")
                fig_hist = px.histogram(df, x='ĐTB', nbins=15)
                fig_hist.add_vline(x=df['ĐTB'].mean(), line_dash="dash", line_color="red")
                st.plotly_chart(fig_hist, use_container_width=True)

                # Dự báo
                st.subheader("Dự Báo AI")
                pred_data = []
                for col in point_cols:
                    curr = df[col].mean().round(2)
                    if df[col].notna().sum() >= 2:
                        X = np.arange(len(df)).reshape(-1,1)
                        y = df[col].dropna()
                        model = LinearRegression().fit(X[:len(y)], y.values.reshape(-1,1))
                        pred = model.predict([[len(df)]])[0][0]
                    else:
                        pred = curr
                    pred_data.append({"Môn": col, "Hiện tại": curr, "Dự báo": round(pred, 2)})
                pred_df = pd.DataFrame(pred_data)
                st.dataframe(pred_df, use_container_width=True, hide_index=True)

                # Cảnh báo
                st.subheader("Học Sinh Cần Hỗ Trợ")
                weak = df[df['ĐTB'] < 6.5][['Họ tên', 'Lớp', 'ĐTB']]
                if not weak.empty:
                    st.warning(f"{len(weak)} học sinh yếu!")
                    st.dataframe(weak, use_container_width=True)
                else:
                    st.success("Tất cả đạt chuẩn!")

        else:
            st.error("Không tìm thấy cột 'Họ tên'.")
            st.dataframe(df.head(10))

    # === GỬI ZALO ===
    st.markdown("---")
    st.subheader("Gửi Báo Cáo Qua Zalo")
    if st.button("GỬI CHO TỪNG MẸ"):
        with st.spinner("Đang gửi..."):
            gui_bao_cao_zalo_tu_dong()

# === HÀM GỬI ZALO (SỬA INDENT + GO) ===
def gui_bao_cao_zalo_tu_dong():
    ZALO_OA_TOKEN = os.environ.get("ZALO_OA_TOKEN")
    if not ZALO_OA_TOKEN:
        st.error("Thiếu ZALO_OA_TOKEN!")
        return

    try:
        ph_df = None
        for file in drive_files:
            if file['name'] == "phu_huynh.xlsx":
                ph_df = download_from_drive(file['id'])
                break
        if ph_df is None:
            st.error("Không tìm thấy phu_huynh.xlsx")
            return

        point_dfs = []
        for file in drive_files:
            if file['name'].startswith("diem_") and file['name'].endswith(".xlsx"):
                df = download_from_drive(file['id'])
                ky = file['name'].replace("diem_", "").replace(".xlsx", "")
                df['Kỳ'] = ky
                point_dfs.append(df)

        if not point_dfs:
            st.error("Không tìm thấy file điểm!")
            return

        full_df = pd.concat(point_dfs, ignore_index=True)
        merged_df = pd.merge(full_df, ph_df, left_on="Họ tên", right_on="Họ tên học sinh", how="inner")
        if merged_df.empty:
            st.error("Không ghép được dữ liệu!")
            return

        success = 0
        for ten_con, group in merged_df.groupby("Họ tên"):
            me_info = group.iloc[0]
            ten_me = me_info.get("Tên mẹ", "Phụ huynh")
            sdt_me = str(me_info.get("SĐT Zalo mẹ", "")).strip()

            scores = group.sort_values("Kỳ")["Toán"].tolist()
            ky_list = group.sort_values("Kỳ")["Kỳ"].tolist()

            if len(scores) >= 2:
                X = np.arange(len(scores)).reshape(-1,1)
                y = np.array(scores)
                model = LinearRegression().fit(X, y.reshape(-1,1))
                predicted = float(model.predict([[len(scores)]])[0])
            else:
                predicted = scores[-1] if scores else 0

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=ky_list, y=scores, mode='lines+markers', name='Thực tế'))
            fig.add_trace(go.Scatter(x=ky_list + ["Dự báo"], y=scores + [predicted], mode='lines', line=dict(dash='dot'), name='Dự báo'))
            fig.update_layout(title=ten_con, height=400)
            safe_name = "".join(c for c in ten_con if c.isalnum())
            chart_path = f"/tmp/chart_{safe_name}.png"
            fig.write_image(chart_path)

            message = f"Chào cô {ten_me}!\nCon {ten_con}: Điểm Toán gần nhất: {scores[-1] if scores else 'N/A'}\nDự báo: {predicted:.2f}"

            if sdt_me and len(sdt_me) >= 10:
                url = "https://openapi.zalo.me/v2.0/oa/message"
                headers = {"access_token": ZALO_OA_TOKEN}
                payload = {"recipient": {"phone": sdt_me}, "message": {"text": message}}
                if os.path.exists(chart_path):
                    with open(chart_path, "rb") as f:
                        r = requests.post(url, data=payload, headers=headers, files={"attachment": f})
                else:
                    r = requests.post(url, json=payload, headers=headers)
                if r.json().get("error") == 0:
                    success += 1
                time.sleep(1)
                if os.path.exists(chart_path):
                    os.remove(chart_path)

        st.success(f"ĐÃ GỬI CHO {success} PHỤ HUYNH!")
    except Exception as e:
        st.error(f"Lỗi: {str(e)}")

# === FOOTER ===
st.markdown("""
<style>.footer{position:fixed;bottom:0;width:100%;background:linear-gradient(90deg,#3498db,#2c3e50);color:white;padding:10px;text-align:center}</style>
<div class='footer'>© 2025 AI Dự Báo Điểm Học Sinh</div>
""", unsafe_allow_html=True)