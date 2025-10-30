import requests
from PIL import Image
from io import BytesIO
import glob
import os
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
import os
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
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
creds_json = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')

if not creds_json:
    raise RuntimeError(
        "LỖI: Biến môi trường 'GOOGLE_APPLICATION_CREDENTIALS' chưa được thiết lập!\n"
        "Vui lòng vào Render Dashboard > Settings > Environment Variables > Thêm key này với nội dung JSON từ file credentials.json"
    )

try:
    creds_info = json.loads(creds_json)
    creds = Credentials.from_service_account_info(creds_info, scopes=SCOPES)
    drive_service = build('drive', 'v3', credentials=creds)
except Exception as e:
    raise RuntimeError(f"Lỗi khi tải thông tin xác thực Google Drive: {str(e)}")


FOLDER_ID = '1K6Z-huJcdphdM42o2NL3kvu6KY7asD_u'  # Thay bằng ID thư mục (ví dụ: 1abc123...)

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
    
    # Đọc toàn bộ file (không header)
    df_raw = pd.read_excel(fh, engine='openpyxl', header=None)
    fh.seek(0)  # Reset để đọc lại
    
    # === AI TỰ ĐỘNG TÌM TIÊU ĐỀ ===
    header_row = None
    name_col_idx = None
    class_col_idx = None
    point_col_idxs = []
    
    # Từ khóa nhận diện
    name_keywords = ["họ tên", "họ và tên", "tên học sinh", "họ sinh"]
    class_keywords = ["lớp", "class"]
    point_keywords = ["toán", "văn", "lý", "hóa", "sinh", "sử", "địa", "gdcd", "tin", "anh", "đtb", "trung bình"]
    
    for idx, row in df_raw.iterrows():
        row_lower = [str(cell).strip().lower() if pd.notna(cell) else "" for cell in row]
        
        # Tìm dòng có "họ tên" và ít nhất 3 môn
        if any(kw in " ".join(row_lower) for kw in name_keywords):
            # Đếm số môn trong dòng
            point_count = sum(any(pkw in cell for pkw in point_keywords) for cell in row_lower)
            if point_count >= 3:
                header_row = idx
                # Tìm cột họ tên, lớp, điểm
                for j, cell in enumerate(row_lower):
                    if any(kw in cell for kw in name_keywords):
                        name_col_idx = j
                    elif any(kw in cell for kw in class_keywords):
                        class_col_idx = j
                    elif any(pkw in cell for pkw in point_keywords):
                        point_col_idxs.append(j)
                break
    
    if header_row is None:
        st.warning("Không tìm thấy dòng tiêu đề học sinh. Dùng file chuẩn.")
        return pd.read_excel(fh, engine='openpyxl')
    
    # === ĐỌC LẠI VỚI TIÊU ĐỀ CHUẨN ===
    df = pd.read_excel(fh, engine='openpyxl', header=header_row)
    
    # === LÀM SẠCH DỮ LIỆU ===
    df = df.dropna(how='all').dropna(how='all', axis=1).reset_index(drop=True)
    
    # Chỉ giữ cột cần thiết
    cols_to_keep = []
    new_names = {}
    
    for col in df.columns:
        col_lower = str(col).strip().lower()
        if any(kw in col_lower for kw in name_keywords):
            cols_to_keep.append(col)
            new_names[col] = "Họ tên"
        elif any(kw in col_lower for kw in class_keywords):
            cols_to_keep.append(col)
            new_names[col] = "Lớp"
        elif any(pkw in col_lower for pkw in point_keywords):
            cols_to_keep.append(col)
            # Chuẩn hóa tên môn
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
            elif "trung bình" in col_lower or "đtb" in col_lower: new_names[col] = "ĐTB"
            else: new_names[col] = col
    
    df = df[cols_to_keep].rename(columns=new_names)
    
    # Xóa dòng không có tên học sinh
    df = df[df['Họ tên'].notna() & (df['Họ tên'].str.strip() != "")]
    
    # Chuyển điểm thành số
    point_cols = [col for col in df.columns if col not in ['Họ tên', 'Lớp']]
    for col in point_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df.reset_index(drop=True)

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

    # === HIỂN THỊ BẢNG HỌC SINH SIÊU THÍCH NGHI ===
    st.subheader("Bảng Điểm Học Sinh")
    if 'Họ tên' in df.columns:
        point_cols = [col for col in df.columns if col not in ['Họ tên', 'Lớp']]
        display_cols = ['Họ tên', 'Lớp'] + point_cols
        display_df = df[display_cols].copy()
        display_df = display_df.dropna(subset=['Họ tên']).reset_index(drop=True)
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        # === TÍNH ĐIỂM TRUNG BÌNH ===
        if point_cols:
            df['ĐTB'] = df[point_cols].mean(axis=1).round(2)
            df = df.sort_values('ĐTB', ascending=False).reset_index(drop=True)

            # === TOP 3 HỌC SINH ===
            st.subheader("Top 3 Học Sinh Xuất Sắc")
            top3 = df.head(3)[['Họ tên', 'Lớp', 'ĐTB'] + point_cols]
            st.dataframe(top3, use_container_width=True, hide_index=True)

            # Biểu đồ Top 3
            fig_top = px.bar(
                top3.melt(id_vars=['Họ tên'], value_vars=point_cols + ['ĐTB'], var_name='Môn', value_name='Điểm'),
                x='Họ tên', y='Điểm', color='Môn', barmode='group',
                title="So Sánh Điểm Top 3 Học Sinh",
                color_discrete_sequence=px.colors.qualitative.Bold
            )
            fig_top.update_layout(height=500)
            st.plotly_chart(fig_top, use_container_width=True)

            # === PHÂN BỐ ĐIỂM TRUNG BÌNH ===
            st.subheader("Phân Bố Điểm Trung Bình Toàn Lớp")
            fig_hist = px.histogram(df, x='ĐTB', nbins=15, title="Phân Bố ĐTB", color_discrete_sequence=['#27ae60'])
            fig_hist.add_vline(x=df['ĐTB'].mean(), line_dash="dash", line_color="red", annotation_text=f"TB: {df['ĐTB'].mean():.2f}")
            st.plotly_chart(fig_hist, use_container_width=True)

            # === DỰ BÁO AI CHO TỪNG MÔN ===
            st.subheader("Dự Báo Điểm Kỳ Tới (AI)")
            predictions = {}
            for col in point_cols:
                if df[col].notna().sum() >= 2:
                    X = np.arange(len(df)).reshape(-1, 1)
                    y = df[col].dropna().values
                    model = LinearRegression().fit(X[:len(y)], y.reshape(-1, 1))
                    pred = model.predict([[len(df)]])[0][0]
                    predictions[col] = round(float(pred), 2)
                else:
                    predictions[col] = df[col].mean().round(2) if df[col].notna().any() else 0

            pred_df = pd.DataFrame(list(predictions.items()), columns=['Môn', 'Dự báo kỳ tới'])
            pred_df['Hiện tại'] = [df[col].mean().round(2) for col in point_cols]
            pred_df['Thay đổi'] = (pred_df['Dự báo kỳ tới'] - pred_df['Hiện tại']).round(2)
            pred_df = pred_df[['Môn', 'Hiện tại', 'Dự báo kỳ tới', 'Thay đổi']]

            st.dataframe(pred_df, use_container_width=True, hide_index=True)

            # Biểu đồ dự báo
            fig_pred = px.line(pred_df, x='Môn', y=['Hiện tại', 'Dự báo kỳ tới'], title="Xu Hướng Điểm Theo Môn", markers=True)
            fig_pred.add_scatter(x=pred_df['Môn'], y=pred_df['Thay đổi'], mode='text', text=pred_df['Thay đổi'].apply(lambda x: f"{x:+.1f}"), textposition="top center", name="Thay đổi")
            st.plotly_chart(fig_pred, use_container_width=True)

            # === CẢNH BÁO HỌC SINH YẾU ===
            st.subheader("Học Sinh Cần Hỗ Trợ")
            weak_students = df[df['ĐTB'] < 6.5][['Họ tên', 'Lớp', 'ĐTB'] + point_cols]
            if not weak_students.empty:
                st.warning(f"**{len(weak_students)} học sinh có ĐTB dưới 6.5**")
                st.dataframe(weak_students, use_container_width=True, hide_index=True)
            else:
                st.success("**Tất cả học sinh đều đạt ĐTB ≥ 6.5!**")

    else:
        st.error("Không tìm thấy cột 'Họ tên'. File Excel có thể không phải bảng điểm học sinh.")
        st.dataframe(df.head(10), use_container_width=True)
            # NÚT GỬI ZALO TỰ ĐỘNG CHO TỪNG MẸ
        # ========================================
        st.markdown("---")
        st.subheader("Gửi Báo Cáo Tự Động Qua Zalo")
        if st.button("GỬI BÁO CÁO CHO TỪNG MẸ QUA ZALO"):
            with st.spinner("Đang đọc dữ liệu và gửi tin nhắn..."):
                gui_bao_cao_zalo_tu_dong()
        # ========================================
# HÀM TỰ ĐỘNG GỬI ZALO CHO TỪNG MẸ
# ========================================
def gui_bao_cao_zalo_tu_dong():
    ZALO_OA_TOKEN = os.environ.get("ZALO_OA_TOKEN")
    if not ZALO_OA_TOKEN:
        st.error("Thiếu ZALO_OA_TOKEN! Vào Render → Settings → Environment Variables để thêm.")
        return

    try:
        # 1. Tải file phụ huynh
        ph_df = None
        for file in drive_files:
            if file['name'] == "phu_huynh.xlsx":
                ph_df = download_from_drive(file['id'])
                break
        if ph_df is None:
            st.error("Không tìm thấy file `phu_huynh.xlsx` trên Google Drive!")
            return

        # 2. Tải tất cả file điểm
        point_dfs = []
        for file in drive_files:
            if file['name'].startswith("diem_") and file['name'].endswith(".xlsx"):
                df = download_from_drive(file['id'])
                ky = file['name'].replace("diem_", "").replace(".xlsx", "")
                df['Kỳ'] = ky
                point_dfs.append(df)

        if not point_dfs:
            st.error("Không tìm thấy file điểm nào (diem_HK1.xlsx, diem_HK2.xlsx...)")
            return

        # 3. Gộp điểm
        full_df = pd.concat(point_dfs, ignore_index=True)

        # 4. Ghép với phụ huynh
        merged_df = pd.merge(full_df, ph_df, on="Họ tên học sinh", how="inner")
        if merged_df.empty:
            st.error("Không khớp được học sinh nào! Kiểm tra tên trong 2 file.")
            return

        # 5. Gửi Zalo
        success = 0
        for ten_con, group in merged_df.groupby("Họ tên học sinh"):
            me_info = group.iloc[0]
            ten_me = me_info.get("Tên mẹ", "Phụ huynh")
            sdt_me = str(me_info.get("SĐT Zalo mẹ", "")).strip()

            # Lấy điểm Toán
            scores = group.sort_values("Kỳ")["Điểm Toán"].tolist()
            ky_list = group.sort_values("Kỳ")["Kỳ"].tolist()

            # Dự báo
            if len(scores) >= 2:
                X = np.array(range(len(scores))).reshape(-1, 1)
                y = np.array(scores)
                model = LinearRegression().fit(X, y)
                predicted = float(model.predict([[len(scores)]])[0])
            else:
                predicted = scores[-1] if scores else 0

            # Biểu đồ
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=ky_list, y=scores, mode='lines+markers', name='Thực tế'))
            fig.add_trace(go.Scatter(x=ky_list + ["Dự báo"], y=scores + [predicted], mode='lines', name='Dự báo', line=dict(dash='dot')))
            fig.update_layout(title=f"{ten_con}", height=400, template="simple_white")
            safe_name = "".join(c for c in ten_con if c.isalnum() or c in " _-")
            chart_path = f"/tmp/chart_{safe_name}.png"
            fig.write_image(chart_path)

            # Tin nhắn
            message = (
                f"Chào cô {ten_me}!\n"
                f"Con {ten_con}:\n"
                f"• Điểm gần nhất: {scores[-1] if scores else 'Chưa có'}\n"
                f"• Dự báo kỳ tới: {predicted:.2f}\n"
                f"{'Cần hỗ trợ thêm!' if predicted < 6.5 else 'Tiếp tục phát huy!'}"
            )

            # Gửi Zalo
            if sdt_me and len(sdt_me) >= 10:
                url = "https://openapi.zalo.me/v2.0/oa/message"
                headers = {"access_token": ZALO_OA_TOKEN}
                payload = {"recipient": {"phone": sdt_me}, "message": {"text": message}}
                if os.path.exists(chart_path):
                    with open(chart_path, "rb") as f:
                        files = {"attachment": f}
                        r = requests.post(url, data=payload, headers=headers, files=files)
                else:
                    r = requests.post(url, json=payload, headers=headers)
                if r.json().get("error") == 0:
                    success += 1
                time.sleep(1)

            # Xóa ảnh
            if os.path.exists(chart_path):
                os.remove(chart_path)

        st.success(f"ĐÃ GỬI THÀNH CÔNG CHO {success} PHỤ HUYNH QUA ZALO!")

    except Exception as e:
        st.error(f"Lỗi gửi Zalo: {str(e)}")

            # ========================================
        


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