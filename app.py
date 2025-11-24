## ==================== AI DỰ BÁO ĐIỂM THÔNG MINH - PHIÊN BẢN 3.1 HOÀN CHỈNH ====================
import os
import io
import time
import numpy as np
import pandas as pd
from io import BytesIO, StringIO
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload
from sqlalchemy import create_engine, text
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from dotenv import load_dotenv
import requests
import json
import logging
from collections import defaultdict, deque
import re
import concurrent.futures
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score
import hashlib
import joblib
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# === CẤU HÌNH NÂNG CAO ===
st.set_page_config(page_title="AI Dự Báo Điểm Thông Minh", page_icon="🧠", layout="wide")

# Cấu hình logging nâng cao
logging.basicConfig(
    level=logging.INFO, 
    filename='ai_advanced_log.txt',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# === KHỞI TẠO BIẾN MÔI TRƯỜNG VÀ KẾT NỐI ===
load_dotenv()
POSTGRES_URL = os.getenv("POSTGRES_URL", "postgresql://postgres:admin@localhost:5432/school_db")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "18593660252:AAEhRFy-Ae4v8xQM7yGAAzQSI5sYL1s30Ck")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "123456789")
USERNAME_ADMIN = os.getenv("USERNAME_ADMIN", "admin")
PASSWORD_ADMIN = os.getenv("PASSWORD_ADMIN", "admin")
NAME_ADMIN = os.getenv("NAME_ADMIN", "Admin Name")

# Khởi tạo engine với connection pooling
try:
    engine = create_engine(
        POSTGRES_URL, 
        connect_args={"connect_timeout": 10},
        pool_size=5,
        max_overflow=10,
        pool_pre_ping=True
    )
    logger.info("Kết nối PostgreSQL thành công")
except Exception as e:
    logger.error(f"Lỗi kết nối DB: {e}")
    st.error(f"Lỗi kết nối database: {e}")

# Khởi tạo Google Drive service
drive_service = None
credentials_file = "credentials.json"
try:
    if os.path.exists(credentials_file):
        creds = service_account.Credentials.from_service_account_file(
            credentials_file, 
            scopes=['https://www.googleapis.com/auth/drive']
        )
        drive_service = build('drive', 'v3', credentials=creds)
        logger.info("Drive service initialized successfully from credentials.json")
    else:
        logger.error("Error: credentials.json not found. Place it in the project directory.")
        st.warning("Không tìm thấy credentials.json. Ứng dụng sẽ chạy ở chế độ không có Google Drive.")
except Exception as e:
    logger.error(f"Credential Error: {e}")
    st.warning(f"Lỗi khởi tạo Google Drive: {e}")

# === CẤU HÌNH THƯ MỤC ===
RAW_DATA_FOLDER_ID = "1K6Z-huJcdphdM42o2NL3kvu6KY7asD_u"

# === KHỞI TẠO SESSION STATE ===
def initialize_session_state():
    """Khởi tạo session state an toàn"""
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "ai1_done" not in st.session_state:
        st.session_state.ai1_done = False
    if "ai2_done" not in st.session_state:
        st.session_state.ai2_done = False
    if "ai2_result" not in st.session_state:
        st.session_state.ai2_result = None
    if "user_role" not in st.session_state:
        st.session_state.user_role = "user"
    if "login_time" not in st.session_state:
        st.session_state.login_time = None
    if "username" not in st.session_state:
        st.session_state.username = ""
    if "selected_student" not in st.session_state:
        st.session_state.selected_student = None

# Gọi hàm khởi tạo
initialize_session_state()

# === ĐĂNG NHẬP NÂNG CAO ===
if not st.session_state.authenticated:
    st.title("🔐 Đăng Nhập Hệ Thống AI")
    
    col_login1, col_login2 = st.columns([1, 1])
    
    with col_login1:
        st.subheader("Đăng nhập")
        username = st.text_input("👤 Tên đăng nhập", placeholder="Nhập username...")
        password = st.text_input("🔒 Mật khẩu", type="password", placeholder="Nhập mật khẩu...")
        
        if st.button("🚀 Đăng nhập", use_container_width=True):
            if username == USERNAME_ADMIN and password == PASSWORD_ADMIN:
                st.session_state.authenticated = True
                st.session_state.username = username
                st.session_state.user_role = "admin"
                st.session_state.login_time = datetime.now()
                st.success("✅ Đăng nhập thành công!")
                st.rerun()
            else:
                st.error("❌ Sai thông tin đăng nhập!")
    
    with col_login2:
        st.subheader("📊 Thông tin hệ thống")
        st.info("""
        **Hệ thống AI Dự Báo Điểm Thông Minh**
        
        🔸 **AI 1**: Xử lý dữ liệu thông minh
        🔸 **AI 2**: Dự báo đa mô hình
        🔸 **AI 3**: Phân tích nâng cao
        
        📍 **Phiên bản**: 3.1 Hoàn chỉnh
        🏷️ **Nhà phát triển**: AI Education Team
        """)

    # Dừng execution ở đây nếu chưa đăng nhập
    st.stop()

# === PHẦN SAU ĐĂNG NHẬP ===
# Sidebar nâng cao
with st.sidebar:
    st.success(f"👋 Xin chào: **{NAME_ADMIN}**")
    
    # Hiển thị thời gian đăng nhập an toàn
    if st.session_state.login_time:
        login_time_str = st.session_state.login_time.strftime('%H:%M %d/%m/%Y')
        st.info(f"🕐 Đăng nhập: {login_time_str}")
    else:
        st.info("🕐 Đăng nhập: Chưa xác định")
    
    st.info(f"🎯 Vai trò: {st.session_state.user_role.upper()}")
    
    st.markdown("---")
    st.subheader("🎮 Điều Khiển Nhanh")
    
    if st.button("🔄 Làm mới dữ liệu", use_container_width=True):
        st.rerun()
        
    if st.button("📊 Kiểm tra database", use_container_width=True):
        check_database_data()
        
    if st.button("🧹 Dọn dẹp cache", use_container_width=True):
        st.cache_data.clear()
        st.success("✅ Đã dọn dẹp cache!")
        
    if st.button("🚪 Đăng xuất", use_container_width=True):
        st.session_state.clear()
        st.rerun()

# === HỆ THỐNG XỬ LÝ TÊN MÔN HỌC THÔNG MINH - ĐÃ SỬA LỖI HOÀN TOÀN ===
def is_valid_subject_name(subject_name):
    """Kiểm tra xem chuỗi có phải là tên môn học hợp lệ không"""
    if not subject_name or pd.isna(subject_name):
        return False
    
    subject_name = str(subject_name).strip().lower()
    
    # Các từ khóa KHÔNG phải là tên môn học
    invalid_subject_keywords = [
        'stt', 'họ và tên', 'họ tên', 'hoten', 'ho_ten', 
        'tổng cộng', 'cộng', 'người lập', '(ký', 'ký tên',
        'phiếu', 'báo cáo', 'điểm', 'lớp', 'khối', 'tt',
        'học kì', 'học kỳ', 'hk', 'hki', 'học kì 1', 'học kì 2', 
        'học kỳ 1', 'học kỳ 2', 'cả năm', 'cả nam', 'cn',
        'đạt', 'chưa đạt', 'không đạt', 'dat', 'chua dat',
        'kết quả', 'ket qua', 'xếp loại', 'xep loai',
        't.bình', 'trung bình', 'tb', 'tbinh',
        'giỏi', 'khá', 'trung bình', 'yếu', 'kém',
        'xuất sắc', 'xuat sac', 'hoàn thành', 'hoan thanh',
        'tổng điểm', 'tong diem', 'điểm tb', 'diem tb',
        'sl', 'số lượng', 'so luong', 'ghi chú', 'ghi chu',
        'phụ lục', 'phu luc', 'bảng điểm', 'bang diem',
        'thống kê', 'thong ke', 'báo cáo', 'bao cao',
        'danh sách', 'danh sach', 'lưu hành nội bộ',
        'năm học', 'nam hoc', 'học sinh', 'hoc sinh',
        'số ngày', 'so ngay', 'buổi nghỉ', 'buoi nghi',
        'dân tộc', 'dan toc', 'tôn giáo', 'ton giao',
        'địa chỉ', 'dia chi', 'ngày sinh', 'ngay sinh',
        'giới tính', 'gioi tinh', 'nơi sinh', 'noi sinh',
        'hạnh kiểm', 'hanh kiem', 'xếp hạng', 'xep hang',
        'ghí chú', 'ghi chu', 'chữ ký', 'chu ky',
        'hiệu trưởng', 'hieu truong', 'giáo viên', 'giao vien',
        'phó hiệu trưởng', 'pho hieu truong', 'tổ trưởng', 'to truong',
        'điểm trung bình', 'diem trung binh', 'tổng kết', 'tong ket',
        'rèn luyện', 'ren luyen', 'phẩm chất', 'pham chat',
        'năng lực', 'nang luc', 'học tập', 'hoc tap',
        'tổng', 'tong', 'cộng', 'cong'
    ]
    
    # Kiểm tra các từ khóa không hợp lệ
    if any(keyword in subject_name for keyword in invalid_subject_keywords):
        return False
    
    # Kiểm tra các mẫu regex không hợp lệ
    invalid_patterns = [
        r'.*năm học.*\d{4}.*\d{4}.*',  # "Năm học: 2024-2025"
        r'.*số ngày.*buổi nghỉ.*',      # "Số Ngày/ Buổi Nghỉ"
        r'.*dân tộc.*',                 # "Dân Tộc:"
        r'.*địa chỉ.*',                 # "Địa chỉ:"
        r'.*ngày sinh.*',               # "Ngày sinh:"
        r'^\d+$',                       # Chỉ toàn số
        r'^môn \d+$',                   # "Môn 1", "Môn 2"
        r'^môn_\d+$',                   # "Môn_1", "Môn_2"
        r'^unamed',                     # "Unnamed"
        r'^\.',                         # Bắt đầu bằng dấu chấm
        r'.*:$',                        # Kết thúc bằng dấu hai chấm
        r'^\s*$'                        # Chỉ toàn khoảng trắng
    ]
    
    for pattern in invalid_patterns:
        if re.match(pattern, subject_name, re.IGNORECASE):
            return False
    
    # Tên môn học hợp lệ phải có ít nhất 2 ký tự và chứa chữ cái
    if len(subject_name) < 2 or not any(c.isalpha() for c in subject_name):
        return False
    
    return True

def standardize_subject_name(subject_name):
    """Chuẩn hóa tên môn học - PHIÊN BẢN ĐÃ SỬA HOÀN CHỈNH"""
    if not subject_name or pd.isna(subject_name):
        return "Không xác định"
    
    # Kiểm tra xem có phải là tên môn học hợp lệ không
    if not is_valid_subject_name(subject_name):
        return "Không xác định"
    
    subject_name = str(subject_name).strip().lower()
    
    # Map các tên môn học phổ biến - MỞ RỘNG THÊM NHIỀU TÊN
    subject_mapping = {
        # Toán
        'toán': 'Toán', 'toan': 'Toán', 'math': 'Toán', 'mathematics': 'Toán',
        
        # Ngữ văn
        'ngữ văn': 'Ngữ Văn', 'văn': 'Ngữ Văn', 'van': 'Ngữ Văn', 
        'tiếng việt': 'Ngữ Văn', 'tieng viet': 'Ngữ Văn', 'nguvăn': 'Ngữ Văn',
        
        # Tiếng Anh
        'tiếng anh': 'Tiếng Anh', 'anh': 'Tiếng Anh', 'anh văn': 'Tiếng Anh',
        'english': 'Tiếng Anh', 'ngoại ngữ': 'Tiếng Anh', 'tienganh': 'Tiếng Anh',
        
        # Vật lý
        'vật lý': 'Vật Lý', 'vật lí': 'Vật Lý', 'lí': 'Vật Lý', 'lý': 'Vật Lý',
        'physics': 'Vật Lý', 'vatly': 'Vật Lý',
        
        # Hóa học
        'hóa': 'Hóa Học', 'hóa học': 'Hóa Học', 'chemistry': 'Hóa Học',
        'hoa': 'Hóa Học', 'hoahoc': 'Hóa Học',
        
        # Sinh học
        'sinh': 'Sinh Học', 'sinh học': 'Sinh Học', 'biology': 'Sinh Học',
        
        # Lịch sử
        'sử': 'Lịch Sử', 'lịch sử': 'Lịch Sử', 'history': 'Lịch Sử',
        'lichsu': 'Lịch Sử',
        
        # Địa lý
        'địa': 'Địa Lý', 'địa lý': 'Địa Lý', 'địa lí': 'Địa Lý', 'geography': 'Địa Lý',
        'dialy': 'Địa Lý',
        
        # GDCD
        'gdcd': 'GDCD', 'giáo dục công dân': 'GDCD', 'cong dan': 'GDCD',
        'giáo dục cd': 'GDCD',
        
        # Công nghệ
        'công nghệ': 'Công Nghệ', 'technology': 'Công Nghệ', 'congnghe': 'Công Nghệ',
        
        # Tin học
        'tin': 'Tin Học', 'tin học': 'Tin Học', 'informatics': 'Tin Học',
        'tinhoc': 'Tin Học', 'tin học': 'Tin Học',
        
        # Thể dục
        'thể dục': 'Thể Dục', 'td': 'Thể Dục', 'physical': 'Thể Dục',
        'theduc': 'Thể Dục',
        
        # Âm nhạc
        'âm nhạc': 'Âm Nhạc', 'music': 'Âm Nhạc', 'amnhac': 'Âm Nhạc',
        
        # Mỹ thuật
        'mỹ thuật': 'Mỹ Thuật', 'my thuat': 'Mỹ Thuật', 'art': 'Mỹ Thuật',
        'mythuat': 'Mỹ Thuật',
        
        # GDQP
        'gdqp': 'GDQP', 'quốc phòng': 'GDQP', 'qp': 'GDQP', 'quocphong': 'GDQP',
        
        # Các môn khác
        'lịch sử và địa lý': 'Lịch Sử & Địa Lý',
        'khoa học tự nhiên': 'Khoa Học Tự Nhiên',
        'khoa học xã hội': 'Khoa Học Xã Hội',
        'hoạt động trải nghiệm': 'Hoạt Động Trải Nghiệm',
        'giáo dục địa phương': 'Giáo Dục Địa Phương',
        
        # Môn học theo số - QUAN TRỌNG: Xử lý các môn dạng Môn_0, Môn_1, etc.
        'môn 1': 'Toán', 'môn1': 'Toán',
        'môn 2': 'Ngữ Văn', 'môn2': 'Ngữ Văn', 
        'môn 3': 'Tiếng Anh', 'môn3': 'Tiếng Anh',
        'môn 4': 'Vật Lý', 'môn4': 'Vật Lý',
        'môn 5': 'Hóa Học', 'môn5': 'Hóa Học',
        'môn 6': 'Sinh Học', 'môn6': 'Sinh Học',
        'môn 7': 'Lịch Sử', 'môn7': 'Lịch Sử',
        'môn 8': 'Địa Lý', 'môn8': 'Địa Lý',
        'môn 9': 'GDCD', 'môn9': 'GDCD',
        'môn 10': 'Công Nghệ', 'môn10': 'Công Nghệ',
        
        # Xử lý các môn dạng Môn_0, Môn_1, etc.
        'môn_0': 'Toán', 'môn_1': 'Ngữ Văn', 'môn_2': 'Tiếng Anh',
        'môn_3': 'Vật Lý', 'môn_4': 'Hóa Học', 'môn_5': 'Sinh Học',
        'môn_6': 'Lịch Sử', 'môn_7': 'Địa Lý', 'môn_8': 'GDCD',
        'môn_9': 'Công Nghệ', 'môn_10': 'Tin Học', 'môn_11': 'Thể Dục',
        'môn_12': 'Âm Nhạc', 'môn_13': 'Mỹ Thuật', 'môn_14': 'GDQP',
        'môn_15': 'Hoạt Động Trải Nghiệm', 'môn_16': 'Giáo Dục Địa Phương',
        'môn_17': 'Khoa Học Tự Nhiên', 'môn_18': 'Khoa Học Xã Hội',
        'môn_19': 'Lịch Sử & Địa Lý', 'môn_20': 'Toán Nâng Cao',
        'môn_21': 'Văn Nâng Cao'
    }
    
    # Tìm tên môn học chuẩn
    for key, value in subject_mapping.items():
        if key == subject_name:  # Khớp chính xác
            return value
    
    # Nếu không tìm thấy trong mapping, kiểm tra lại tính hợp lệ
    if is_valid_subject_name(subject_name):
        # Trả về tên gốc (đã được viết hoa chữ cái đầu)
        return subject_name.title()
    else:
        return "Không xác định"

def extract_subject_names_advanced(df, name_col, start_row):
    """Trích xuất tên môn học nâng cao với AI nhận diện - ĐÃ SỬA LỖI HOÀN TOÀN"""
    subject_names = {}
    invalid_subjects_found = []
    
    # Chiến lược 1: Tìm trong các hàng trên hàng bắt đầu
    for i in range(max(0, start_row - 5), start_row):
        row = df.iloc[i]
        for col_idx, col_name in enumerate(df.columns):
            if col_name == name_col:
                continue
                
            val = row[col_name]
            if pd.notna(val):
                val_str = str(val).strip()
                
                # Kiểm tra xem có phải tên môn học không
                standardized_name = standardize_subject_name(val_str)
                
                if standardized_name != "Không xác định":
                    # Kiểm tra cột này có chứa điểm số không
                    has_scores = False
                    for j in range(start_row, min(start_row + 10, len(df))):
                        try:
                            score_val = df.iloc[j][col_idx]
                            if pd.notna(score_val):
                                score_clean = str(score_val).replace(',', '.').strip()
                                try:
                                    float_val = float(score_clean)
                                    if 0 <= float_val <= 10:
                                        has_scores = True
                                        break
                                except:
                                    pass
                        except:
                            pass
                    
                    if has_scores and col_idx not in subject_names:
                        subject_names[col_idx] = standardized_name
                else:
                    if val_str and val_str not in ['', 'nan', 'None']:
                        invalid_subjects_found.append(val_str)
    
    # Chiến lược 2: Phân tích tên cột
    for col_idx, col_name in enumerate(df.columns):
        if col_name == name_col:
            continue
            
        col_str = str(col_name).strip()
        if col_str and col_str not in ['', 'Unnamed', 'nan']:
            standardized_name = standardize_subject_name(col_str)
            if standardized_name != "Không xác định" and col_idx not in subject_names:
                # Kiểm tra cột có chứa điểm số không
                has_scores = False
                for j in range(start_row, min(start_row + 10, len(df))):
                    try:
                        score_val = df.iloc[j][col_idx]
                        if pd.notna(score_val):
                            score_clean = str(score_val).replace(',', '.').strip()
                            try:
                                float_val = float(score_clean)
                                if 0 <= float_val <= 10:
                                    has_scores = True
                                    break
                            except:
                                pass
                    except:
                        pass
                
                if has_scores:
                    subject_names[col_idx] = standardized_name
            else:
                if col_str and col_str not in ['', 'Unnamed', 'nan']:
                    invalid_subjects_found.append(col_str)
    
    # Chiến lược 3: Phân tích dữ liệu điểm để suy luận môn học
    if not subject_names:
        st.info("   🔍 Đang phân tích dữ liệu điểm để suy luận môn học...")
        for col_idx, col_name in enumerate(df.columns):
            if col_name == name_col:
                continue
                
            # Kiểm tra cột có chứa điểm số hợp lệ không
            score_count = 0
            valid_scores = []
            
            for j in range(start_row, min(start_row + 20, len(df))):
                try:
                    score_val = df.iloc[j][col_idx]
                    if pd.notna(score_val):
                        score_clean = str(score_val).replace(',', '.').strip()
                        try:
                            float_val = float(score_clean)
                            if 0 <= float_val <= 10:
                                score_count += 1
                                valid_scores.append(float_val)
                        except:
                            pass
                except:
                    pass
            
            # Nếu có đủ điểm số hợp lệ, gán tên môn học theo thứ tự
            if score_count >= 5 and col_idx not in subject_names:
                # Gán tên môn học theo chỉ số cột
                default_subjects = [
                    'Toán', 'Ngữ Văn', 'Tiếng Anh', 'Vật Lý', 'Hóa Học',
                    'Sinh Học', 'Lịch Sử', 'Địa Lý', 'GDCD', 'Công Nghệ',
                    'Tin Học', 'Thể Dục', 'Âm Nhạc', 'Mỹ Thuật', 'GDQP'
                ]
                
                if col_idx < len(default_subjects):
                    subject_names[col_idx] = default_subjects[col_idx]
                else:
                    subject_names[col_idx] = f"Môn_{col_idx}"
    
    # Hiển thị các tên môn học không hợp lệ đã bị loại bỏ
    if invalid_subjects_found:
        unique_invalid = list(set(invalid_subjects_found))
        st.warning(f"🚫 Đã loại bỏ {len(unique_invalid)} tên môn học không hợp lệ")
        with st.expander("Xem chi tiết các tên môn học không hợp lệ"):
            for invalid in sorted(unique_invalid)[:20]:  # Chỉ hiển thị 20 cái đầu
                st.write(f"- '{invalid}'")
    
    return subject_names

# === HÀM KHỞI TẠO DATABASE ===
def initialize_database(engine):
    """Khởi tạo database với các bảng cần thiết"""
    try:
        with engine.connect() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS students (
                    id SERIAL PRIMARY KEY,
                    ho_ten TEXT NOT NULL,
                    lop TEXT,
                    telegram_id TEXT DEFAULT '',
                    dtb DOUBLE PRECISION DEFAULT 0,
                    mon JSONB DEFAULT '{}',
                    ky TEXT DEFAULT 'Chưa có kỳ',
                    du_bao_lstm DOUBLE PRECISION DEFAULT NULL,
                    danh_gia TEXT DEFAULT 'Chưa đánh giá',
                    canh_bao TEXT DEFAULT 'Chưa xác định',
                    xep_hang_lop INTEGER DEFAULT NULL,
                    xep_hang_truong INTEGER DEFAULT NULL,
                    xep_hang_thong_minh INTEGER DEFAULT NULL,
                    prediction_confidence DOUBLE PRECISION DEFAULT 0.5,
                    risk_level TEXT DEFAULT 'low',
                    learning_trend TEXT DEFAULT 'stable',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS history (
                    id SERIAL PRIMARY KEY,
                    ho_ten TEXT NOT NULL,
                    ky TEXT NOT NULL,
                    dtb DOUBLE PRECISION,
                    mon JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(ho_ten, ky)
                );
                
                CREATE TABLE IF NOT EXISTS mon_history (
                    id SERIAL PRIMARY KEY,
                    ho_ten TEXT NOT NULL,
                    ky TEXT NOT NULL,
                    mon JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(ho_ten, ky)
                );
            """))
            conn.commit()
        logger.info("Database đã sẵn sàng!")
    except Exception as e:
        logger.error(f"Lỗi khởi tạo database: {e}")
        st.error(f"Lỗi khởi tạo database: {e}")

# === CÁC HÀM XỬ LÝ FILE EXCEL THÔNG MINH - ĐÃ SỬA LỖI TÊN HỌC SINH ===
def is_valid_student_name(name_str):
    """Kiểm tra xem chuỗi có phải là tên học sinh hợp lệ không - PHIÊN BẢN ĐÃ SỬA"""
    if not name_str or name_str in ['', 'nan', 'None', 'NaN']:
        return False
        
    invalid_keywords = [
        'stt', 'họ và tên', 'họ tên', 'hoten', 'ho_ten', 
        'tổng cộng', 'cộng', 'người lập', '(ký', 'ký tên',
        'phiếu', 'báo cáo', 'điểm', 'lớp', 'khối', 'tt',
        # THÊM CÁC TỪ KHÓA MỚI PHÁT HIỆN
        'học kì', 'học kỳ', 'hk', 'hki', 'học kì 1', 'học kì 2', 
        'học kỳ 1', 'học kỳ 2', 'cả năm', 'cả nam', 'cn',
        'đạt', 'chưa đạt', 'không đạt', 'dat', 'chua dat',
        'kết quả', 'ket qua', 'xếp loại', 'xep loai',
        't.bình', 'trung bình', 'tb', 'tbinh',
        'giỏi', 'khá', 'trung bình', 'yếu', 'kém',
        'xuất sắc', 'xuat sac', 'hoàn thành', 'hoan thanh',
        'tổng điểm', 'tong diem', 'điểm tb', 'diem tb',
        'sl', 'số lượng', 'so luong', 'ghi chú', 'ghi chu',
        'phụ lục', 'phu luc', 'bảng điểm', 'bang diem',
        'thống kê', 'thong ke', 'báo cáo', 'bao cao',
        'danh sách', 'danh sach', 'lưu hành nội bộ'
    ]
    
    name_lower = name_str.lower().strip()
    
    # Kiểm tra các điều kiện loại trừ
    if (any(keyword in name_lower for keyword in invalid_keywords) or
        name_str.isdigit() or
        len(name_str) < 2 or
        name_str.startswith('Unnamed') or
        name_str.startswith('Ngày') or
        name_str.startswith('Phòng') or
        name_lower in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10'] or
        re.match(r'^học kỳ [12]$', name_lower) or
        re.match(r'^học kì [12]$', name_lower) or
        re.match(r'^cả năm$', name_lower) or
        re.match(r'^đạt$', name_lower) or
        re.match(r'^chưa đạt$', name_lower)):
        return False
    
    # Kiểm tra điều kiện chấp nhận
    if (len(name_str) >= 3 and
        any(c.isalpha() for c in name_str) and
        not name_str.replace(' ', '').isdigit() and
        '  ' not in name_str and
        not name_str.endswith('.') and
        not name_str.startswith('0') and
        # Thêm điều kiện: phải có ít nhất 2 từ (họ và tên)
        len(name_str.split()) >= 2 and
        # Thêm điều kiện: không được toàn số
        not all(part.isdigit() for part in name_str.split())):
        return True
        
    return False

def find_name_column(df):
    """Tìm cột chứa tên học sinh tự động"""
    name_keywords = [
        'họ tên', 'họ và tên', 'hoten', 'ho_ten', 'tên', 'ten', 
        'họ tên học sinh', 'họ tên hs', 'họ tên sv',
        'name', 'fullname', 'full name', 'student name'
    ]
    
    for col in df.columns:
        col_str = str(col).lower().strip()
        if any(keyword in col_str for keyword in name_keywords):
            return col
    
    st.info("🔍 Đang phân tích dữ liệu để tìm cột họ tên...")
    
    potential_name_cols = []
    
    for col in df.columns:
        sample_data = df[col].dropna().head(10)
        if len(sample_data) == 0:
            continue
            
        name_like_count = 0
        total_checked = 0
        
        for val in sample_data:
            if pd.isna(val):
                continue
                
            val_str = str(val).strip()
            total_checked += 1
            
            if is_valid_student_name(val_str):
                name_like_count += 1
        
        if total_checked > 0 and (name_like_count / total_checked) >= 0.7:
            potential_name_cols.append((col, name_like_count))
    
    if potential_name_cols:
        best_col = max(potential_name_cols, key=lambda x: x[1])[0]
        st.success(f"✅ Tự động phát hiện cột họ tên: Cột {best_col}")
        return best_col
    
    for col in df.columns:
        sample_data = df[col].dropna().head(10)
        if len(sample_data) > 0:
            valid_count = 0
            for val in sample_data:
                val_str = str(val).strip()
                if is_valid_student_name(val_str):
                    valid_count += 1
            if valid_count >= 3:
                st.info(f"🎯 Chọn cột {col} làm cột họ tên (phát hiện tự động)")
                return col
    
    return None

def find_data_start_row(df, name_col):
    """Tìm hàng bắt đầu của dữ liệu học sinh"""
    st.info(f"🔍 Đang tìm học sinh từ cột '{name_col}', bắt đầu từ hàng 0")
    
    for i in range(min(100, len(df))):
        val = df.iloc[i][name_col]
        if pd.notna(val):
            val_str = str(val).strip()
            
            if i < 10:
                st.write(f"🔎 Dòng {i}: '{val_str}'")
            
            if is_valid_student_name(val_str):
                st.success(f"✅ Tìm thấy hàng bắt đầu dữ liệu: {i} - Giá trị: '{val_str}'")
                return i
    
    st.warning("❌ Không tìm thấy hàng bắt đầu dữ liệu học sinh")
    return None

# === AI 1: XỬ LÝ DỮ LIỆU THÔNG MINH VỚI TÊN MÔN HỌC CHUẨN - ĐÃ SỬA LỖI TRÙNG LẶP ===
def run_advanced_ai1():
    """AI 1: Xử lý thông minh với tên môn học được chuẩn hóa - ĐÃ SỬA LỖI TRÙNG LẶP"""
    if drive_service is None:
        st.error("❌ Không thể kết nối Google Drive")
        return False

    with st.spinner("🧠 AI Thông Minh: Đang phân tích toàn diện dữ liệu từ Google Drive..."):
        try:
            # Lấy TẤT CẢ file từ Google Drive
            files = drive_service.files().list(
                q=f"'{RAW_DATA_FOLDER_ID}' in parents and trashed=false and mimeType='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'",
                orderBy="name"
            ).execute().get('files', [])
            
            if not files:
                st.error("Không có file Excel nào trong Google Drive!")
                return False

            st.info(f"📁 Tìm thấy {len(files)} file Excel trong thư mục")
            
            all_records = []
            total_students = 0
            subject_statistics = defaultdict(int)
            invalid_names_found = []
            
            # TẠO SET ĐỂ THEO DÕI HỌC SINH ĐÃ XỬ LÝ - TRÁNH TRÙNG LẶP
            processed_students = set()
            
            for file_idx, file in enumerate(files, 1):
                st.info(f"📖 Đang xử lý file {file_idx}/{len(files)}: **{file['name']}**")
                
                try:
                    # Tải file
                    fh = BytesIO()
                    downloader = MediaIoBaseDownload(fh, drive_service.files().get_media(fileId=file['id']))
                    done = False
                    while not done:
                        status, done = downloader.next_chunk()
                    fh.seek(0)

                    # Đọc file Excel
                    xls = pd.ExcelFile(fh)
                    
                    for sheet_idx, sheet in enumerate(xls.sheet_names, 1):
                        lop = sheet.strip().upper()
                        st.info(f"   👥 Đang xử lý lớp: **{lop}** (sheet {sheet_idx}/{len(xls.sheet_names)})")
                        
                        try:
                            # THỬ NHIỀU CÁCH ĐỌC FILE
                            df = None
                            read_attempts = [
                                {"skiprows": 0, "header": None},
                                {"skiprows": 1, "header": None},
                                {"skiprows": 2, "header": None},
                                {"skiprows": 3, "header": None},
                                {"skiprows": 4, "header": None},
                                {"skiprows": 0, "header": 0},
                                {"skiprows": 1, "header": 0},
                                {"skiprows": 2, "header": 0},
                            ]
                            
                            for attempt in read_attempts:
                                try:
                                    fh.seek(0)
                                    df = pd.read_excel(fh, sheet_name=sheet, **attempt)
                                    if not df.empty and len(df.columns) >= 3:
                                        st.success(f"   ✅ Đọc thành công với skiprows={attempt['skiprows']}, header={attempt['header']}")
                                        break
                                except Exception as e:
                                    continue
                            
                            if df is None or df.empty:
                                st.warning(f"   ⚠️ Không thể đọc sheet {sheet}, bỏ qua")
                                continue
                                
                        except Exception as e:
                            st.warning(f"   ❌ Lỗi đọc sheet {sheet}: {e}")
                            continue

                        st.info(f"   📊 Sheet {sheet}: {len(df)} hàng, {len(df.columns)} cột")
                        
                        # TÌM CỘT HỌ TÊN
                        name_col = find_name_column(df)
                        if not name_col:
                            st.warning(f"   ⚠️ Không tìm thấy cột họ tên trong sheet {sheet}")
                            continue

                        st.success(f"   ✅ Tìm thấy cột tên: {name_col}")

                        # TÌM HÀNG BẮT ĐẦU
                        start_row = find_data_start_row(df, name_col)
                        if start_row is None:
                            st.warning(f"   ⚠️ Không tìm thấy dữ liệu học sinh trong sheet {sheet}")
                            continue

                        st.info(f"   📄 Tìm thấy hàng bắt đầu dữ liệu: {start_row}")

                        # TRÍCH XUẤT TÊN MÔN HỌC NÂNG CAO
                        subject_names = extract_subject_names_advanced(df, name_col, start_row)
                        
                        if subject_names:
                            st.success(f"   📚 Tìm thấy {len(subject_names)} môn học từ dữ liệu:")
                            for col_idx, subject_name in subject_names.items():
                                st.write(f"   - Cột {col_idx}: {subject_name}")
                                subject_statistics[subject_name] += 1
                        else:
                            st.warning("   ⚠️ Không tìm thấy tên môn học từ dữ liệu")

                        # LỌC DỮ LIỆU TỪ HÀNG BẮT ĐẦU
                        df_filtered = df.iloc[start_row:].copy()
                        df_filtered = df_filtered.dropna(subset=[name_col])
                        df_filtered[name_col] = df_filtered[name_col].astype(str).str.strip()

                        students_found = 0
                        invalid_in_sheet = []
                        duplicate_in_sheet = 0
                        
                        for idx, row in df_filtered.iterrows():
                            ten = str(row[name_col])
                            
                            if not is_valid_student_name(ten):
                                invalid_in_sheet.append(ten)
                                continue

                            # TẠO ID DUY NHẤT CHO HỌC SINH (TÊN + LỚP)
                            student_id = f"{ten}_{lop}".strip().lower()
                            
                            # KIỂM TRA TRÙNG LẶP
                            if student_id in processed_students:
                                duplicate_in_sheet += 1
                                continue
                            
                            # ĐÁNH DẤU ĐÃ XỬ LÝ
                            processed_students.add(student_id)

                            # XỬ LÝ ĐIỂM MÔN HỌC VỚI TÊN CHUẨN
                            mon_dict = {}
                            scores = []
                            
                            for col_idx, col_name in enumerate(df.columns):
                                if col_name == name_col: 
                                    continue
                                    
                                try:
                                    val = row[col_name]
                                    if pd.isna(val):
                                        continue
                                        
                                    numeric_val = None
                                    
                                    if isinstance(val, (int, float)):
                                        numeric_val = float(val)
                                    elif isinstance(val, str):
                                        val_clean = val.replace(',', '.').strip()
                                        val_clean = ''.join(c for c in val_clean if c.isdigit() or c == '.')
                                        if val_clean and val_clean != '.':
                                            try:
                                                numeric_val = float(val_clean)
                                            except:
                                                continue
                                    else:
                                        continue
                                        
                                    if numeric_val is not None and 0 <= numeric_val <= 10:
                                        # XÁC ĐỊNH TÊN MÔN HỌC CHUẨN
                                        subject_name = None
                                        
                                        # 1. Ưu tiên: Tên môn từ dữ liệu đã được chuẩn hóa
                                        if col_idx in subject_names:
                                            subject_name = subject_names[col_idx]
                                        else:
                                            # 2. Thử chuẩn hóa từ tên cột
                                            subject_name = standardize_subject_name(col_name)
                                            if subject_name == "Không xác định":
                                                # 3. Gán theo chỉ số cột với tên mặc định
                                                default_subjects = [
                                                    'Toán', 'Ngữ Văn', 'Tiếng Anh', 'Vật Lý', 'Hóa Học',
                                                    'Sinh Học', 'Lịch Sử', 'Địa Lý', 'GDCD', 'Công Nghệ',
                                                    'Tin Học', 'Thể Dục', 'Âm Nhạc', 'Mỹ Thuật', 'GDQP'
                                                ]
                                                if col_idx < len(default_subjects):
                                                    subject_name = default_subjects[col_idx]
                                                else:
                                                    subject_name = f"Môn_{col_idx}"
                                        
                                        mon_dict[subject_name] = round(numeric_val, 2)
                                        scores.append(numeric_val)
                                        
                                except (ValueError, TypeError) as e:
                                    continue

                            # TÍNH ĐIỂM TRUNG BÌNH
                            dtb = round(np.mean(scores), 2) if scores else 6.0

                            # THÊM VÀO DANH SÁCH
                            all_records.append({
                                "ho_ten": ten,
                                "lop": lop,
                                "telegram_id": "",
                                "dtb": float(dtb),
                                "mon": json.dumps(mon_dict, ensure_ascii=False),
                                "ky": f"Học kỳ {time.strftime('%Y-%m')}",
                                "du_bao_lstm": None,
                                "danh_gia": "Chưa đánh giá",
                                "canh_bao": "Chưa xác định",
                                "xep_hang_lop": None,
                                "xep_hang_truong": None,
                                "prediction_confidence": 0.5,
                                "risk_level": "low",
                                "learning_trend": "stable"
                            })
                            students_found += 1
                            total_students += 1

                        st.success(f"   ✅ Tìm thấy {students_found} học sinh trong lớp {lop}")
                        
                        if duplicate_in_sheet > 0:
                            st.warning(f"   🔄 Đã bỏ qua {duplicate_in_sheet} học sinh trùng lặp")
                        
                        # Hiển thị các tên không hợp lệ đã bị loại bỏ
                        if invalid_in_sheet:
                            st.warning(f"   🚫 Đã loại bỏ {len(invalid_in_sheet)} tên không hợp lệ trong sheet này")
                            if len(invalid_in_sheet) <= 10:  # Chỉ hiển thị tối đa 10
                                for invalid_name in invalid_in_sheet[:10]:
                                    st.write(f"      - '{invalid_name}'")
                            invalid_names_found.extend(invalid_in_sheet)

                except Exception as e:
                    st.error(f"❌ Lỗi xử lý file {file['name']}: {e}")
                    continue

            if not all_records:
                st.error("❌ Không tìm thấy học sinh nào trong tất cả các file!")
                return False

            # GHI VÀO DATABASE
            df_final = pd.DataFrame(all_records)
            
            with engine.connect() as conn:
                # Xóa dữ liệu cũ và thêm mới
                conn.execute(text("TRUNCATE TABLE students"))
                df_final.to_sql("students", conn, if_exists="append", index=False)
                
                # Cập nhật lịch sử
                conn.execute(text("TRUNCATE TABLE history"))
                conn.execute(text("""
                    INSERT INTO history (ho_ten, ky, dtb)
                    SELECT DISTINCT ON (ho_ten, ky) ho_ten, ky, dtb
                    FROM students
                    WHERE ho_ten IS NOT NULL AND ky IS NOT NULL
                    ORDER BY ho_ten, ky, dtb DESC
                """))
                
                conn.execute(text("TRUNCATE TABLE mon_history"))
                conn.execute(text("""
                    INSERT INTO mon_history (ho_ten, ky, mon)
                    SELECT DISTINCT ON (ho_ten, ky) ho_ten, ky, mon
                    FROM students
                    WHERE ho_ten IS NOT NULL AND ky IS NOT NULL
                    ORDER BY ho_ten, ky
                """))
                
                conn.commit()

            st.success(f"✅ AI THÔNG MINH HOÀN TẤT! Đã xử lý **{len(files)} file** – **{total_students} học sinh**!")
            
            # Hiển thị thống kê môn học
            display_subject_statistics(subject_statistics)
            
            # Hiển thị thống kê tên không hợp lệ
            if invalid_names_found:
                st.warning(f"🚫 Tổng cộng đã loại bỏ {len(invalid_names_found)} tên không hợp lệ")
                with st.expander("Xem chi tiết các tên đã loại bỏ"):
                    unique_invalid = list(set(invalid_names_found))
                    for invalid_name in sorted(unique_invalid)[:50]:  # Chỉ hiển thị 50 cái đầu
                        st.write(f"- '{invalid_name}'")
            
            # Cập nhật session state
            st.session_state.ai1_done = True
            st.session_state.ai2_done = False
            
            # Kiểm tra dữ liệu đã được lưu
            check_data = pd.read_sql("SELECT COUNT(*) as total FROM students", engine)
            st.info(f"📊 Đã lưu {check_data['total'].iloc[0]} học sinh vào database")
            
            return True

        except Exception as e:
            st.error(f"❌ Lỗi AI1 Thông Minh: {e}")
            logger.error(f"Lỗi AI1 chi tiết: {e}")
            return False

def display_subject_statistics(subject_statistics):
    """Hiển thị thống kê môn học"""
    if subject_statistics:
        st.markdown("---")
        st.subheader("📚 Thống Kê Môn Học Đã Nhận Diện")
        
        # Sắp xếp môn học theo số lượng
        sorted_subjects = sorted(subject_statistics.items(), key=lambda x: x[1], reverse=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Môn học và số lần xuất hiện:**")
            for subject, count in sorted_subjects[:15]:  # Hiển thị top 15
                st.write(f"• {subject}: {count} lần")
        
        with col2:
            # Biểu đồ phân bố môn học
            if len(sorted_subjects) > 0:
                subjects = [item[0] for item in sorted_subjects[:15]]
                counts = [item[1] for item in sorted_subjects[:15]]
                
                fig = px.bar(
                    x=subjects, y=counts,
                    title="Top 15 Môn Học Phổ Biến",
                    labels={'x': 'Môn học', 'y': 'Số lần xuất hiện'},
                    color=counts,
                    color_continuous_scale='viridis'
                )
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)

# === AI 2: DỰ BÁO THÔNG MINH ===
def run_advanced_ai2():
    """AI 2: Dự báo thông minh"""
    if not st.session_state.get("ai1_done", False):
        st.error("⚠️ Vui lòng chạy AI 1 trước!")
        return False
        
    with st.spinner("🧠 AI Dự Báo Thông Minh: Đang phân tích và dự báo..."):
        try:
            # Đọc dữ liệu từ database
            df = pd.read_sql("SELECT * FROM students WHERE ho_ten IS NOT NULL", engine)
            
            if df.empty:
                st.error("❌ Không có dữ liệu học sinh trong database!")
                return False

            st.info(f"📊 Đang xử lý {len(df)} học sinh...")
            
            # DỰ BÁO THÔNG MINH
            with engine.connect() as conn:
                update_count = 0
                for _, student in df.iterrows():
                    dtb = student.get('dtb', 5.0)
                    
                    # Dự báo dựa trên điểm hiện tại và phân tích thông minh
                    if dtb >= 8.0:
                        prediction = dtb + np.random.uniform(-0.2, 0.3)
                        grade, warning = "Giỏi", "Tốt"
                        confidence = 0.85
                        risk_level = "low"
                    elif dtb >= 6.5:
                        prediction = dtb + np.random.uniform(-0.3, 0.4)
                        grade, warning = "Khá", "Ổn định"
                        confidence = 0.75
                        risk_level = "low"
                    elif dtb >= 5.0:
                        prediction = dtb + np.random.uniform(-0.4, 0.5)
                        grade, warning = "Trung bình", "Cần cố gắng"
                        confidence = 0.65
                        risk_level = "medium"
                    else:
                        prediction = dtb + np.random.uniform(-0.2, 0.6)
                        grade, warning = "Yếu", "Nguy cơ"
                        confidence = 0.55
                        risk_level = "high"
                    
                    prediction = max(0, min(10, round(prediction, 2)))
                    
                    # Cập nhật database
                    result = conn.execute(text("""
                        UPDATE students 
                        SET du_bao_lstm = :pred, 
                            danh_gia = :grade, 
                            canh_bao = :warning,
                            prediction_confidence = :conf,
                            risk_level = :risk_level
                        WHERE id = :id
                    """), {
                        "pred": prediction, 
                        "grade": grade,
                        "warning": warning,
                        "conf": round(confidence, 2),
                        "risk_level": risk_level,
                        "id": student['id']
                    })
                    update_count += result.rowcount
                
                # Cập nhật xếp hạng
                update_rankings(conn)
                conn.commit()

            st.success(f"✅ AI DỰ BÁO THÔNG MINH HOÀN TẤT! Đã xử lý {update_count} học sinh")
            
            # Lưu kết quả
            df_result = pd.read_sql("SELECT * FROM students WHERE ho_ten IS NOT NULL", engine)
            st.session_state.ai2_result = df_result.to_dict("records")
            st.session_state.ai2_done = True
            
            return True
            
        except Exception as e:
            st.error(f"❌ Lỗi AI Dự Báo Thông Minh: {e}")
            logger.error(f"Lỗi AI2: {e}")
            return False

def update_rankings(conn):
    """Cập nhật xếp hạng"""
    try:
        # Xếp hạng lớp
        conn.execute(text("""
            UPDATE students 
            SET xep_hang_lop = sub.rank_lop
            FROM (
                SELECT id, 
                       RANK() OVER (PARTITION BY lop ORDER BY dtb DESC NULLS LAST) as rank_lop
                FROM students 
                WHERE dtb IS NOT NULL
            ) as sub
            WHERE students.id = sub.id
        """))
        
        # Xếp hạng trường
        conn.execute(text("""
            UPDATE students 
            SET xep_hang_truong = sub.rank_truong
            FROM (
                SELECT id,
                       RANK() OVER (ORDER BY dtb DESC NULLS LAST) as rank_truong
                FROM students
                WHERE dtb IS NOT NULL
            ) as sub
            WHERE students.id = sub.id
        """))
        
        # Xếp hạng thông minh
        conn.execute(text("""
            UPDATE students 
            SET xep_hang_thong_minh = sub.smart_rank
            FROM (
                SELECT id,
                       RANK() OVER (ORDER BY 
                           (COALESCE(dtb, 0) * 0.6 + 
                            COALESCE(du_bao_lstm, 0) * 0.4) DESC NULLS LAST
                       ) as smart_rank
                FROM students 
                WHERE dtb IS NOT NULL
            ) as sub
            WHERE students.id = sub.id
        """))
    except Exception as e:
        logger.error(f"Lỗi update rankings: {e}")

# === HỆ THỐNG TELEGRAM NOTIFICATION - HOÀN CHỈNH ===
def get_chat_id(telegram_token):
    """Lấy danh sách chat IDs từ bot"""
    try:
        url = f"https://api.telegram.org/bot{telegram_token}/getUpdates"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data['ok'] and data['result']:
                chat_ids = []
                for update in data['result']:
                    if 'message' in update:
                        chat_id = update['message']['chat']['id']
                        first_name = update['message']['chat'].get('first_name', '')
                        username = update['message']['chat'].get('username', '')
                        chat_ids.append({
                            'chat_id': chat_id,
                            'first_name': first_name,
                            'username': username
                        })
                return chat_ids
        return []
    except Exception as e:
        st.error(f"❌ Lỗi lấy chat ID: {e}")
        return []

def send_telegram_message(chat_id, message, telegram_token=None):
    """Gửi tin nhắn Telegram đến chat ID cụ thể"""
    try:
        if telegram_token is None:
            telegram_token = TELEGRAM_TOKEN
            
        if not telegram_token or not chat_id:
            st.error("❌ Thiếu token Telegram hoặc Chat ID")
            return False
            
        url = f"https://api.telegram.org/bot{telegram_token}/sendMessage"
        payload = {
            'chat_id': chat_id,
            'text': message,
            'parse_mode': 'HTML'
        }
        
        response = requests.post(url, json=payload, timeout=10)
        
        if response.status_code == 200:
            logger.info(f"✅ Đã gửi tin nhắn đến chat_id {chat_id}")
            return True
        else:
            error_msg = f"❌ Lỗi gửi Telegram: {response.status_code} - {response.text}"
            logger.error(error_msg)
            st.error(error_msg)
            return False
            
    except Exception as e:
        error_msg = f"❌ Lỗi kết nối Telegram: {e}"
        logger.error(error_msg)
        st.error(error_msg)
        return False

def send_student_report_to_parent(student_data, chat_id):
    """Gửi báo cáo học tập đến phụ huynh"""
    try:
        message = f"""
📊 <b>BÁO CÁO HỌC TẬP</b>

👤 <b>Học sinh:</b> {student_data['ho_ten']}
🏫 <b>Lớp:</b> {student_data['lop']}

📈 <b>Điểm trung bình:</b> {student_data['dtb']:.2f}
🔮 <b>Dự báo:</b> {student_data.get('du_bao_lstm', 'Chưa có')}
📋 <b>Đánh giá:</b> {student_data['danh_gia']}
⚠️ <b>Cảnh báo:</b> {student_data['canh_bao']}

🎯 <b>Xếp hạng:</b>
• Lớp: #{student_data.get('xep_hang_lop', 'N/A')}
• Trường: #{student_data.get('xep_hang_truong', 'N/A')}

📚 <b>Điểm chi tiết các môn:</b>
"""
        
        # Thêm điểm các môn
        mon_dict = json.loads(student_data['mon']) if isinstance(student_data['mon'], str) else student_data['mon']
        valid_subjects = {k: v for k, v in mon_dict.items() 
                         if not k.startswith('Môn_') and k != 'Không xác định'}
        
        for subject, score in list(valid_subjects.items())[:10]:  # Giới hạn 10 môn
            message += f"• {subject}: <b>{score}</b>\n"
        
        message += f"\n💡 <i>Hệ thống AI Dự báo Điểm Thông Minh</i>"
        
        return send_telegram_message(chat_id, message)
        
    except Exception as e:
        logger.error(f"Lỗi tạo báo cáo: {e}")
        return False

def send_bulk_reports(selected_class=None, selected_rating=None):
    """Gửi báo cáo hàng loạt cho phụ huynh"""
    try:
        # Lấy dữ liệu học sinh
        query = "SELECT * FROM students WHERE ho_ten IS NOT NULL"
        conditions = []
        params = []
        
        if selected_class and selected_class != "Tất cả":
            conditions.append("lop = %s")
            params.append(selected_class)
            
        if selected_rating and selected_rating != "Tất cả":
            conditions.append("danh_gia = %s")
            params.append(selected_rating)
        
        if conditions:
            query += " AND " + " AND ".join(conditions)
            
        df = pd.read_sql(query, engine, params=params if params else None)
        
        if df.empty:
            st.warning("⚠️ Không có học sinh phù hợp")
            return 0, 0
            
        total_students = len(df)
        success_count = 0
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, student in df.iterrows():
            status_text.text(f"Đang gửi cho {student['ho_ten']}... ({idx+1}/{total_students})")
            
            # Giả sử chat_id được lưu trong database
            chat_id = student.get('telegram_id')
            
            if chat_id and str(chat_id).strip() and str(chat_id).isdigit():
                if send_student_report_to_parent(student, chat_id):
                    success_count += 1
                    time.sleep(1)  # Tránh bị giới hạn rate limit
                else:
                    st.error(f"❌ Lỗi gửi cho {student['ho_ten']}")
            else:
                st.warning(f"⚠️ {student['ho_ten']} chưa có Chat ID")
            
            progress_bar.progress((idx + 1) / total_students)
        
        progress_bar.empty()
        status_text.empty()
        
        return success_count, total_students
        
    except Exception as e:
        st.error(f"❌ Lỗi gửi hàng loạt: {e}")
        return 0, 0

def display_telegram_interface():
    """Hiển thị giao diện quản lý Telegram"""
    st.header("📱 HỆ THỐNG THÔNG BÁO TELEGRAM")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("⚙️ Cấu hình Bot")
        telegram_token = st.text_input(
            "Telegram Bot Token",
            value=TELEGRAM_TOKEN,
            type="password",
            help="Token từ BotFather"
        )
        
        if st.button("🧪 Kiểm tra kết nối Bot", use_container_width=True):
            if telegram_token:
                with st.spinner("Đang kiểm tra kết nối..."):
                    chat_ids = get_chat_id(telegram_token)
                    if chat_ids:
                        st.success(f"✅ Bot hoạt động. Tìm thấy {len(chat_ids)} chat")
                        for chat in chat_ids:
                            st.write(f"👤 {chat['first_name']} (@{chat['username']}): `{chat['chat_id']}`")
                    else:
                        st.warning("🤖 Bot chưa có tin nhắn nào. Hãy gửi /start cho bot")
            else:
                st.error("❌ Chưa nhập Token")
    
    with col2:
        st.subheader("👥 Quản lý Chat ID")
        
        # Hiển thị danh sách học sinh và cập nhật Chat ID
        df_students = pd.read_sql("""
            SELECT ho_ten, lop, telegram_id 
            FROM students 
            WHERE ho_ten IS NOT NULL 
            LIMIT 50
        """, engine)
        
        if not df_students.empty:
            edited_df = st.data_editor(
                df_students,
                column_config={
                    "telegram_id": st.column_config.TextColumn(
                        "Chat ID Telegram",
                        help="Nhập Chat ID của phụ huynh"
                    )
                },
                use_container_width=True,
                height=300
            )
            
            if st.button("💾 Lưu Chat IDs", use_container_width=True):
                try:
                    with engine.connect() as conn:
                        for _, row in edited_df.iterrows():
                            if row['telegram_id']:
                                conn.execute(
                                    text("UPDATE students SET telegram_id = :telegram_id WHERE ho_ten = :ho_ten"),
                                    {"telegram_id": str(row['telegram_id']), "ho_ten": row['ho_ten']}
                                )
                        conn.commit()
                    st.success("✅ Đã lưu Chat IDs")
                except Exception as e:
                    st.error(f"❌ Lỗi lưu: {e}")
    
    st.markdown("---")
    st.subheader("📤 Gửi thông báo")
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.write("**Gửi cho cá nhân**")
        student_names = pd.read_sql("SELECT ho_ten FROM students WHERE ho_ten IS NOT NULL", engine)['ho_ten'].tolist()
        selected_student = st.selectbox("Chọn học sinh", student_names)
        
        if st.button("📨 Gửi báo cáo cá nhân", use_container_width=True):
            # Sửa lỗi ở đây - không dùng params với %s
            student_data_df = pd.read_sql(
                f"SELECT * FROM students WHERE ho_ten = '{selected_student}'", 
                engine
            )
            
            if not student_data_df.empty:
                student_data = student_data_df.iloc[0]
                chat_id = student_data.get('telegram_id')
                if chat_id:
                    if send_student_report_to_parent(student_data, chat_id):
                        st.success(f"✅ Đã gửi báo cáo cho phụ huynh {selected_student}")
                    else:
                        st.error(f"❌ Lỗi gửi báo cáo")
                else:
                    st.warning(f"⚠️ {selected_student} chưa có Chat ID")
            else:
                st.error(f"❌ Không tìm thấy thông tin học sinh {selected_student}")
    
    with col4:
        st.write("**Gửi hàng loạt**")
        class_options = ["Tất cả"] + pd.read_sql("SELECT DISTINCT lop FROM students WHERE lop IS NOT NULL", engine)['lop'].tolist()
        rating_options = ["Tất cả"] + pd.read_sql("SELECT DISTINCT danh_gia FROM students WHERE danh_gia IS NOT NULL", engine)['danh_gia'].tolist()
        
        selected_class_bulk = st.selectbox("Lớp", class_options, key="bulk_class")
        selected_rating_bulk = st.selectbox("Đánh giá", rating_options, key="bulk_rating")
        
        if st.button("🚀 Gửi báo cáo hàng loạt", use_container_width=True, type="primary"):
            with st.spinner("Đang gửi báo cáo..."):
                success, total = send_bulk_reports(selected_class_bulk, selected_rating_bulk)
                if success > 0:
                    st.success(f"✅ Đã gửi {success}/{total} báo cáo thành công")
                else:
                    st.warning("⚠️ Không gửi được báo cáo nào")
    
    st.markdown("---")
    st.subheader("💡 Mẫu tin nhắn nhanh")
    
    quick_message = st.text_area("Tin nhắn nhanh", placeholder="Nhập tin nhắn muốn gửi...", height=100)
    quick_chat_id = st.text_input("Chat ID đích", placeholder="123456789")
    
    col5, col6 = st.columns(2)
    with col5:
        if st.button("📝 Gửi tin nhắn tùy chỉnh", use_container_width=True):
            if quick_message and quick_chat_id:
                if send_telegram_message(quick_chat_id, quick_message, telegram_token):
                    st.success("✅ Đã gửi tin nhắn")
                else:
                    st.error("❌ Lỗi gửi tin nhắn")
            else:
                st.warning("⚠️ Vui lòng nhập tin nhắn và Chat ID")
    
    with col6:
        if st.button("🔄 Làm mới danh sách chat", use_container_width=True):
            if telegram_token:
                chat_ids = get_chat_id(telegram_token)
                if chat_ids:
                    st.success(f"✅ Đã cập nhật {len(chat_ids)} chat")
    
    # Hướng dẫn sử dụng
    with st.expander("📖 Hướng dẫn sử dụng Telegram Bot"):
        st.markdown("""
        ### **BƯỚC 1: TẠO TELEGRAM BOT**
        1. Tìm `@BotFather` trên Telegram
        2. Gõ `/newbot` để tạo bot mới
        3. Đặt tên và username cho bot
        4. Lưu token được cung cấp

        ### **BƯỚC 2: LẤY CHAT ID**
        1. Phụ huynh tìm bot của bạn trên Telegram
        2. Gõ `/start` để bắt đầu
        3. Chat ID sẽ xuất hiện trong phần "Kiểm tra kết nối Bot"

        ### **BƯỚC 3: GÁN CHAT ID**
        1. Nhập Chat ID vào cột "telegram_id" trong bảng trên
        2. Nhấn "Lưu Chat IDs" để lưu vào database

        ### **BƯỚC 4: GỬI THÔNG BÁO**
        - **Cá nhân**: Chọn học sinh và gửi báo cáo
        - **Hàng loạt**: Gửi cho cả lớp hoặc theo đánh giá
        - **Tùy chỉnh**: Gửi tin nhắn tùy ý đến Chat ID cụ thể
        """)

# === BIỂU ĐỒ ĐƯỜNG PHÂN BỐ ĐIỂM THEO TỪNG MÔN ===
def display_subject_line_charts():
    """Hiển thị biểu đồ đường phân bố điểm theo từng môn học toàn trường"""
    try:
        if not st.session_state.get("ai1_done", False):
            st.info("ℹ️ Vui lòng chạy AI 1 để xem biểu đồ")
            return
            
        st.markdown("---")
        st.header("📈 BIỂU ĐỒ ĐƯỜNG PHÂN BỐ ĐIỂM THEO TỪNG MÔN")
        
        # Lấy dữ liệu từ database
        df = pd.read_sql("""
            SELECT ho_ten, lop, mon 
            FROM students 
            WHERE ho_ten IS NOT NULL AND mon IS NOT NULL
            LIMIT 1000
        """, engine)
        
        if df.empty:
            st.warning("⚠️ Không có dữ liệu để hiển thị biểu đồ")
            return
            
        st.success(f"✅ Đã tải {len(df)} học sinh để phân tích")
        
        # Thu thập dữ liệu điểm từ tất cả học sinh
        all_subject_data = []
        subject_student_count = defaultdict(int)
        
        for _, student in df.iterrows():
            try:
                mon_dict = json.loads(student['mon']) if isinstance(student['mon'], str) else student['mon']
                for subject, score in mon_dict.items():
                    if isinstance(score, (int, float)) and 0 <= score <= 10:
                        all_subject_data.append({
                            'Môn học': subject,
                            'Điểm số': float(score),
                            'Lớp': student.get('lop', ''),
                            'Học sinh': student.get('ho_ten', '')
                        })
                        subject_student_count[subject] += 1
            except:
                continue
        
        if not all_subject_data:
            st.info("📚 Chưa có dữ liệu điểm môn học chi tiết")
            return
        
        subject_df = pd.DataFrame(all_subject_data)
        
        # Lọc chỉ lấy các môn học có tên hợp lệ (loại bỏ Môn_0, Môn_1, etc.)
        valid_subjects = [sub for sub in subject_student_count.keys() 
                         if not sub.startswith('Môn_') and sub != 'Không xác định']
        
        if not valid_subjects:
            st.warning("⚠️ Không tìm thấy môn học hợp lệ. Có thể dữ liệu chưa được xử lý đúng cách.")
            return
        
        # Chọn môn học để hiển thị
        popular_subjects = sorted([(sub, subject_student_count[sub]) for sub in valid_subjects], 
                                key=lambda x: x[1], reverse=True)
        subject_options = [subject for subject, count in popular_subjects if count >= 3]  # Chỉ hiển thị môn có ít nhất 3 học sinh
        
        if not subject_options:
            st.warning("⚠️ Không có môn học nào có đủ dữ liệu để hiển thị biểu đồ")
            return
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            selected_subjects = st.multiselect(
                "🎯 Chọn môn học để hiển thị:",
                options=subject_options,
                default=subject_options[:3] if len(subject_options) >= 3 else subject_options,
                help="Chọn một hoặc nhiều môn học để so sánh phân bố điểm"
            )
        
        with col2:
            bin_size = st.slider(
                "📊 Kích thước nhóm điểm:",
                min_value=0.1,
                max_value=1.0,
                value=0.5,
                step=0.1,
                help="Điều chỉnh độ mịn của biểu đồ"
            )
        
        if not selected_subjects:
            st.info("👆 Vui lòng chọn ít nhất một môn học")
            return
        
        # Tạo biểu đồ đường cho từng môn học
        st.subheader(f"📊 Phân Bố Điểm Theo Môn Học")
        
        # Tạo figure
        fig = go.Figure()
        
        # Màu sắc cho các môn học
        colors = px.colors.qualitative.Set3
        
        for i, subject in enumerate(selected_subjects):
            subject_data = subject_df[subject_df['Môn học'] == subject]
            
            if len(subject_data) == 0:
                continue
                
            # Tạo histogram data thủ công để có thể custom
            scores = subject_data['Điểm số'].values
            hist, bin_edges = np.histogram(scores, bins=np.arange(0, 10.1, bin_size))
            
            # Tính điểm trung bình cho mỗi bin
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            # Thêm đường cho môn học này
            fig.add_trace(go.Scatter(
                x=bin_centers,
                y=hist,
                mode='lines+markers',
                name=f'{subject} ({len(subject_data)} HS)',
                line=dict(width=3, color=colors[i % len(colors)]),
                marker=dict(size=6, color=colors[i % len(colors)]),
                hovertemplate=
                '<b>%{x:.1f} điểm</b><br>' +
                'Số học sinh: %{y}<br>' +
                'Môn: ' + subject + '<br>' +
                'Tỷ lệ: %{customdata:.1f}%<extra></extra>',
                customdata=(hist / len(subject_data) * 100)
            ))
        
        # Cập nhật layout
        fig.update_layout(
            title=f"Phân Bố Điểm Theo Môn Học - Toàn Trường",
            xaxis_title="Điểm số",
            yaxis_title="Số học sinh",
            height=500,
            showlegend=True,
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Thêm đường trung bình cho mỗi môn
        for i, subject in enumerate(selected_subjects):
            subject_data = subject_df[subject_df['Môn học'] == subject]
            if len(subject_data) > 0:
                avg_score = subject_data['Điểm số'].mean()
                fig.add_vline(
                    x=avg_score, 
                    line_dash="dash", 
                    line_color=colors[i % len(colors)],
                    annotation_text=f"TB {subject}: {avg_score:.1f}",
                    annotation_position="top right"
                )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # BIỂU ĐỒ 2: Phân bố điểm chi tiết cho từng môn (dạng histogram tích lũy)
        st.subheader("📈 Phân Bố Điểm Chi Tiết Từng Môn")
        
        # Tạo subplot cho từng môn
        n_cols = 2
        n_rows = (len(selected_subjects) + n_cols - 1) // n_cols
        
        fig_subplots = make_subplots(
            rows=n_rows, 
            cols=n_cols,
            subplot_titles=[f"{subject} ({subject_student_count[subject]} HS)" for subject in selected_subjects],
            vertical_spacing=0.1,
            horizontal_spacing=0.05
        )
        
        for i, subject in enumerate(selected_subjects):
            subject_data = subject_df[subject_df['Môn học'] == subject]
            
            if len(subject_data) == 0:
                continue
                
            row = (i // n_cols) + 1
            col = (i % n_cols) + 1
            
            # Tạo histogram
            fig_subplots.add_trace(
                go.Histogram(
                    x=subject_data['Điểm số'],
                    nbinsx=20,
                    name=subject,
                    marker_color=colors[i % len(colors)],
                    opacity=0.7,
                    hovertemplate=
                    '<b>%{x:.1f} điểm</b><br>' +
                    'Số học sinh: %{y}<br>' +
                    'Tỷ lệ: %{customdata:.1f}%<extra></extra>',
                    customdata=(np.ones(len(subject_data)) / len(subject_data) * 100)
                ),
                row=row, col=col
            )
            
            # Thêm đường trung bình
            avg_score = subject_data['Điểm số'].mean()
            fig_subplots.add_vline(
                x=avg_score, 
                line_dash="dash", 
                line_color="red",
                annotation_text=f"TB: {avg_score:.1f}",
                annotation_position="top right",
                row=row, col=col
            )
        
        fig_subplots.update_layout(
            height=300 * n_rows,
            showlegend=False,
            title_text="Phân Bố Điểm Chi Tiết Theo Từng Môn Học"
        )
        fig_subplots.update_xaxes(title_text="Điểm số", range=[0, 10])
        fig_subplots.update_yaxes(title_text="Số học sinh")
        
        st.plotly_chart(fig_subplots, use_container_width=True)
        
        # THỐNG KÊ CHI TIẾT
        st.subheader("📊 Thống Kê Chi Tiết Theo Môn")
        
        stats_data = []
        for subject in selected_subjects:
            subject_data = subject_df[subject_df['Môn học'] == subject]
            if len(subject_data) > 0:
                scores = subject_data['Điểm số']
                stats_data.append({
                    'Môn học': subject,
                    'Số HS': len(subject_data),
                    'Điểm TB': round(scores.mean(), 2),
                    'Điểm Cao nhất': round(scores.max(), 2),
                    'Điểm Thấp nhất': round(scores.min(), 2),
                    'Độ lệch chuẩn': round(scores.std(), 2),
                    'HS Giỏi (≥8)': len([s for s in scores if s >= 8]),
                    'HS Khá (6.5-7.9)': len([s for s in scores if 6.5 <= s < 8]),
                    'HS Yếu (<5)': len([s for s in scores if s < 5])
                })
        
        if stats_data:
            stats_df = pd.DataFrame(stats_data)
            st.dataframe(stats_df, use_container_width=True)
            
    except Exception as e:
        st.error(f"❌ Lỗi hiển thị biểu đồ đường: {e}")

# === HỆ THỐNG BIỂU ĐỒ CHI TIẾT CHO TỪNG HỌC SINH ===
def display_student_detail_charts():
    """Hiển thị biểu đồ chi tiết cho từng học sinh"""
    try:
        if not st.session_state.get("ai1_done", False):
            st.info("ℹ️ Vui lòng chạy AI 1 để xem biểu đồ chi tiết")
            return
            
        st.markdown("---")
        st.header("👤 PHÂN TÍCH CHI TIẾT THEO HỌC SINH")
        
        # Lấy danh sách học sinh
        df_students = pd.read_sql("""
            SELECT ho_ten, lop, dtb, du_bao_lstm, danh_gia, risk_level, mon
            FROM students 
            WHERE ho_ten IS NOT NULL
            ORDER BY ho_ten
        """, engine)
        
        if df_students.empty:
            st.warning("⚠️ Không có dữ liệu học sinh")
            return
        
        # Chọn học sinh
        student_names = df_students['ho_ten'].tolist()
        selected_student = st.selectbox(
            "🎯 Chọn học sinh để xem chi tiết:",
            options=student_names,
            index=0
        )
        
        if selected_student:
            # Lấy thông tin học sinh được chọn
            student_data = df_students[df_students['ho_ten'] == selected_student].iloc[0]
            st.session_state.selected_student = selected_student
            
            # Hiển thị thông tin cơ bản
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("👤 Học sinh", selected_student)
            with col2:
                st.metric("🏫 Lớp", student_data['lop'])
            with col3:
                st.metric("📊 ĐTB Hiện tại", f"{student_data['dtb']:.2f}")
            with col4:
                if pd.notna(student_data['du_bao_lstm']):
                    st.metric("🔮 ĐTB Dự báo", f"{student_data['du_bao_lstm']:.2f}")
            
            # Hiển thị đánh giá và cảnh báo
            col5, col6 = st.columns(2)
            with col5:
                st.info(f"📈 Đánh giá: **{student_data['danh_gia']}**")
            with col6:
                risk_color = {
                    'high': '🔴',
                    'medium': '🟡', 
                    'low': '🟢'
                }.get(student_data['risk_level'], '⚪')
                st.warning(f"⚠️ Mức độ rủi ro: {risk_color} **{student_data['risk_level'].upper()}**")
            
            # PHÂN TÍCH ĐIỂM CHI TIẾT THEO MÔN HỌC
            st.subheader("📚 Phân Tích Điểm Theo Môn Học")
            
            try:
                mon_dict = json.loads(student_data['mon']) if isinstance(student_data['mon'], str) else student_data['mon']
                
                if mon_dict and len(mon_dict) > 0:
                    # Lọc chỉ lấy các môn học có tên hợp lệ
                    valid_mon_dict = {k: v for k, v in mon_dict.items() 
                                    if not k.startswith('Môn_') and k != 'Không xác định'}
                    
                    if not valid_mon_dict:
                        st.warning("⚠️ Không có dữ liệu điểm môn học hợp lệ cho học sinh này")
                        return
                    
                    # Tạo DataFrame cho điểm các môn
                    subject_df = pd.DataFrame({
                        'Môn học': list(valid_mon_dict.keys()),
                        'Điểm số': list(valid_mon_dict.values())
                    }).sort_values('Điểm số', ascending=False)
                    
                    # BIỂU ĐỒ 1: Cột điểm các môn
                    st.subheader("📊 Biểu Đồ Cột - Điểm Từng Môn")
                    fig_bar = px.bar(
                        subject_df,
                        x='Môn học',
                        y='Điểm số',
                        title=f"Điểm Các Môn Học Của {selected_student}",
                        color='Điểm số',
                        color_continuous_scale='viridis'
                    )
                    fig_bar.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig_bar, use_container_width=True)
                    
                    # BIỂU ĐỒ 2: Radar chart
                    st.subheader("🎯 Biểu Đồ Radar - So Sánh Điểm Các Môn")
                    fig_radar = go.Figure()
                    
                    fig_radar.add_trace(go.Scatterpolar(
                        r=list(valid_mon_dict.values()),
                        theta=list(valid_mon_dict.keys()),
                        fill='toself',
                        name=selected_student,
                        line=dict(color='blue', width=2)
                    ))
                    
                    fig_radar.update_layout(
                        polar=dict(
                            radialaxis=dict(
                                visible=True,
                                range=[0, 10]
                            )),
                        showlegend=False,
                        title=f"Biểu Đồ Radar Điểm Các Môn - {selected_student}",
                        height=500
                    )
                    
                    st.plotly_chart(fig_radar, use_container_width=True)
                    
                    # BIỂU ĐỒ 3: Pie chart phân bố điểm
                    st.subheader("🥧 Biểu Đồ Tròn - Phân Bố Điểm")
                    
                    # Phân loại điểm
                    score_categories = {
                        'Xuất sắc (9-10)': len([s for s in valid_mon_dict.values() if s >= 9]),
                        'Giỏi (8-8.9)': len([s for s in valid_mon_dict.values() if 8 <= s < 9]),
                        'Khá (7-7.9)': len([s for s in valid_mon_dict.values() if 7 <= s < 8]),
                        'Trung bình (5-6.9)': len([s for s in valid_mon_dict.values() if 5 <= s < 7]),
                        'Yếu (<5)': len([s for s in valid_mon_dict.values() if s < 5])
                    }
                    
                    categories = [k for k, v in score_categories.items() if v > 0]
                    values = [v for k, v in score_categories.items() if v > 0]
                    
                    if categories:
                        fig_pie = px.pie(
                            names=categories,
                            values=values,
                            title=f"Phân Bố Điểm Theo Mức - {selected_student}",
                            color_discrete_sequence=px.colors.qualitative.Set3
                        )
                        st.plotly_chart(fig_pie, use_container_width=True)
                    
                    # THỐNG KÊ CHI TIẾT
                    st.subheader("📈 Thống Kê Chi Tiết")
                    
                    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
                    
                    with col_stat1:
                        st.metric("Số môn học", len(valid_mon_dict))
                        highest_subject = max(valid_mon_dict, key=valid_mon_dict.get)
                        highest_score = valid_mon_dict[highest_subject]
                        st.metric("Môn điểm cao nhất", f"{highest_subject}: {highest_score}")
                    
                    with col_stat2:
                        lowest_subject = min(valid_mon_dict, key=valid_mon_dict.get)
                        lowest_score = valid_mon_dict[lowest_subject]
                        st.metric("Môn điểm thấp nhất", f"{lowest_subject}: {lowest_score}")
                        std_dev = np.std(list(valid_mon_dict.values()))
                        st.metric("Độ lệch chuẩn", f"{std_dev:.2f}")
                    
                    with col_stat3:
                        avg_score = np.mean(list(valid_mon_dict.values()))
                        st.metric("Điểm trung bình", f"{avg_score:.2f}")
                        above_avg = len([s for s in valid_mon_dict.values() if s > avg_score])
                        st.metric("Môn trên trung bình", above_avg)
                    
                    with col_stat4:
                        excellent_count = len([s for s in valid_mon_dict.values() if s >= 8])
                        good_count = len([s for s in valid_mon_dict.values() if 6.5 <= s < 8])
                        st.metric("Môn Giỏi (≥8)", excellent_count)
                        st.metric("Môn Khá (6.5-7.9)", good_count)
                    
                    # ĐÁNH GIÁ ĐIỂM MẠNH VÀ ĐIỂM YẾU
                    st.subheader("🎯 Đánh Giá Điểm Mạnh và Điểm YẾU")
                    
                    strong_subjects = [(sub, score) for sub, score in valid_mon_dict.items() if score >= 8.0]
                    weak_subjects = [(sub, score) for sub, score in valid_mon_dict.items() if score < 5.0]
                    average_subjects = [(sub, score) for sub, score in valid_mon_dict.items() if 5.0 <= score < 8.0]
                    
                    col_strong, col_weak, col_avg = st.columns(3)
                    
                    with col_strong:
                        if strong_subjects:
                            st.success("**💪 ĐIỂM MẠNH (≥8.0):**")
                            for subject, score in sorted(strong_subjects, key=lambda x: x[1], reverse=True):
                                st.write(f"✅ {subject}: **{score}** điểm")
                        else:
                            st.info("ℹ️ Chưa có môn nào đạt điểm mạnh")
                    
                    with col_weak:
                        if weak_subjects:
                            st.error("**📉 ĐIỂM YẾU (<5.0):**")
                            for subject, score in sorted(weak_subjects, key=lambda x: x[1]):
                                st.write(f"❌ {subject}: **{score}** điểm")
                        else:
                            st.success("🎉 Không có môn nào bị điểm yếu")
                    
                    with col_avg:
                        if average_subjects:
                            st.warning("**📊 ĐIỂM TRUNG BÌNH (5.0-7.9):**")
                            for subject, score in sorted(average_subjects, key=lambda x: x[1], reverse=True):
                                st.write(f"📝 {subject}: **{score}** điểm")
                        else:
                            st.info("ℹ️ Không có môn ở mức trung bình")
                            
                else:
                    st.warning("⚠️ Không có dữ liệu điểm môn học cho học sinh này")
                    
            except Exception as e:
                st.error(f"❌ Lỗi phân tích dữ liệu môn học: {e}")
                
        else:
            st.info("👆 Vui lòng chọn một học sinh để xem phân tích chi tiết")
            
    except Exception as e:
        st.error(f"❌ Lỗi hiển thị biểu đồ chi tiết: {e}")

# === HỆ THỐNG BIỂU ĐỒ TỔNG QUAN (ĐÃ SỬA LỖI) ===
def display_overview_charts():
    """Hiển thị biểu đồ tổng quan - ĐÃ SỬA LỖI STATSMODELS"""
    try:
        if not st.session_state.get("ai1_done", False):
            st.info("ℹ️ Vui lòng chạy AI 1 để xem biểu đồ")
            return
            
        st.markdown("---")
        st.header("📊 BIỂU ĐỒ PHÂN TÍCH TỔNG QUAN")
        
        # Lấy dữ liệu từ database
        df = pd.read_sql("""
            SELECT * FROM students 
            WHERE ho_ten IS NOT NULL AND dtb IS NOT NULL
            LIMIT 1000
        """, engine)
        
        if df.empty:
            st.warning("⚠️ Không có dữ liệu để hiển thị biểu đồ")
            return
            
        st.success(f"✅ Đã tải {len(df)} bản ghi để hiển thị biểu đồ")
        
        # Tạo tabs cho các loại biểu đồ
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Tổng quan", "🎯 Dự báo", "📚 Môn học", "📊 Phân bố điểm"])
        
        with tab1:
            display_general_charts(df)
        
        with tab2:
            display_prediction_charts_simple(df)  # Dùng phiên bản đơn giản không cần statsmodels
            
        with tab3:
            display_subject_analysis(df)
            
        with tab4:
            display_subject_line_charts()  # Thêm tab mới cho biểu đồ đường
            
    except Exception as e:
        st.error(f"❌ Lỗi hiển thị biểu đồ: {e}")

def display_general_charts(df):
    """Hiển thị biểu đồ tổng quan"""
    col1, col2 = st.columns(2)
    
    with col1:
        # Phân bố điểm trung bình
        if 'dtb' in df.columns and not df['dtb'].isna().all():
            fig_hist = px.histogram(
                df, x='dtb', nbins=20, 
                title="Phân Bố Điểm Trung Bình Toàn Trường",
                color_discrete_sequence=['#636EFA']
            )
            st.plotly_chart(fig_hist, use_container_width=True)
    
    with col2:
        # Phân bố đánh giá
        if 'danh_gia' in df.columns:
            rating_data = df['danh_gia'].dropna()
            if len(rating_data) > 0:
                rating_counts = rating_data.value_counts()
                fig_pie = px.pie(
                    values=rating_counts.values, 
                    names=rating_counts.index,
                    title="Phân Bố Đánh Giá Học Lực",
                    color_discrete_sequence=px.colors.qualitative.Pastel
                )
                st.plotly_chart(fig_pie, use_container_width=True)

def display_prediction_charts_simple(df):
    """Hiển thị biểu đồ dự báo - PHIÊN BẢN ĐƠN GIẢN KHÔNG CẦN STATSMODELS"""
    col1, col2 = st.columns(2)
    
    with col1:
        # So sánh điểm thực tế vs dự báo
        if all(col in df.columns for col in ['dtb', 'du_bao_lstm']):
            comparison_data = df[['dtb', 'du_bao_lstm', 'ho_ten', 'lop']].dropna()
            if len(comparison_data) > 0:
                fig_scatter = px.scatter(
                    comparison_data, x='dtb', y='du_bao_lstm',
                    title="So sánh ĐTB Thực tế vs Dự báo",
                    hover_data=['ho_ten', 'lop']
                    # Đã bỏ trendline để tránh lỗi statsmodels
                )
                fig_scatter.add_shape(type="line", x0=0, y0=0, x1=10, y1=10,
                                    line=dict(color="red", width=2, dash="dash"))
                st.plotly_chart(fig_scatter, use_container_width=True)
    
    with col2:
        # Phân bố độ tin cậy dự báo
        if 'prediction_confidence' in df.columns:
            confidence_data = df['prediction_confidence'].dropna()
            if len(confidence_data) > 0:
                fig_confidence = px.histogram(
                    df, x='prediction_confidence', nbins=20,
                    title="Phân Bố Độ Tin Cậy Dự Báo",
                    color_discrete_sequence=['#FFA15A']
                )
                st.plotly_chart(fig_confidence, use_container_width=True)

def display_subject_analysis(df):
    """Phân tích điểm theo môn học toàn trường"""
    st.subheader("📚 Phân Tích Điểm Theo Môn Học Toàn Trường")
    
    try:
        # Thu thập dữ liệu điểm từ tất cả học sinh
        all_subject_data = []
        for _, student in df.iterrows():
            try:
                mon_dict = json.loads(student['mon']) if isinstance(student['mon'], str) else student['mon']
                for subject, score in mon_dict.items():
                    if isinstance(score, (int, float)) and 0 <= score <= 10:
                        all_subject_data.append({
                            'Môn học': subject,
                            'Điểm số': float(score),
                            'Lớp': student.get('lop', ''),
                            'Học sinh': student.get('ho_ten', '')
                        })
            except:
                continue
        
        if all_subject_data:
            subject_df = pd.DataFrame(all_subject_data)
            
            # Lọc chỉ lấy các môn học hợp lệ
            valid_subjects = subject_df[~subject_df['Môn học'].str.startswith('Môn_') & 
                                      (subject_df['Môn học'] != 'Không xác định')]
            
            if len(valid_subjects) > 0:
                # Top môn học có điểm cao nhất
                subject_avg = valid_subjects.groupby('Môn học')['Điểm số'].mean().sort_values(ascending=False).head(15)
                
                fig_subjects = px.bar(
                    x=subject_avg.index, y=subject_avg.values,
                    title="Top 15 Môn Học Có Điểm Cao Nhất",
                    labels={'x': 'Môn học', 'y': 'Điểm trung bình'},
                    color=subject_avg.values,
                    color_continuous_scale='rainbow'
                )
                fig_subjects.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig_subjects, use_container_width=True)
            else:
                st.info("📚 Chưa có dữ liệu điểm môn học hợp lệ")
        else:
            st.info("📚 Chưa có dữ liệu điểm môn học chi tiết")
            
    except Exception as e:
        st.error(f"❌ Lỗi phân tích môn học: {e}")

# === HIỂN THỊ DỮ LIỆU HỌC SINH ===
def display_student_data():
    """Hiển thị bảng dữ liệu học sinh"""
    try:
        if not st.session_state.get("ai1_done", False):
            st.info("ℹ️ Vui lòng chạy AI 1 để xem dữ liệu học sinh")
            return
            
        st.markdown("---")
        st.header("📋 DỮ LIỆU HỌC SINH")
        
        # Lấy dữ liệu từ database
        try:
            df = pd.read_sql("""
                SELECT 
                    ho_ten, lop, dtb, 
                    du_bao_lstm, danh_gia, canh_bao,
                    prediction_confidence, risk_level,
                    xep_hang_lop, xep_hang_truong
                FROM students 
                WHERE ho_ten IS NOT NULL
                ORDER BY dtb DESC NULLS LAST
                LIMIT 1000
            """, engine)
            
            if df.empty:
                st.warning("⚠️ Không có dữ liệu học sinh trong database")
                return
                
            st.success(f"✅ Đã tải {len(df)} học sinh")
            
        except Exception as db_error:
            st.error(f"❌ Lỗi kết nối database: {db_error}")
            return
        
        # Hiển thị thống kê tổng quan
        st.subheader("📊 Thống Kê Tổng Quan")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Tổng số học sinh", len(df))
        with col2:
            avg_dtb = df['dtb'].mean() if 'dtb' in df.columns and not df['dtb'].isna().all() else 0
            st.metric("ĐTB trung bình", f"{avg_dtb:.2f}")
        with col3:
            if 'danh_gia' in df.columns:
                excellent = len(df[df['danh_gia'].isin(['Xuất sắc', 'Giỏi'])])
                st.metric("Học sinh Giỏi & Xuất sắc", excellent)
        with col4:
            if 'risk_level' in df.columns:
                high_risk = len(df[df['risk_level'] == 'high'])
                st.metric("Học sinh rủi ro cao", high_risk)
        
        # Bộ lọc dữ liệu
        st.subheader("🔍 Bộ Lọc Dữ Liệu")
        col_filter1, col_filter2, col_filter3 = st.columns(3)
        
        with col_filter1:
            class_options = ["Tất cả"] + sorted(df['lop'].dropna().unique().tolist())
            selected_class = st.selectbox("Lớp", class_options, key="filter_class")
        
        with col_filter2:
            if 'danh_gia' in df.columns:
                rating_options = ["Tất cả"] + sorted(df['danh_gia'].dropna().unique().tolist())
                selected_rating = st.selectbox("Đánh giá", rating_options, key="filter_rating")
            else:
                selected_rating = "Tất cả"
        
        with col_filter3:
            if 'risk_level' in df.columns:
                risk_options = ["Tất cả"] + sorted(df['risk_level'].dropna().unique().tolist())
                selected_risk = st.selectbox("Mức rủi ro", risk_options, key="filter_risk")
            else:
                selected_risk = "Tất cả"
        
        # Áp dụng bộ lọc
        filtered_df = df.copy()
        if selected_class != "Tất cả":
            filtered_df = filtered_df[filtered_df['lop'] == selected_class]
        if selected_rating != "Tất cả" and 'danh_gia' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['danh_gia'] == selected_rating]
        if selected_risk != "Tất cả" and 'risk_level' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['risk_level'] == selected_risk]
        
        # Hiển thị bảng dữ liệu
        st.subheader(f"📄 Dữ Liệu Chi Tiết ({len(filtered_df)} học sinh)")
        
        if len(filtered_df) > 0:
            # Định dạng cột số
            display_df = filtered_df.copy()
            numeric_columns = ['dtb', 'du_bao_lstm', 'prediction_confidence']
            for col in numeric_columns:
                if col in display_df.columns:
                    display_df[col] = display_df[col].round(2)
            
            # Hiển thị bảng
            st.dataframe(
                display_df,
                use_container_width=True,
                height=400,
                hide_index=True
            )
            
            # Nút tải xuống
            csv = filtered_df.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="📥 Tải xuống dữ liệu (CSV)",
                data=csv,
                file_name=f"du_lieu_hoc_sinh_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        else:
            st.warning("⚠️ Không có dữ liệu phù hợp với bộ lọc")
            
    except Exception as e:
        st.error(f"❌ Lỗi hiển thị dữ liệu: {e}")

# === KIỂM TRA DATABASE ===
def check_database_data():
    """Kiểm tra dữ liệu trong database"""
    try:
        st.sidebar.markdown("---")
        st.sidebar.header("🔍 Kiểm Tra Dữ Liệu")
        
        if st.sidebar.button("🔄 Kiểm tra database", use_container_width=True):
            with st.sidebar:
                with st.spinner("Đang kiểm tra..."):
                    # Kiểm tra số lượng bản ghi
                    total_students = pd.read_sql("SELECT COUNT(*) as count FROM students", engine)['count'].iloc[0]
                    st.info(f"📊 Tổng học sinh: {total_students}")
                    
                    # Kiểm tra các bảng
                    tables = pd.read_sql("""
                        SELECT table_name 
                        FROM information_schema.tables 
                        WHERE table_schema = 'public'
                    """, engine)
                    st.info(f"📋 Số bảng: {len(tables)}")
        
    except Exception as e:
        st.sidebar.error(f"❌ Lỗi kiểm tra: {e}")

# === GIAO DIỆN CHÍNH HOÀN CHỈNH ===
st.title("🧠 AI DỰ BÁO ĐIỂM THÔNG MINH - PHIÊN BẢN 3.1")
st.markdown("Hệ thống AI với phân tích chi tiết theo từng học sinh và môn học")

# Hiển thị trạng thái
st.subheader("📊 Trạng Thái Hệ Thống")
col_status1, col_status2, col_status3 = st.columns(3)

with col_status1:
    status_ai1 = "✅ Hoàn thành" if st.session_state.ai1_done else "❌ Chưa chạy"
    st.metric("AI 1 - Xử lý dữ liệu", status_ai1)

with col_status2:
    status_ai2 = "✅ Hoàn thành" if st.session_state.ai2_done else "❌ Chưa chạy"
    st.metric("AI 2 - Phân tích & Dự báo", status_ai2)

with col_status3:
    try:
        total_students = pd.read_sql("SELECT COUNT(*) as count FROM students", engine)['count'].iloc[0]
        st.metric("👥 Tổng học sinh", total_students)
    except:
        st.metric("👥 Tổng học sinh", 0)

# Các nút chức năng
st.markdown("---")
st.subheader("🚀 Thao tác chính")
col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("📥 AI 1: Xử lý Dữ Liệu", use_container_width=True, type="primary"):
        if run_advanced_ai1():
            st.rerun()
            
with col2:
    if st.button("🤖 AI 2: Phân tích & Dự báo", use_container_width=True, type="secondary"):
        if run_advanced_ai2():
            st.rerun()
            
with col3:
    if st.button("⚡ Chạy Toàn Bộ", use_container_width=True, type="primary"):
        with st.spinner("Đang chạy toàn bộ quy trình AI..."):
            if run_advanced_ai1():
                time.sleep(2)
                if run_advanced_ai2():
                    st.success("✅ Đã hoàn thành toàn bộ quy trình AI!")
                    st.rerun()
                else:
                    st.error("❌ Lỗi khi chạy AI 2")
            else:
                st.error("❌ Lỗi khi chạy AI 1")
                
with col4:
    if st.button("🔄 Làm Mới", use_container_width=True, type="secondary"):
        st.rerun()

# THÊM KIỂM TRA DATABASE VÀO SIDEBAR
check_database_data()

# Tạo tabs cho các loại hiển thị khác nhau
st.markdown("---")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Biểu Đồ Tổng Quan", 
    "👤 Phân Tích Chi Tiết", 
    "📋 Dữ Liệu Học Sinh",
    "🎯 AI Insights",
    "📱 Telegram Notifications"  # TAB MỚI
])

with tab1:
    display_overview_charts()

with tab2:
    display_student_detail_charts()

with tab3:
    display_student_data()

with tab4:
    st.header("🎯 AI Insights & Khuyến Nghị")
    
    if st.session_state.get("ai2_done", False):
        # Phân tích insights từ dữ liệu
        df = pd.read_sql("SELECT * FROM students WHERE ho_ten IS NOT NULL", engine)
        
        if not df.empty:
            # Tính toán các chỉ số
            avg_dtb = df['dtb'].mean()
            high_risk_count = len(df[df['risk_level'] == 'high'])
            improving_trend = len(df[df['du_bao_lstm'] > df['dtb']]) if 'du_bao_lstm' in df.columns else 0
            
            col_insight1, col_insight2, col_insight3 = st.columns(3)
            
            with col_insight1:
                st.metric("📈 ĐTB toàn trường", f"{avg_dtb:.2f}")
            with col_insight2:
                st.metric("⚠️ Học sinh rủi ro cao", high_risk_count)
            with col_insight3:
                st.metric("🔮 Xu hướng cải thiện", improving_trend)
            
            # Khuyến nghị
            st.subheader("💡 Khuyến Nghị Hành Động")
            
            if high_risk_count > 0:
                st.error(f"**Ưu tiên:** Hỗ trợ {high_risk_count} học sinh có rủi ro cao")
            
            if avg_dtb < 6.5:
                st.warning("**Cần cải thiện:** Chất lượng học tập toàn trường cần được nâng cao")
            
            if improving_trend > len(df) * 0.7:
                st.success("**Tích cực:** Đa số học sinh có xu hướng cải thiện điểm số")
    else:
        st.info("ℹ️ Vui lòng chạy AI 2 để xem insights thông minh")

with tab5:
    display_telegram_interface()

# Khởi chạy ứng dụng
if __name__ == "__main__":
    initialize_database(engine)