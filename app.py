import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import os
import yaml

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

    # Upload file or generate demo data for 2000 students
    uploaded_file = st.file_uploader("Upload CSV file", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
    else:
        # Generate demo data for 2000 students
        np.random.seed(42)
        student_ids = [f"S{i:04d}" for i in range(1, 2001)]
        subjects = ['Math', 'Physics', 'Chemistry', 'English']
        previous_subjects = [f'Previous_{s}' for s in subjects]
        data = {
            'StudentID': student_ids,
        }
        for subject in subjects:
            data[subject] = np.random.normal(7.0, 1.0, 2000).clip(0, 10)  # Điểm ngẫu nhiên trung bình 7, độ lệch 1
            data[f'Previous_{subject}'] = data[subject] - np.random.normal(0.5, 0.5, 2000).clip(-2, 2)  # Tiến bộ ngẫu nhiên
        df = pd.DataFrame(data)

    # Sidebar filter
    st.sidebar.subheader("Lọc Dữ Liệu")
    grade_level = st.sidebar.selectbox("Chọn Lớp", options=["Tất cả"] + [f"Lớp {i}" for i in range(10, 13)])
    score_range = st.sidebar.slider("Phạm Vi Điểm", 0.0, 10.0, (0.0, 10.0), 0.5)

    # Filter data
    if grade_level != "Tất cả":
        df = df[df['StudentID'].str[1:3].astype(int) % 10 == int(grade_level[-1])]  # Giả lập lớp dựa trên ID
    df = df[(df[subjects] >= score_range[0]).all(axis=1) & (df[subjects] <= score_range[1]).all(axis=1)]

    # Tổng Quan Toàn Trường
    st.subheader("Tổng Quan Toàn Trường")
    if not df.empty:
        # Thống kê trung bình và phân bố
        avg_scores = df[subjects].mean()
        st.write("**Trung Bình Điểm Hiện Tại Theo Môn Học:**")
        st.write(avg_scores)

        # Biểu đồ hộp (box plot) để xem phân bố điểm
        melted_df = pd.melt(df, id_vars=['StudentID'], value_vars=subjects, var_name="Subject", value_name="Score")
        fig_overview = px.box(melted_df, x="Subject", y="Score", title="Phân Bố Điểm Số Toàn Trường",
                             labels={"Score": "Điểm", "Subject": "Môn Học"},
                             color_discrete_sequence=px.colors.qualitative.Plotly)
        st.plotly_chart(fig_overview, use_container_width=True)
    else:
        st.write("Không có dữ liệu để hiển thị.")

    # Comparison section
    st.subheader("So Sánh Tiến Bộ Với Lần Trước")
    col_progress = st.columns(2)
    with col_progress[0]:
        st.write("**Điểm Hiện Tại**")
        current_subjects = [col for col in df.columns if not col.startswith("Previous_") and col != "StudentID"]
        if not df.empty:
            st.write(df[["StudentID"] + current_subjects].head(10))  # Hiển thị 10 học sinh đầu để tránh quá tải
        else:
            st.write("Không có dữ liệu để hiển thị.")
    with col_progress[1]:
        st.write("**Điểm Lần Trước**")
        previous_subjects = [col for col in df.columns if col.startswith("Previous_")]
        if not df.empty and "StudentID" in df.columns and all(col in df.columns for col in previous_subjects):
            st.write(df[["StudentID"] + previous_subjects].head(10))  # Hiển thị 10 học sinh đầu
        else:
            st.write("Không có dữ liệu để hiển thị.")

    # Progress calculation
    st.write("**Tiến Bộ**")
    progress_data = {}
    for subject in [col.replace("Previous_", "") for col in df.columns if col.startswith("Previous_")]:
        current_col = subject
        previous_col = f"Previous_{subject}"
        if current_col in df.columns and previous_col in df.columns:
            progress_data[f"{subject}_Progress"] = df[current_col] - df[previous_col]
    progress_df = pd.DataFrame({
        "StudentID": df["StudentID"],
        **progress_data
    })
    if not progress_df.empty:
        st.write(progress_df.head(10))  # Hiển thị 10 học sinh đầu
    else:
        st.write("Không có dữ liệu tiến bộ để hiển thị.")

    # Detailed Progress Chart
    st.subheader("Biểu Đồ Tiến Bộ Chi Tiết")
    progress_melt = pd.melt(progress_df.head(100), id_vars=["StudentID"],  # Giới hạn 100 học sinh để tối ưu
                            value_vars=[col for col in progress_df.columns if col.endswith("_Progress")],
                            var_name="Subject", value_name="Progress")
    progress_melt["Subject"] = progress_melt["Subject"].str.replace("_Progress", "")
    if not progress_melt.empty:
        fig_progress = px.bar(progress_melt, x="StudentID", y="Progress", color="Subject",
                            title="Tiến Bộ Học Tập Theo Học Sinh",
                            labels={"Progress": "Chênh Lệch Điểm", "StudentID": "Mã Học Sinh"},
                            color_discrete_sequence=px.colors.qualitative.Plotly)
        fig_progress.update_traces(text=progress_melt["Progress"].round(2), textposition="auto")
        fig_progress.update_layout(
            xaxis_title="Mã Học Sinh",
            yaxis_title="Chênh Lệch Điểm",
            showlegend=True,
            height=400
        )
        st.plotly_chart(fig_progress, use_container_width=True)
    else:
        st.write("Không có dữ liệu để vẽ biểu đồ.")

    # Progress Notifications
    st.subheader("Thông Báo Tiến Bộ")
    for index, row in progress_df.head(10).iterrows():  # Giới hạn 10 học sinh
        student_id = row["StudentID"]
        for subject in [col.replace("_Progress", "") for col in progress_df.columns if col.endswith("_Progress")]:
            progress_col = f"{subject}_Progress"
            progress_value = row[progress_col]
            if progress_value > 0.5:
                st.success(f"Học sinh {student_id}: Tiến bộ {subject} +{progress_value:.2f} điểm!")
            elif -0.5 <= progress_value <= 0.5:
                st.warning(f"Học sinh {student_id}: Tiến bộ {subject} ổn định ({progress_value:.2f} điểm)")
            else:
                st.error(f"Học sinh {student_id}: Giảm {subject} {abs(progress_value):.2f} điểm!")

    # Thống Kê Tổng Quan
    st.subheader("Thống Kê Tổng Quan")
    if not df.empty:
        avg_scores = df[current_subjects].mean()
        st.write("**Trung Bình Điểm Hiện Tại Theo Môn Học:**")
        st.write(avg_scores)

        fig_avg = px.bar(x=avg_scores.index, y=avg_scores.values,
                        title="Trung Bình Điểm Hiện Tại Theo Môn Học",
                        labels={"x": "Môn Học", "y": "Điểm Trung Bình"},
                        color_discrete_sequence=px.colors.qualitative.Plotly)
        st.plotly_chart(fig_avg, use_container_width=True)
    else:
        st.write("Không có dữ liệu để thống kê.")

    # Biểu Đồ Dự Báo Tiến Bộ
    st.subheader("Biểu Đồ Dự Báo Tiến Bộ")
    if not progress_df.empty:
        forecast_periods = 3
        forecast_data = {}
        for subject in [col.replace("_Progress", "") for col in progress_df.columns if col.endswith("_Progress")]:
            progress_col = f"{subject}_Progress"
            current_progress = progress_df[progress_col].mean()
            forecast = [df[subject].iloc[0] + (current_progress * i) for i in range(1, forecast_periods + 1)]
            forecast_data[f"{subject}_Forecast"] = forecast

        forecast_df = pd.DataFrame(forecast_data, index=[f"Kỳ {i+2}" for i in range(forecast_periods)])
        st.write("**Dự Báo Điểm Số Các Kỳ Tới (Dựa Trên Tiến Bộ Hiện Tại):**")
        st.write(forecast_df)

        melted_forecast = pd.melt(forecast_df.reset_index(), id_vars=["index"], var_name="Subject", value_name="Predicted Score")
        melted_forecast["index"] = melted_forecast["index"].astype(str)
        fig_forecast = px.line(melted_forecast, x="index", y="Predicted Score", color="Subject",
                              title="Dự Báo Tiến Bộ Điểm Số",
                              labels={"index": "Kỳ Học", "Predicted Score": "Điểm Dự Báo"},
                              color_discrete_sequence=px.colors.qualitative.Plotly)
        st.plotly_chart(fig_forecast, use_container_width=True)
    else:
        st.write("Không có dữ liệu để dự báo.")

    # Run AI button (placeholder)
    if st.button("Chạy Phân Tích AI"):
        st.write("Đang xử lý... (Thêm logic AI của bạn ở đây)")