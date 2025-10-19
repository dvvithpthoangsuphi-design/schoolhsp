import streamlit as st
import pandas as pd
import plotly.express as px
import os
import yaml

# Load config from file
config_path = 'auth_config.yaml'
config = None

if os.path.exists(config_path):
    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
else:
    st.error("File YAML không tồn tại tại: " + config_path)

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

    # Upload file or use demo data
    uploaded_file = st.file_uploader("Upload CSV file", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.DataFrame({
            "StudentID": ["S001", "S002", "S003"],
            "Math": [7.5, 8.0, 6.5],
            "Physics": [8.0, 7.5, 7.0],
            "Chemistry": [7.2, 6.8, 7.5],
            "English": [8.5, 7.0, 6.0],
            "Previous_Math": [7.0, 7.8, 6.0],
            "Previous_Physics": [7.5, 7.2, 6.5],
            "Previous_Chemistry": [6.8, 6.5, 7.0],
            "Previous_English": [8.0, 6.5, 5.5]
        })

    # Center the chart with school-wide overview
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.subheader("Biểu Đồ Dự Báo Điểm Toàn Trường")
        current_subjects = [col for col in df.columns if not col.startswith("Previous_")]
        melted_df = pd.melt(df, id_vars=["StudentID"], value_vars=current_subjects[1:],  # Loại StudentID khỏi y-axis
                            var_name="Subject", value_name="Score")
        fig = px.line(melted_df, x="StudentID", y="Score", color="Subject",
                      title="Điểm Hiện Tại Toàn Trường",
                      labels={"Score": "Điểm", "StudentID": "Mã Học Sinh"},
                      color_discrete_sequence=px.colors.qualitative.Plotly)
        st.plotly_chart(fig, use_container_width=True)

        # Select student for detailed view
        selected_student = st.selectbox("Chọn học sinh để xem chi tiết", df["StudentID"])
        if selected_student:
            student_data = df[df["StudentID"] == selected_student]
            fig_student = px.line(student_data, x="StudentID", y=current_subjects[1:],  # Loại StudentID khỏi y-axis
                                 title=f"Điểm Hiện Tại của {selected_student}",
                                 labels={"value": "Điểm", "variable": "Môn Học"},
                                 color_discrete_sequence=px.colors.qualitative.Plotly)
            st.plotly_chart(fig_student, use_container_width=True)

    # Comparison section below chart
    st.subheader("So Sánh Tiến Bộ Với Lần Trước")
    col_progress = st.columns(2)
    with col_progress[0]:
        st.write("**Điểm Hiện Tại**")
        if not df.empty and all(col in df.columns for col in current_subjects):
            st.write(df[current_subjects])
        else:
            st.write("Không có dữ liệu để hiển thị.")
    with col_progress[1]:
        st.write("**Điểm Lần Trước**")
        previous_subjects = [col for col in df.columns if col.startswith("Previous_")]
        if not df.empty and "StudentID" in df.columns and all(col in df.columns for col in previous_subjects):
            st.write(df[["StudentID"] + previous_subjects])
        else:
            st.write("Không có dữ liệu để hiển thị.")

    # Progress calculation for all subjects
    st.write("**Tiến Bộ**")
    progress_data = {}
    for subject in current_subjects:
        if f"Previous_{subject}" in df.columns:
            progress_data[f"{subject}_Progress"] = df[subject] - df[f"Previous_{subject}"]
    progress_df = pd.DataFrame({
        "StudentID": df["StudentID"],
        **progress_data
    })
    if not progress_df.empty:
        st.write(progress_df)
    else:
        st.write("Không có dữ liệu tiến bộ để hiển thị.")

    # Detailed Progress Chart for all subjects
    st.subheader("Biểu Đồ Tiến Bộ Chi Tiết")
    progress_melt = pd.melt(progress_df, id_vars=["StudentID"], 
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

    # Simulate sending progress notifications to parents (hidden from screen)
    def send_progress_notification(student_id, subject, progress_value):
        # Giả lập gửi email/phân hệ thống cho phụ huynh
        message = f"Thân gửi phụ huynh học sinh {student_id},\nHọc sinh có tiến bộ {subject}: "
        if progress_value > 0.5:
            message += f"+{progress_value:.2f} điểm!"
        elif -0.5 <= progress_value <= 0.5:
            message += f"ổn định ({progress_value:.2f} điểm)."
        else:
            message += f"giảm {abs(progress_value):.2f} điểm."
        message += "\nTrân trọng,\nTrường học"
        # Trong thực tế, dùng thư viện như smtplib để gửi email
        print(f"Gửi thông báo cho phụ huynh {student_id}: {message}")  # Giả lập

    # Progress calculation and notification (send to parents instead of display)
    for index, row in progress_df.iterrows():
        student_id = row["StudentID"]
        for subject in [col.replace("_Progress", "") for col in progress_df.columns if col.endswith("_Progress")]:
            progress_col = f"{subject}_Progress"
            progress_value = row[progress_col]
            send_progress_notification(student_id, subject, progress_value)

    # Run AI button (placeholder)
    if st.button("Chạy Phân Tích AI"):
        st.write("Đang xử lý... (Thêm logic AI của bạn ở đây)")