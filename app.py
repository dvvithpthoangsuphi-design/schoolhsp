import streamlit as st
import pandas as pd
import plotly.express as px
import os
import yaml

# Load config (from ENV or file)
config_path = 'config.yaml'
if os.path.exists(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
elif 'AUTH_CONFIG' in os.environ:
    config = yaml.safe_load(os.environ['AUTH_CONFIG'])

# Sidebar for navigation
st.sidebar.title("Menu")
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

# Authentication
if not st.session_state.authenticated:
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")
    if st.sidebar.button("Login"):
        if username in config['credentials']['usernames'] and config['credentials']['usernames'][username]['password'] == password:
            st.session_state.authenticated = True
            st.sidebar.success(f"Welcome, {config['credentials']['usernames'][username]['name']}!")
        else:
            st.sidebar.error("Invalid username or password")
else:
    st.sidebar.button("Logout", on_click=lambda: setattr(st.session_state, 'authenticated', False))

# Main content only if authenticated
if st.session_state.authenticated:
    st.title("AI Dự Báo Điểm Học Sinh")

    # Upload file or use demo data
    uploaded_file = st.file_uploader("Upload CSV file", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
    else:
        # Demo data with multiple subjects
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

    # Center the chart
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.subheader("Biểu Đồ Dự Báo Điểm")
        # Dynamic subjects for line chart
        current_subjects = [col for col in df.columns if not col.startswith("Previous_")]
        fig = px.line(df, x="StudentID", y=current_subjects,
                      title="Dự Báo Điểm Hiện Tại",
                      labels={"value": "Điểm", "variable": "Môn Học"},
                      color_discrete_sequence=px.colors.qualitative.Plotly)
        st.plotly_chart(fig, use_container_width=True)

    # Comparison section below chart
    st.subheader("So Sánh Tiến Bộ Với Lần Trước")
    col_progress = st.columns(2)
    with col_progress[0]:
        st.write("**Điểm Hiện Tại**")
        st.write(df[["StudentID"] + current_subjects])
    with col_progress[1]:
        st.write("**Điểm Lần Trước**")
        previous_subjects = [col for col in df.columns if col.startswith("Previous_")]
        st.write(df[["StudentID"] + previous_subjects])

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
    st.write(progress_df)

    # Detailed Progress Chart for all subjects
    st.subheader("Biểu Đồ Tiến Bộ Chi Tiết")
    progress_melt = pd.melt(progress_df, id_vars=["StudentID"], 
                            value_vars=[col for col in progress_df.columns if col.endswith("_Progress")],
                            var_name="Subject", value_name="Progress")
    # Remove "_Progress" from subject names for display
    progress_melt["Subject"] = progress_melt["Subject"].str.replace("_Progress", "")
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

    # Progress Notifications for all subjects
    st.subheader("Thông Báo Tiến Bộ")
    for index, row in progress_df.iterrows():
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

    # Run AI button (placeholder)
    if st.button("Chạy Phân Tích AI"):
        st.write("Đang xử lý... (Thêm logic AI của bạn ở đây)")