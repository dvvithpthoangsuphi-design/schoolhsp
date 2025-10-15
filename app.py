import streamlit as st
import pandas as pd
import yaml
import os
from streamlit_authenticator import Authenticate
from model import load_data, train_model, generate_alerts, generate_report
from view import create_charts

# Cấu hình trang (đầu file)
st.set_page_config(
    page_title="AI Học Tập - Dự Báo Điểm",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load config
config = None
config_path = 'config.yaml'
if os.path.exists(config_path):
    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        st.info("Load config từ file local thành công!")
    except Exception as e:
        st.error(f"Lỗi load YAML: {e}")
        st.stop()
elif 'auth_config' in st.secrets:
    try:
        config = yaml.safe_load(st.secrets['auth_config'])
        st.info("Load config từ secrets thành công!")
    except Exception as e:
        st.error(f"Lỗi load secrets: {e}")
        st.stop()
else:
    st.error("Không tìm thấy config.yaml hoặc secrets!")
    st.stop()

# Khởi tạo authenticator
authenticator = Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days']
)

# Form login
if 'authentication_status' not in st.session_state or not st.session_state['authentication_status']:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("### 🛡️ Đăng Nhập Để Truy Cập")
        authentication_status = authenticator.login(
            location='main',
            fields={'Form name': 'Đăng Nhập', 'Username': 'Tên đăng nhập', 'Password': 'Mật khẩu', 'Login': 'Đăng Nhập'},
            key='auth_login_form_unique'
        )
    
    if st.session_state.get('authentication_status'):
        name = st.session_state.get('name')
        username = st.session_state.get('username')
        st.session_state['name'] = name
        st.session_state['username'] = username
        st.session_state['authentication_status'] = True
        st.success(f"Chào mừng {name}! Đang tải...")
        st.rerun()
    elif st.session_state.get('authentication_status') is False:
        st.error('Username/password sai!')
    elif st.session_state.get('authentication_status') is None:
        st.warning('Vui lòng nhập thông tin.')
    st.stop()
else:
    # Đã login
    name = st.session_state.get('name')
    username = st.session_state.get('username')
    st.sidebar.success(f"Chào {name} (Trường: {username})")
    authenticator.logout(location='sidebar', button_name='Đăng Xuất', key='auth_logout_unique')

    # Hero Section
    st.markdown("""
    # 🧠 **Hệ thống AI Dự báo Điểm Học sinh**
    ### Phân tích điểm học tập, dự báo kết quả, cảnh báo kịp thời & **biểu đồ tương tác**
    *Upload CSV hoặc nhập URL để bắt đầu. Hỗ trợ AI PyTorch cho dự báo chính xác cao.*
    """, unsafe_allow_html=True)

    # Cache với filter theo trường
    @st.cache_data
    def cached_load_and_train_filtered(source, school_id):
        df = load_data(source)
        if 'SchoolID' not in df.columns:
            df['SchoolID'] = school_id
        df = df[df['SchoolID'] == school_id]
        if df.empty:
            raise ValueError("Không có dữ liệu cho trường này!")
        model = train_model(df, epochs=10)
        return df, model

    # Sidebar
    with st.sidebar:
        st.header("⚙️ **Cấu hình**")
        source = st.text_input("📁 Nguồn dữ liệu (CSV path hoặc URL JSON)", value="your_grades.csv")
        upload_file = st.file_uploader("📤 Hoặc upload CSV trực tiếp", type="csv")

        if upload_file is not None:
            df_temp = pd.read_csv(upload_file)
            df_temp['SchoolID'] = username
            df_temp.to_csv("temp_upload.csv", index=False)
            source = "temp_upload.csv"
            st.success("✅ File đã upload thành công!")

        if st.button("🎯 **Quick Demo** (Dữ liệu mẫu)"):
            source = "your_grades.csv"
            st.rerun()

        show_all_charts = st.checkbox("📊 Hiển thị tất cả biểu đồ (có thể chậm với dữ liệu lớn)", value=False)

    # Nút chạy
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 **Chạy Phân tích AI**", type="primary"):
            if source:
                progress_bar = st.progress(0)
                status_text = st.empty()
                with st.spinner("Đang tải..."):
                    try:
                        status_text.text("Đang tải dữ liệu...")
                        progress_bar.progress(20)
                        df, model = cached_load_and_train_filtered(source, username)
                        status_text.text("Đang huấn luyện AI...")
                        progress_bar.progress(50)
                        status_text.text("Đang tạo báo cáo & biểu đồ...")
                        progress_bar.progress(80)
                        alerts = generate_alerts(df, model)
                        report = generate_report(df, alerts)
                        charts = create_charts(df)
                        # Lưu session
                        st.session_state['df'] = df
                        st.session_state['alerts'] = alerts
                        st.session_state['report'] = report
                        st.session_state['charts'] = charts
                        progress_bar.progress(100)
                        status_text.text("Hoàn tất!")
                        st.balloons()
                    except Exception as e:
                        st.error(f"Lỗi: {str(e)}")
            else:
                st.error("Vui lòng nhập nguồn dữ liệu!")

    # Tabs (lấy từ session)
    if 'df' in st.session_state:
        df = st.session_state['df']
        alerts = st.session_state['alerts']
        report = st.session_state['report']
        charts = st.session_state['charts']
        
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Dữ liệu", "⚠️ Dự báo & Cảnh báo", "📈 Báo cáo", "🖼️ Biểu đồ Tương tác"])

        with tab1:
            st.subheader("📋 Dữ liệu Đầu vào")
            st.dataframe(df.head(10), use_container_width=True)
            st.caption(f"*Tổng {len(df)} bản ghi. Cột AvgMidterm là feature AI tự tính.*")

        with tab2:
            st.subheader("🔮 Dự báo Điểm & Cảnh báo")
            warnings_col1, warnings_col2 = st.columns(2)
            with warnings_col1:
                st.metric("Số cảnh báo (Dự báo <5.0)", sum(1 for alert in alerts if "CẢNH BÁO" in alert['msg']))
            with warnings_col2:
                st.metric("Số thông báo tốt", sum(1 for alert in alerts if "TỐT" in alert['msg']))
            
            for alert in alerts:
                level_emoji = "🚨 **CẢNH BÁO**" if "CẢNH BÁO" in alert['msg'] else "✅ **Tốt**"
                with st.expander(f"{level_emoji}: {alert['subject']} (ID: {alert['student_id']})"):
                    st.write(f"*{alert['msg']}*")
                    st.caption(f"Lớp: {alert['class']}")

        with tab3:
            st.subheader("📊 Báo cáo Chi tiết")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("👥 Tổng học sinh", report['Tổng học sinh'])
            with col2:
                st.metric("🏫 Tổng lớp", report['Tổng lớp'])
            with col3:
                st.metric("📈 Trung bình toàn trường", f"{report['Trung bình toàn trường']:.2f}/10")
            
            col4, col5 = st.columns(2)
            with col4:
                st.metric("⚠️ Số cảnh báo", report['Số lượng cảnh báo'])
            with col5:
                st.metric("📥 Chi tiết cảnh báo", len(report['Chi tiết cảnh báo']))
            
            subjects = {k: v for k, v in report.items() if k.startswith('Trung bình ')}
            if subjects:
                st.subheader("📚 Trung bình theo môn")
                subject_df = pd.DataFrame(list(subjects.items()), columns=['Môn học', 'Điểm TB'])
                st.dataframe(subject_df, use_container_width=True)
            
            excel_path = 'bao_cao_ban_giam_hieu.xlsx'
            if os.path.exists(excel_path):
                with open(excel_path, 'rb') as file:
                    st.download_button(
                        label="📥 Tải Báo cáo Excel",
                        data=file.read(),
                        file_name='bao_cao_ban_giam_hieu.xlsx',
                        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                        use_container_width=True
                    )

        with tab4:
            st.subheader("🖼️ Biểu đồ Tương tác (Plotly)")
            st.info("💡 **Tương tác**: Hover để xem chi tiết, zoom/pan để khám phá dữ liệu.")
            
            if not show_all_charts:
                st.warning("⚠️ **Chế độ nhanh**: Chỉ hiển thị 1-2 ví dụ đầu. Tích checkbox sidebar để xem tất cả.")
                sample_keys = ['student_1', 'subject_Toán', 'class_A1', 'school_overview']
                for key in sample_keys:
                    if key in charts:
                        with st.expander(key.replace('_', ' ').title()):
                            st.plotly_chart(charts[key], use_container_width=True)
            else:
                st.subheader("👤 Biểu đồ Học sinh")
                for key, fig in charts.items():
                    if key.startswith('student_'):
                        with st.expander(key.replace('student_', 'Học sinh ').replace('_', ' ')):
                            st.plotly_chart(fig, use_container_width=True)
                
                st.subheader("📚 Biểu đồ Môn học")
                for key, fig in charts.items():
                    if key.startswith('subject_'):
                        with st.expander(key.replace('subject_', 'Môn ').replace('_', ' ')):
                            st.plotly_chart(fig, use_container_width=True)
                
                st.subheader("🏫 Biểu đồ Lớp học")
                for key, fig in charts.items():
                    if key.startswith('class_'):
                        with st.expander(key.replace('class_', 'Lớp ').replace('_', ' ')):
                            st.plotly_chart(fig, use_container_width=True)
                
                st.subheader("🌐 Tổng quan Toàn trường")
                if 'school_overview' in charts:
                    st.plotly_chart(charts['school_overview'], use_container_width=True)

# Footer
st.markdown("---")
col_left, col_right = st.columns([3, 1])
with col_left:
    st.markdown("*👨‍💻 Dự án AI Dự báo Điểm Học sinh | Powered by Streamlit, Plotly & PyTorch*")
with col_right:
    st.markdown("[📧 Liên hệ](mailto:your-email@example.com)")

if __name__ == "__main__":
    pass