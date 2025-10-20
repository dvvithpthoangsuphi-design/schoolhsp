import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import os
import yagmail
import requests
import json

# --- CẤU HÌNH API VÀ EMAIL ---
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY" # Thay bằng API Key của bạn
SENDER_EMAIL = "your_email@gmail.com"
APP_PASSWORD = "YOUR_APP_PASSWORD"
BGH_EMAIL = "ban_giam_hieu@truong.edu.vn"

# --- LẤY DỮ LIỆU ĐỂ VẼ BIỂU ĐỒ ---
def get_student_data():
    try:
        df_grades = pd.read_csv('student_grades.csv')
        df_grades['report_date'] = pd.to_datetime(df_grades['report_date'])
        df_comments = pd.read_csv('teacher_comments.csv')
        return df_grades, df_comments
    except FileNotFoundError:
        print("Lỗi: Không tìm thấy tệp dữ liệu cần thiết.")
        return pd.DataFrame(), pd.DataFrame()

# --- CHỨC NĂNG PHÂN TÍCH VỚI GEMINI API ---
def analyze_comments_with_gemini(comments):
    print("Đang phân tích nhận xét bằng Gemini API...")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={GEMINI_API_KEY}"
    results = []
    for comment in comments:
        prompt = f"Phân tích nhận xét sau và cho biết nó có xu hướng tích cực hay tiêu cực. Nhận xét: '{comment}'"
        data = {"contents": [{"parts": [{"text": prompt}]}]}
        try:
            response = requests.post(url, json=data)
            if response.status_code == 200:
                sentiment = response.json()['candidates'][0]['content']['parts'][0]['text']
                results.append(sentiment.lower().strip().replace('.', ''))
            else:
                results.append("Lỗi phân tích")
        except Exception as e:
            results.append(f"Lỗi: {e}")
    return results

# --- VẼ BIỂU ĐỒ ---
def create_chart_directory(charts_dir='report_charts'):
    if not os.path.exists(charts_dir):
        os.makedirs(charts_dir)
    return charts_dir

def plot_student_progress(df_student, student_name, subject, output_path):
    if not df_student.empty and f'{subject}_score' in df_student.columns:
        plt.figure(figsize=(10, 6))
        sns.lineplot(x='report_date', y=f'{subject}_score', data=df_student, marker='o')
        plt.title(f'Tiến độ môn {subject} của {student_name}')
        plt.savefig(output_path)
        plt.close()

def plot_class_performance(df_class, subject, output_path):
    if not df_class.empty and f'{subject}_score' in df_class.columns:
        df_avg = df_class.groupby('report_date')[f'{subject}_score'].mean().reset_index()
        plt.figure(figsize=(12, 7))
        sns.lineplot(x='report_date', y=f'{subject}_score', data=df_avg, marker='o', label='Điểm trung bình của lớp')
        plt.title(f'Điểm trung bình môn {subject} của cả lớp')
        plt.savefig(output_path)
        plt.close()

# --- GỬI EMAIL ---
def send_email(to_email, subject, body, attachments=None):
    # Cần thay đổi SENDER_EMAIL và APP_PASSWORD
    if SENDER_EMAIL == "your_email@gmail.com":
        print("Lỗi: Cần cấu hình thông tin email gửi. Không thể gửi email.")
        return
    try:
        yag = yagmail.SMTP(SENDER_EMAIL, APP_PASSWORD)
        yag.send(to=to_email, subject=subject, contents=body, attachments=attachments)
        print(f"Đã gửi email thành công đến: {to_email}")
    except Exception as e:
        print(f"Lỗi khi gửi email: {e}")

# --- QUY TRÌNH TỰ ĐỘNG HÓA CHÍNH ---
def run_full_automation():
    print(f"[{datetime.now()}] Bắt đầu quy trình tự động hóa...")
    
    df_grades, df_comments = get_student_data()
    if df_grades.empty:
        print("Quá trình dừng lại.")
        return

    charts_dir = create_chart_directory()
    latest_report_date = df_grades['report_date'].max()

    # Phân tích nhận xét bằng Gemini API
    df_comments['sentiment'] = analyze_comments_with_gemini(df_comments['comment'].tolist())
    df_final = pd.merge(df_grades, df_comments[['student_id', 'comment', 'sentiment']], on='student_id', how='left')

    # Vẽ biểu đồ và gửi báo cáo
    all_subjects = [col.split('_')[0] for col in df_grades.columns if '_score' in col]
    bgh_attachments = []
    
    # Vẽ và đính kèm biểu đồ tổng hợp
    for subject in all_subjects:
        plot_class_performance(df_grades, subject, os.path.join(charts_dir, f'class_performance_{subject}.png'))
        bgh_attachments.append(os.path.join(charts_dir, f'class_performance_{subject}.png'))
    
    # Gửi báo cáo tổng hợp cho BGH
    bgh_report_body = f"Kính gửi Ban Giám hiệu,\n\nĐây là báo cáo tổng hợp tình hình học tập của lớp tính đến ngày {latest_report_date.strftime('%d/%m/%Y')}. Vui lòng xem các biểu đồ đính kèm.\n\nTrân trọng."
    send_email(BGH_EMAIL, f"Báo cáo Tổng hợp Học tập Lớp", bgh_report_body, attachments=bgh_attachments)
    
    # Gửi báo cáo chi tiết cho phụ huynh
    for student_id in df_grades['student_id'].unique():
        df_student = df_grades[df_grades['student_id'] == student_id]
        df_latest_student = df_student[df_student['report_date'] == latest_report_date].copy()
        
        if df_latest_student.empty: continue
        
        student_name = df_latest_student['student_name'].iloc[0]
        parent_email = df_latest_student['parent_email'].iloc[0]
        comment = df_final[(df_final['student_id'] == student_id) & (df_final['report_date'] == latest_report_date)]['comment'].iloc[0]
        sentiment = df_final[(df_final['student_id'] == student_id) & (df_final['report_date'] == latest_report_date)]['sentiment'].iloc[0]
        avg_score = df_latest_student['avg_score'].iloc[0]

        attachments = []
        for subject in all_subjects:
            plot_student_progress(df_student, student_name, subject, os.path.join(charts_dir, f'progress_{student_name}_{subject}.png'))
            attachments.append(os.path.join(charts_dir, f'progress_{student_name}_{subject}.png'))
        
        report_subject = f"Báo cáo học tập của {student_name}"
        report_body = f"""
Kính gửi phụ huynh học sinh {student_name},

Chúng tôi xin gửi báo cáo chi tiết về tình hình học tập của em đến ngày {latest_report_date.strftime('%d/%m/%Y')}.

* Điểm trung bình hiện tại: {avg_score:.2f}
* Nhận xét từ giáo viên: "{comment}"
* Phân tích cảm xúc nhận xét: Nhận xét này có xu hướng "{sentiment}"

Vui lòng xem các biểu đồ đính kèm để thấy rõ hơn về tiến độ học tập của em.
"""
        # Gửi cảnh báo nếu cần
        if avg_score < 5.5 or 'tiêu cực' in sentiment:
            alert_subject = f"[CẢNH BÁO] Tình hình học tập của {student_name}"
            send_email(parent_email, alert_subject, report_body, attachments=attachments)
        else:
            send_email(parent_email, report_subject, report_body, attachments=attachments)

    print(f"[{datetime.now()}] Đã hoàn tất quy trình.")

if __name__ == "__main__":
    run_full_automation()
    