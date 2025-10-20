import pandas as pd
import pickle
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import os
import yagmail # Thư viện gửi email
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression

# --- CẤU HÌNH EMAIL ---
SENDER_EMAIL = "your_email@gmail.com" # THAY THẾ BẰNG EMAIL CỦA BẠN
APP_PASSWORD = "YOUR_APP_PASSWORD"   # THAY THẾ BẰNG MẬT KHẨU ỨNG DỤNG CỦA BẠN (KHÔNG PHẢI MẬT KHẨU GMAIL)
BGH_EMAIL = "ban_giam_hieu@truong.edu.vn" # Email của Ban Giám hiệu

# --- HUẤN LUYỆN VÀ LƯU MÔ HÌNH ---
def train_and_save_models():
    """Huấn luyện và lưu trữ tất cả các mô hình AI."""
    print("Bắt đầu huấn luyện các mô hình...")
    
    try:
        df_grades = pd.read_csv('student_grades.csv')
        df_comments = pd.read_csv('teacher_comments.csv')
    except FileNotFoundError as e:
        print(f"Lỗi: Không tìm thấy tệp dữ liệu cần thiết. Vui lòng kiểm tra lại. Lỗi: {e}")
        return False

    # 1. Huấn luyện mô hình Phân loại (Classification)
    features_clf = [col for col in df_grades.columns if col.endswith('_score')]
    if 'final_category' in df_grades.columns and len(features_clf) > 0:
        X_clf, y_clf = df_grades[features_clf], df_grades['final_category']
        model_clf = RandomForestClassifier(n_estimators=100, random_state=42)
        model_clf.fit(X_clf, y_clf)
        with open('model_classification.pkl', 'wb') as f: pickle.dump(model_clf, f)
        print("Đã huấn luyện và lưu 'model_classification.pkl'.")
    else:
        print("Không đủ dữ liệu để huấn luyện mô hình Phân loại. Bỏ qua.")
        
    # 2. Huấn luyện mô hình Dự đoán (Regression) - Dự đoán điểm Hóa từ Toán và Lý
    features_reg = ['math_score', 'physics_score']
    if 'chemistry_score' in df_grades.columns and all(f in df_grades.columns for f in features_reg):
        X_reg, y_reg = df_grades[features_reg], df_grades['chemistry_score']
        model_reg = LinearRegression()
        model_reg.fit(X_reg, y_reg)
        with open('model_regression.pkl', 'wb') as f: pickle.dump(model_reg, f)
        print("Đã huấn luyện và lưu 'model_regression.pkl'.")
    else:
        print("Không đủ dữ liệu để huấn luyện mô hình Dự đoán Hóa. Bỏ qua.")

    # 3. Huấn luyện mô hình NLP
    if 'comment' in df_comments.columns and 'label' in df_comments.columns:
        X_nlp, y_nlp = df_comments['comment'], df_comments['label']
        vectorizer_nlp = TfidfVectorizer()
        X_nlp_vectorized = vectorizer_nlp.fit_transform(X_nlp)
        model_nlp = LogisticRegression(max_iter=1000) # Tăng max_iter nếu dữ liệu lớn
        model_nlp.fit(X_nlp_vectorized, y_nlp)
        with open('model_nlp.pkl', 'wb') as f: pickle.dump(model_nlp, f)
        with open('vectorizer_nlp.pkl', 'wb') as f: pickle.dump(vectorizer_nlp, f)
        print("Đã huấn luyện và lưu 'model_nlp.pkl' và 'vectorizer_nlp.pkl'.")
    else:
        print("Không đủ dữ liệu để huấn luyện mô hình NLP. Bỏ qua.")
    
    print("Tất cả các mô hình đã được huấn luyện và lưu thành công (nếu đủ dữ liệu).")
    return True

def load_models():
    """Tải các mô hình đã lưu."""
    models = {}
    try:
        models['clf'] = pd.read_pickle('model_classification.pkl')
        print("Đã tải mô hình phân loại.")
    except FileNotFoundError:
        print("Không tìm thấy model_classification.pkl. Một số chức năng sẽ bị hạn chế.")
    try:
        models['reg'] = pd.read_pickle('model_regression.pkl')
        print("Đã tải mô hình dự đoán.")
    except FileNotFoundError:
        print("Không tìm thấy model_regression.pkl. Một số chức năng sẽ bị hạn chế.")
    try:
        models['nlp'] = pd.read_pickle('model_nlp.pkl')
        models['vectorizer_nlp'] = pd.read_pickle('vectorizer_nlp.pkl')
        print("Đã tải mô hình NLP và vectorizer.")
    except FileNotFoundError:
        print("Không tìm thấy model_nlp.pkl hoặc vectorizer_nlp.pkl. Một số chức năng sẽ bị hạn chế.")
    return models

# --- VẼ BIỂU ĐỒ ---
def create_chart_directory(charts_dir='report_charts'):
    """Tạo thư mục lưu biểu đồ nếu chưa tồn tại."""
    if not os.path.exists(charts_dir):
        os.makedirs(charts_dir)
    return charts_dir

def plot_student_progress(df_student, student_name, subject, output_path):
    if not df_student.empty and f'{subject}_score' in df_student.columns:
        plt.figure(figsize=(10, 6))
        sns.lineplot(x='report_date', y=f'{subject}_score', data=df_student, marker='o')
        plt.title(f'Tiến độ môn {subject} của {student_name}', fontsize=16)
        plt.xlabel('Thời điểm báo cáo', fontsize=12)
        plt.ylabel(f'Điểm số môn {subject}', fontsize=12)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
    else:
        print(f"Không đủ dữ liệu để vẽ biểu đồ tiến độ môn {subject} cho {student_name}.")

def plot_class_performance(df_class, subject, output_path):
    if not df_class.empty and f'{subject}_score' in df_class.columns:
        df_avg = df_class.groupby('report_date')[f'{subject}_score'].mean().reset_index()
        plt.figure(figsize=(12, 7))
        sns.lineplot(x='report_date', y=f'{subject}_score', data=df_avg, marker='o', label='Điểm trung bình của lớp')
        plt.title(f'Điểm trung bình môn {subject} của cả lớp', fontsize=16)
        plt.xlabel('Thời điểm báo cáo', fontsize=12)
        plt.ylabel('Điểm trung bình', fontsize=12)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
    else:
        print(f"Không đủ dữ liệu để vẽ biểu đồ điểm trung bình môn {subject} của lớp.")

def plot_individual_overall_performance(df_student, student_name, all_subjects, output_path):
    subjects_with_data = [s for s in all_subjects if f'{s}_score' in df_student.columns]
    if not df_student.empty and subjects_with_data:
        df_melt = df_student.melt(id_vars=['report_date'], value_vars=[f'{s}_score' for s in subjects_with_data], var_name='subject', value_name='score')
        # Loại bỏ '_score' khỏi tên môn học
        df_melt['subject'] = df_melt['subject'].str.replace('_score', '')
        plt.figure(figsize=(12, 7))
        sns.lineplot(x='report_date', y='score', hue='subject', data=df_melt, marker='o')
        plt.title(f'Tổng hợp tiến độ các môn của {student_name}', fontsize=16)
        plt.xlabel('Thời điểm báo cáo', fontsize=12)
        plt.ylabel('Điểm số', fontsize=12)
        plt.grid(True)
        plt.legend(title='Môn học', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
    else:
        print(f"Không đủ dữ liệu để vẽ biểu đồ tổng hợp cho {student_name}.")

def plot_class_overall_performance(df_class, all_subjects, output_path):
    subjects_with_data = [s for s in all_subjects if f'{s}_score' in df_class.columns]
    if not df_class.empty and subjects_with_data:
        df_melt = df_class.melt(id_vars=['report_date'], value_vars=[f'{s}_score' for s in subjects_with_data], var_name='subject', value_name='score')
        df_avg = df_melt.groupby(['report_date', 'subject'])['score'].mean().reset_index()
        # Loại bỏ '_score' khỏi tên môn học
        df_avg['subject'] = df_avg['subject'].str.replace('_score', '')
        plt.figure(figsize=(12, 7))
        sns.lineplot(x='report_date', y='score', hue='subject', data=df_avg, marker='o')
        plt.title('Điểm trung bình tổng hợp các môn của cả lớp', fontsize=16)
        plt.xlabel('Thời điểm báo cáo', fontsize=12)
        plt.ylabel('Điểm trung bình', fontsize=12)
        plt.grid(True)
        plt.legend(title='Môn học', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
    else:
        print("Không đủ dữ liệu để vẽ biểu đồ tổng hợp của cả lớp.")

# --- GỬI EMAIL ---
def send_email(to_email, subject, body, attachments=None):
    if not SENDER_EMAIL or not APP_PASSWORD:
        print("Lỗi: Thông tin email người gửi hoặc mật khẩu ứng dụng chưa được cấu hình. Không thể gửi email.")
        return

    try:
        yag = yagmail.SMTP(SENDER_EMAIL, APP_PASSWORD)
        yag.send(to=to_email, subject=subject, contents=body, attachments=attachments)
        print(f"Đã gửi email thành công đến: {to_email}")
    except Exception as e:
        print(f"Lỗi khi gửi email đến {to_email}: {e}")

# --- QUY TRÌNH TỰ ĐỘNG HÓA CHÍNH ---
def run_full_automation():
    print(f"[{datetime.now()}] Bắt đầu quy trình tự động hóa...")
    
    # 1. Tải dữ liệu
    try:
        df_grades = pd.read_csv('student_grades.csv')
        df_comments = pd.read_csv('teacher_comments.csv')
        # Chuyển đổi cột ngày tháng
        df_grades['report_date'] = pd.to_datetime(df_grades['report_date'])
    except FileNotFoundError as e:
        print(f"Lỗi: Không tìm thấy tệp dữ liệu cần thiết. Vui lòng kiểm tra lại. Lỗi: {e}")
        return
    except Exception as e:
        print(f"Lỗi khi đọc hoặc xử lý dữ liệu: {e}")
        return

    # 2. Tải các mô hình AI đã huấn luyện
    models = load_models()
    model_clf = models.get('clf')
    model_reg = models.get('reg')
    model_nlp = models.get('nlp')
    vectorizer_nlp = models.get('vectorizer_nlp')

    df_final = df_grades.copy() # Sử dụng bản sao để thêm cột phân tích

    # 3. Phân tích dữ liệu bằng AI
    # Phân loại kết quả cuối kỳ
    features_for_clf = [col for col in df_grades.columns if col.endswith('_score')]
    if model_clf and len(features_for_clf) > 0:
        try:
            df_final['predicted_category'] = model_clf.predict(df_grades[features_for_clf])
        except Exception as e:
            print(f"Lỗi khi dự đoán phân loại: {e}")
            df_final['predicted_category'] = "Không dự đoán được"
    else:
        df_final['predicted_category'] = "Không có mô hình phân loại"
        
    # Dự đoán điểm Hóa
    features_for_reg = ['math_score', 'physics_score']
    if model_reg and all(f in df_grades.columns for f in features_for_reg):
        try:
            df_final['predicted_chem'] = model_reg.predict(df_grades[features_for_reg])
        except Exception as e:
            print(f"Lỗi khi dự đoán điểm Hóa: {e}")
            df_final['predicted_chem'] = -1.0
    else:
        df_final['predicted_chem'] = "Không có mô hình dự đoán Hóa"

    # Phân tích cảm xúc nhận xét
    if model_nlp and vectorizer_nlp and 'comment' in df_comments.columns:
        try:
            df_comments_vectorized = vectorizer_nlp.transform(df_comments['comment'])
            df_comments['sentiment'] = model_nlp.predict(df_comments_vectorized)
            df_final = pd.merge(df_final, df_comments[['student_id', 'comment', 'sentiment']], on='student_id', how='left')
        except Exception as e:
            print(f"Lỗi khi phân tích nhận xét NLP: {e}")
            df_final['comment'] = "Không có nhận xét"
            df_final['sentiment'] = "Không phân tích được"
    else:
        df_final['comment'] = "Không có mô hình NLP"
        df_final['sentiment'] = "Không phân tích được"

    # 4. Tạo thư mục lưu biểu đồ
    charts_dir = create_chart_directory()
        
    # 5. Vẽ biểu đồ
    all_subjects = [col.replace('_score', '') for col in df_grades.columns if '_score' in col]
    latest_report_date = df_grades['report_date'].max()

    print("Bắt đầu vẽ biểu đồ...")
    for student_id in df_grades['student_id'].unique():
        df_student = df_grades[df_grades['student_id'] == student_id].sort_values(by='report_date')
        student_name = df_student['student_name'].iloc[0]
        
        # Biểu đồ theo từng môn của từng học sinh
        for subject in all_subjects:
            plot_student_progress(df_student, student_name, subject, os.path.join(charts_dir, f'progress_{student_name}_{subject}.png'))
        
        # Biểu đồ tổng hợp của từng học sinh
        plot_individual_overall_performance(df_student, student_name, all_subjects, os.path.join(charts_dir, f'overall_progress_{student_name}.png'))
        
    # Biểu đồ tổng quan của cả lớp
    for subject in all_subjects:
        plot_class_performance(df_grades, subject, os.path.join(charts_dir, f'class_performance_class_{subject}.png'))
    plot_class_overall_performance(df_grades, all_subjects, os.path.join(charts_dir, 'class_overall_performance.png'))
    
    print(f"[{datetime.now()}] Đã tạo tất cả các biểu đồ thành công.")

    # 6. Gửi báo cáo và cảnh báo
    df_latest_data = df_final[df_final['report_date'] == latest_report_date].copy()
    
    # Gửi báo cáo tổng hợp cho Ban Giám hiệu
    bgh_report_body = f"Kính gửi Ban Giám hiệu,\n\nĐây là báo cáo tổng hợp tình hình học tập của lớp tính đến ngày {latest_report_date.strftime('%d/%m/%Y')}. Vui lòng xem các biểu đồ đính kèm để biết chi tiết.\n\nTrân trọng."
    bgh_attachments = [os.path.join(charts_dir, f'class_performance_class_{s}.png') for s in all_subjects]
    bgh_attachments.append(os.path.join(charts_dir, 'class_overall_performance.png'))
    send_email(BGH_EMAIL, f"Báo cáo Tổng hợp Học tập Lớp - {latest_report_date.strftime('%d/%m/%Y')}", bgh_report_body, attachments=bgh_attachments)
    
    # Gửi báo cáo chi tiết cho phụ huynh và cảnh báo (nếu cần)
    for index, row in df_latest_data.iterrows():
        parent_email = row['parent_email']
        student_name = row['student_name']
        
        # Chuẩn bị danh sách các biểu đồ cá nhân để đính kèm
        individual_attachments = [
            os.path.join(charts_dir, f'progress_{student_name}_{s}.png') for s in all_subjects
        ]
        individual_attachments.append(os.path.join(charts_dir, f'overall_progress_{student_name}.png'))

        report_subject = f"Báo cáo học tập của {student_name} - {latest_report_date.strftime('%d/%m/%Y')}"
        report_body = f"""
Kính gửi phụ huynh học sinh {student_name},

Chúng tôi xin gửi báo cáo chi tiết về tình hình học tập của em đến ngày {latest_report_date.strftime('%d/%m/%Y')}.

* **Điểm trung bình hiện tại:** {row['avg_score']:.2f}
* **Dự đoán kết quả cuối kỳ:** {row['predicted_category']}
* **Dự đoán điểm Hóa (ví dụ):** {row['predicted_chem']:.2f}
* **Nhận xét từ giáo viên:** "{row['comment']}"
* **Phân tích cảm xúc nhận xét:** Nhận xét này có xu hướng "{row['sentiment']}"

Vui lòng xem các biểu đồ đính kèm để có cái nhìn trực quan về tiến độ học tập của em qua từng môn và tổng hợp.

Trân trọng,
Nhà trường.
"""
        # Kiểm tra điều kiện cảnh báo
        is_at_risk = False
        if row['avg_score'] < 5.5: # Tiêu chí điểm trung bình dưới 5.5
            is_at_risk = True
        if row['sentiment'] == 'tiêu cực': # Tiêu chí nhận xét tiêu cực
            is_at_risk = True

        if is_at_risk:
            alert_subject = f"[CẢNH BÁO] Tình hình học tập của {student_name}"
            alert_body = f"""
Kính gửi phụ huynh học sinh {student_name},

Đây là một cảnh báo quan trọng về tình hình học tập của em trong thời gian gần đây. Chúng tôi nhận thấy:
* Điểm trung bình hiện tại: {row['avg_score']:.2f} (dưới mức trung bình)
* Nhận xét của giáo viên có xu hướng tiêu cực: "{row['comment']}"

Nhà trường rất mong phụ huynh cùng quan tâm, đồng hành để giúp đỡ em cải thiện kết quả học tập. Vui lòng xem các biểu đồ đính kèm để biết thêm chi tiết.

Trân trọng,
Nhà trường.
"""
            send_email(parent_email, alert_subject, alert_body, attachments=individual_attachments)
        else:
            send_email(parent_email, report_subject, report_body, attachments=individual_attachments)
            
    print(f"[{datetime.now()}] Đã hoàn tất quy trình báo cáo và cảnh báo.")

if __name__ == "__main__":
    # Bước 1: Huấn luyện các mô hình (chỉ cần chạy một lần hoặc khi dữ liệu huấn luyện thay đổi)
    if train_and_save_models(): # Chỉ chạy các bước tiếp theo nếu huấn luyện thành công
        # Bước 2: Chạy quy trình tự động hóa đầy đủ
        run_full_automation()