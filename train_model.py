# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

# --- BƯỚC 1: ĐỌC DỮ LIỆU VÀ CHIA TẬP HUẤN LUYỆN ---
print("Bắt đầu huấn luyện mô hình AI...")
try:
    # Đọc dữ liệu từ tệp CSV mà bạn đã tạo ở bước trước
   # Cung cấp encoding 'utf-8' để đọc các ký tự tiếng Việt
    df = pd.read_csv('training_data.csv', encoding='latin1')
except FileNotFoundError:
    print("Lỗi: Không tìm thấy tệp 'training_data.csv'. Vui lòng kiểm tra lại.")
    exit()

# Chia dữ liệu thành 2 phần: 80% để huấn luyện và 20% để kiểm thử
X_train, X_test, y_train, y_test = train_test_split(
    df['comment'],    # Đây là các câu nhận xét
    df['label'],      # Đây là các nhãn tương ứng ('tích cực'/'tiêu cực')
    test_size=0.2,
    random_state=42
)
print("Đã chia dữ liệu thành công.")

# --- BƯỚC 2: VECTOR HÓA DỮ LIỆU VĂN BẢN ---
# Chuyển đổi văn bản thành dữ liệu số mà mô hình có thể hiểu
print("Đang vector hóa dữ liệu văn bản...")
vectorizer = TfidfVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)
print("Đã vector hóa dữ liệu thành công.")

# --- BƯỚC 3: HUẤN LUYỆN MÔ HÌNH ---
# Khởi tạo và huấn luyện mô hình Logistic Regression
print("Đang huấn luyện mô hình AI...")
model = LogisticRegression()
model.fit(X_train_vectorized, y_train)
print("Huấn luyện mô hình hoàn tất.")

# --- BƯỚC 4: ĐÁNH GIÁ VÀ LƯU MÔ HÌNH ---
# Đánh giá độ chính xác của mô hình
y_pred = model.predict(X_test_vectorized)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nĐộ chính xác của mô hình trên tập kiểm thử: {accuracy * 100:.2f}%")

# Lưu mô hình và bộ vectorizer vào tệp để sử dụng sau này
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("Đã lưu mô hình và bộ vectorizer thành công dưới dạng 'model.pkl' và 'vectorizer.pkl'.")