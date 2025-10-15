import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import json
import urllib.request
import base64  # Để mã hóa thông tin liên lạc
from sklearn.metrics import mean_squared_error, mean_absolute_error  # Để đánh giá
import warnings
warnings.filterwarnings('ignore')

# Tạo dữ liệu mẫu (giữ nguyên, thêm feature engineering)
def create_sample_data():
    data = {
        'StudentID': [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6],
        'Class': ['A1', 'A1', 'A1', 'A1', 'A2', 'A2', 'A2', 'A2', 'A1', 'A1', 'A2', 'A2'],
        'Subject': ['Toán', 'Lý', 'Toán', 'Lý', 'Toán', 'Lý', 'Toán', 'Lý', 'Hóa', 'Văn', 'Hóa', 'Văn'],
        'Score1': [7.5, 6.0, 8.0, 9.0, 4.5, 5.0, 9.5, 8.5, 7.0, 6.5, 5.5, 8.0],
        'Score2': [8.0, 6.5, 7.5, 8.5, 4.0, 4.5, 9.0, 8.0, 7.5, 7.0, 6.0, 8.5],
        'Score3': [7.0, 5.5, 8.5, 9.5, 3.5, 4.0, 9.5, 8.5, 8.0, 6.0, 5.0, 9.0],
        'FinalScore': [7.5, 6.0, 8.0, 9.0, 4.0, 4.5, 9.3, 8.3, 7.5, 6.5, 5.5, 8.5],
        'StudentEmail': ['hs1@example.com', 'hs1@example.com', 'hs2@example.com', 'hs2@example.com', 'hs3@example.com', 'hs3@example.com', 'hs4@example.com', 'hs4@example.com', 'hs5@example.com', 'hs5@example.com', 'hs6@example.com', 'hs6@example.com'],
        'ParentEmail': ['ph1@example.com', 'ph1@example.com', 'ph2@example.com', 'ph2@example.com', 'ph3@example.com', 'ph3@example.com', 'ph4@example.com', 'ph4@example.com', 'ph5@example.com', 'ph5@example.com', 'ph6@example.com', 'ph6@example.com'],
        'StudentZaloID': ['zalo_hs1', 'zalo_hs1', 'zalo_hs2', 'zalo_hs2', 'zalo_hs3', 'zalo_hs3', 'zalo_hs4', 'zalo_hs4', 'zalo_hs5', 'zalo_hs5', 'zalo_hs6', 'zalo_hs6'],
        'ParentZaloID': ['zalo_ph1', 'zalo_ph1', 'zalo_ph2', 'zalo_ph2', 'zalo_ph3', 'zalo_ph3', 'zalo_ph4', 'zalo_ph4', 'zalo_ph5', 'zalo_ph5', 'zalo_ph6', 'zalo_ph6'],
        'StudentMessengerID': ['mess_hs1', 'mess_hs1', 'mess_hs2', 'mess_hs2', 'mess_hs3', 'mess_hs3', 'mess_hs4', 'mess_hs4', 'mess_hs5', 'mess_hs5', 'mess_hs6', 'mess_hs6'],
        'ParentMessengerID': ['mess_ph1', 'mess_ph1', 'mess_ph2', 'mess_ph2', 'mess_ph3', 'mess_ph3', 'mess_ph4', 'mess_ph4', 'mess_ph5', 'mess_ph5', 'mess_ph6', 'mess_ph6']
    }
    return pd.DataFrame(data)

# Load dữ liệu từ nguồn (thêm mã hóa cho thông tin liên lạc)
def load_data(source):
    if source.endswith('.csv'):
        try:
            df = pd.read_csv(source)
        except FileNotFoundError:
            print("File CSV không tồn tại, sử dụng dữ liệu mẫu.")
            df = create_sample_data()
    elif source.startswith('http'):
        try:
            with urllib.request.urlopen(source) as response:
                data = json.loads(response.read().decode())
            df = pd.DataFrame(data)
            print(f"Dữ liệu đã tải từ API: {source}")
        except Exception as e:
            print(f"Lỗi khi tải từ API: {e}. Sử dụng dữ liệu mẫu.")
            df = create_sample_data()
    else:
        print("Nguồn không hợp lệ. Sử dụng dữ liệu mẫu.")
        df = create_sample_data()
    
    # Feature engineering: Thêm cột trung bình giữa kỳ
    df['AvgMidterm'] = df[['Score1', 'Score2', 'Score3']].mean(axis=1)
    
    # Bổ sung cột nếu thiếu
    required_cols = ['StudentID', 'Class', 'Subject', 'Score1', 'Score2', 'Score3', 'FinalScore', 'AvgMidterm',
                     'StudentEmail', 'ParentEmail', 'StudentZaloID', 'ParentZaloID', 'StudentMessengerID', 'ParentMessengerID']
    for col in required_cols:
        if col not in df.columns:
            if col == 'FinalScore':
                df[col] = df[['Score1', 'Score2', 'Score3']].mean(axis=1)
            elif col == 'AvgMidterm':
                df[col] = df[['Score1', 'Score2', 'Score3']].mean(axis=1)
            else:
                print(f"Cảnh báo: Cột {col} thiếu trong dữ liệu. Bỏ qua gửi cho kênh này.")
                df[col] = None
    
    # Mã hóa thông tin liên lạc để bảo mật (base64)
    for col in ['StudentEmail', 'ParentEmail', 'StudentZaloID', 'ParentZaloID', 'StudentMessengerID', 'ParentMessengerID']:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: base64.b64encode(str(x).encode('utf-8')).decode('utf-8') if x else None)
    
    return df

# Mô hình AI dự báo mạnh hơn: MLP với hidden layers và dropout
class GradePredictor(nn.Module):
    def __init__(self, input_size=4):  # Thêm AvgMidterm → 4 inputs
        super(GradePredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)  # Hidden layer 1
        self.fc2 = nn.Linear(64, 32)         # Hidden layer 2
        self.fc3 = nn.Linear(32, 1)          # Output layer
        self.dropout = nn.Dropout(0.2)       # Dropout để tránh overfitting
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        return torch.sigmoid(self.fc3(x)) * 10  # Giới hạn 0-10

def train_model(df, epochs=20):  # Tăng epochs để huấn luyện tốt hơn
    # Chuẩn bị dữ liệu với feature mới
    X = df[['Score1', 'Score2', 'Score3', 'AvgMidterm']].values
    y = df['FinalScore'].values.reshape(-1, 1)
    
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)  # Tăng batch size
    
    model = GradePredictor()
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001)  # Optimizer tốt hơn
    
    # Huấn luyện
    model.train()
    for epoch in range(epochs):
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
    
    # Đánh giá mô hình
    model.eval()
    with torch.no_grad():
        preds = model(X_tensor).numpy()
    mse = mean_squared_error(y, preds)
    mae = mean_absolute_error(y, preds)
    print(f"Đánh giá mô hình: MSE = {mse:.2f}, MAE = {mae:.2f}")
    
    # Hỗ trợ GPU nếu có
    if torch.cuda.is_available():
        model = model.cuda()
    
    return model

def predict_final_score(model, scores):
    # scores giờ là [Score1, Score2, Score3, AvgMidterm]
    input_tensor = torch.tensor([scores], dtype=torch.float32)
    if torch.cuda.is_available():
        input_tensor = input_tensor.cuda()
    with torch.no_grad():
        pred = model(input_tensor).item()
    return round(pred, 2)

# Tạo alerts (cập nhật với feature mới)
def generate_alerts(df, model):
    alerts = []
    for _, row in df.iterrows():
        scores = [row['Score1'], row['Score2'], row['Score3'], row['AvgMidterm']]
        predicted = predict_final_score(model, scores)
        
        msg = f"Học sinh {row['StudentID']} - Lớp {row['Class']} - Môn {row['Subject']}: Dự báo {predicted}/10"
        if predicted < 5.0:
            msg += " - CẢNH BÁO: Điểm yếu, cần cải thiện!"
            level = "CẢNH BÁO"
        else:
            msg += " - Tốt, tiếp tục duy trì!"
            level = "THÔNG BÁO"
        
        alert = {
            'student_id': row['StudentID'],
            'class': row['Class'],
            'subject': row['Subject'],
            'msg': msg,
            'level': level,
            'student_email': row['StudentEmail'],
            'parent_email': row['ParentEmail'],
            'student_zalo_id': row['StudentZaloID'],
            'parent_zalo_id': row['ParentZaloID'],
            'student_mess_id': row['StudentMessengerID'],
            'parent_mess_id': row['ParentMessengerID']
        }
        alerts.append(alert)
    
    return alerts

# Tạo report (thêm metrics mô hình nếu cần)
def generate_report(df, alerts):
    report = {
        'Tổng học sinh': len(df['StudentID'].unique()),
        'Tổng lớp': len(df['Class'].unique()),
        'Trung bình toàn trường': df['FinalScore'].mean(),
        'Số lượng cảnh báo': sum(1 for alert in alerts if alert['level'] == 'CẢNH BÁO'),
        'Chi tiết cảnh báo': [alert['msg'] for alert in alerts if alert['level'] == 'CẢNH BÁO']
    }
    
    subjects = df['Subject'].unique()
    for sub in subjects:
        report[f'Trung bình {sub}'] = df[df['Subject'] == sub]['FinalScore'].mean()
    
    # Xuất Excel
    report_df = pd.DataFrame([report])
    report_df.to_excel('bao_cao_ban_giam_hieu.xlsx', index=False)
    
    return report