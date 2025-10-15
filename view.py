import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd

# Tạo biểu đồ Plotly tương tác (trả về dict figs)
def create_charts(df):
    charts = {}
    
    # 1. Biểu đồ cho từng học sinh: Bar chart tương tác
    students = df['StudentID'].unique()
    for student in students:
        student_df = df[df['StudentID'] == student]
        subjects = student_df['Subject'].tolist()
        scores = student_df['FinalScore'].tolist()
        
        fig = go.Figure(data=[
            go.Bar(x=subjects, y=scores, marker_color='skyblue', text=scores, textposition='auto')
        ])
        fig.update_layout(
            title=f'Điểm Học tập Học sinh {student}',
            xaxis_title='Môn học',
            yaxis_title='Điểm cuối kỳ',
            yaxis_range=[0, 10],
            height=400,
            showlegend=False
        )
        charts[f'student_{student}'] = fig
    
    # 2. Biểu đồ cho từng môn học: Histogram phân bố điểm
    subjects = df['Subject'].unique()
    for sub in subjects:
        sub_df = df[df['Subject'] == sub]
        scores = sub_df['FinalScore'].tolist()
        
        fig = px.histogram(x=scores, nbins=5, title=f'Phân bố Điểm Môn {sub}',
                           labels={'x': 'Điểm', 'count': 'Số lượng'},
                           color_discrete_sequence=['lightgreen'])
        fig.update_layout(height=400, showlegend=False)
        charts[f'subject_{sub}'] = fig
    
    # 3. Biểu đồ cho từng lớp học: Bar chart trung bình điểm (FIX CHÍNH: colorscale='Reds')
    classes = df['Class'].unique()
    for cls in classes:
        class_df = df[df['Class'] == cls]
        avg_scores = class_df.groupby('Subject')['FinalScore'].mean().reset_index()
        
        fig = px.bar(avg_scores, x='Subject', y='FinalScore', title=f'Trung bình Điểm Lớp {cls}',
                     color='FinalScore', color_continuous_scale='Reds')  # FIX: Thay 'salmon' bằng 'Reds'
        fig.update_layout(height=400, yaxis_range=[0, 10], showlegend=False)
        charts[f'class_{cls}'] = fig
    
    # 4. Biểu đồ tổng quan toàn trường: Pie chart tương tác
    avg_school = df['FinalScore'].mean()
    ratios = {
        'Giỏi (>=8)': len(df[df['FinalScore'] >= 8]),
        'Khá (5-8)': len(df[(df['FinalScore'] >= 5) & (df['FinalScore'] < 8)]),
        'Yếu (<5)': len(df[df['FinalScore'] < 5])
    }
    
    fig = go.Figure(data=[go.Pie(
        labels=list(ratios.keys()),
        values=list(ratios.values()),
        hole=0.3,  # Donut style
        marker_colors=['gold', 'lightblue', 'lightcoral'],
        textinfo='label+percent',
        pull=[0.1, 0, 0]  # Explode slice đầu
    )])
    fig.update_layout(
        title=f'Tổng quan Học tập Toàn trường (TB: {avg_school:.2f})',
        height=400
    )
    charts['school_overview'] = fig
    
    return charts