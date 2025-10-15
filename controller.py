from model import load_data, train_model, generate_alerts, generate_report
from view import display_data, display_model_trained, display_alerts, display_report, create_charts

def run_program(source):
    # Load data từ Model
    df = load_data(source)
    
    # Display data qua View
    display_data(df)
    
    # Train model từ Model (tăng epochs)
    model = train_model(df, epochs=50)  # Có thể điều chỉnh epochs cao hơn cho dữ liệu lớn
    display_model_trained()
    
    # Generate alerts từ Model và gửi qua View
    alerts = generate_alerts(df, model)
    display_alerts(alerts)
    
    # Generate report từ Model
    report = generate_report(df, alerts)
    display_report(report)
    
    # Create charts qua View
    create_charts(df)