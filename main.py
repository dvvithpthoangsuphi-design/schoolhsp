from controller import run_program

if __name__ == "__main__":
    source = input("Nhập nguồn dữ liệu (CSV path hoặc API URL JSON, ví dụ: 'your_grades.csv' hoặc 'https://example.com/grades.json'): ").strip()
    if not source:
        source = 'your_grades.csv'  # Mặc định
    
    run_program(source)