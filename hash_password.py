import streamlit_authenticator as stauth

# Danh sách password thật (thay bằng password bạn muốn cho từng trường)
passwords_to_hash = ['pass123', 'securepass']  # Ví dụ: school_a và school_b

# Gen hashed đúng cách cho version mới
hashed_passwords = stauth.Hasher(passwords=passwords_to_hash).generate()
print(hashed_passwords)  # Output: List hashed để copy vào config.yaml