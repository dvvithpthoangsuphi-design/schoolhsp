import qrcode
from PIL import Image

def create_qr_with_logo(data, logo_path, qr_color, logo_fill_color):
    """
    Tạo mã QR có logo và màu sắc tùy chỉnh.

    Args:
        data (str): Dữ liệu cần mã hóa (ví dụ: link Google Forms).
        logo_path (str): Đường dẫn đến tệp hình ảnh logo.
        qr_color (str): Màu của mã QR (ví dụ: 'blue', 'green', '#007bff').
        logo_fill_color (str): Màu nền của logo trong mã QR.
    """
    # 1. Tạo mã QR
    qr = qrcode.QRCode(
        error_correction=qrcode.constants.ERROR_CORRECT_H # Độ chính xác cao nhất
    )
    qr.add_data(data)
    qr.make(fit=True)

    # 2. Tạo hình ảnh mã QR với màu tùy chỉnh
    img_qr = qr.make_image(
        fill_color=qr_color,
        back_color="white"
    ).convert('RGB')

    # 3. Mở và chỉnh sửa logo
    logo = Image.open(logo_path)
    # Thay đổi kích thước logo để vừa với mã QR
    logo = logo.resize((img_qr.size[0] // 4, img_qr.size[1] // 4), Image.LANCZOS)
    
    # 4. Tạo nền cho logo và dán logo vào
    pos = ((img_qr.size[0] - logo.size[0]) // 2, (img_qr.size[1] - logo.size[1]) // 2)
    img_qr.paste(logo, pos)
    
    # 5. Lưu hình ảnh mã QR
    img_qr.save("my_qr_code.png")
    print("Đã tạo mã QR thành công và lưu vào 'my_qr_code.png'")

# --- CẤU HÌNH SỬ DỤNG ---
if __name__ == "__main__":
    # Dữ liệu bạn muốn mã hóa vào QR
    form_link = "https://forms.gle/oAXpirockzFxJ6H77" 
    
    # Đường dẫn đến tệp logo của trường
    school_logo = "logo.jpg" 
    
    # Màu mã QR và màu nền logo
    qr_color = "darkblue"
    
    create_qr_with_logo(form_link, school_logo, qr_color, "white")