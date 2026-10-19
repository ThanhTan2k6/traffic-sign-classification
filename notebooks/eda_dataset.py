import os
import glob
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# CẤU HÌNH ĐƯỜNG DẪN TỚI DATASET CỦA BẠN
# ==========================================
DATASET_DIR = r"d:\HocKy5\thigiacmaytinh\prj\dataset_cropped_for_lenet"

def analyze_folder_based_dataset(data_dir):
    """Phân tích Dataset sắp xếp theo dạng Thư mục chứa từng lớp"""
    print("="*50)
    print("PHÂN TÍCH DATASET DẠNG THƯ MỤC (CLASS FOLDERS)")
    print("="*50)
    
    classes = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    
    if not classes:
        print("[-] Không tìm thấy các thư mục con phân lớp!")
        return None

    stats = []
    total_images = 0
    img_sizes = []

    for cls in sorted(classes):
        cls_path = os.path.join(data_dir, cls)
        # Quét các định dạng ảnh phổ biến
        images = []
        for ext in ('*.jpg', '*.png', '*.jpeg', '*.BMP'):
            images.extend(glob.glob(os.path.join(cls_path, ext)))
            images.extend(glob.glob(os.path.join(cls_path, ext.upper())))
        
        count = len(images)
        total_images += count
        
        # Đọc thử kích thước của 10 ảnh đầu tiên trong lớp để kiểm tra resolution
        for img_p in images[:10]:
            try:
                with Image.open(img_p) as img:
                    img_sizes.append(img.size) # (width, height)
            except Exception:
                pass

        stats.append({"Lớp / Class": cls, "Số lượng ảnh": count})

    df = pd.DataFrame(stats)
    df["Tỷ lệ (%)"] = (df["Số lượng ảnh"] / total_images * 100).round(2)
    
    print(f"\n[+] Tổng số lượng ảnh (N): {total_images} ảnh")
    print(f"[+] Tổng số lớp (C): {len(classes)} lớp\n")
    print(df.to_string(index=False))

    # Thống kê Kích thước ảnh
    if img_sizes:
        df_sizes = pd.DataFrame(img_sizes, columns=["Width", "Height"])
        print("\n--- Kích thước ảnh (Image Resolutions) ---")
        print(f"Kích thước nhỏ nhất (Min): {df_sizes['Width'].min()}x{df_sizes['Height'].min()}")
        print(f"Kích thước lớn nhất (Max): {df_sizes['Width'].max()}x{df_sizes['Height'].max()}")
        print(f"Kích thước trung bình (Avg): {int(df_sizes['Width'].mean())}x{int(df_sizes['Height'].mean())}")

    # Vẽ biểu đồ phân bố lớp
    plt.figure(figsize=(10, 5))
    sns.barplot(data=df, x="Lớp / Class", y="Số lượng ảnh", palette="viridis")
    plt.title("Phân bố số lượng mẫu theo từng Lớp biển báo", fontsize=14)
    plt.xlabel("Lớp / Class", fontsize=12)
    plt.ylabel("Số lượng ảnh", fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    return df

def analyze_yolo_labels(labels_dir):
    """Phân tích Dataset nếu dữ liệu ở định dạng nhãn YOLO (.txt)"""
    print("="*50)
    print("PHÂN TÍCH DATASET DẠNG NHÃN YOLO (.txt)")
    print("="*50)
    
    txt_files = glob.glob(os.path.join(labels_dir, "*.txt"))
    if not txt_files:
        print("[-] Không tìm thấy file nhãn .txt!")
        return
        
    class_counts = {}
    total_objects = 0
    
    for file_path in txt_files:
        with open(file_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if parts:
                    cls_id = int(parts[0])
                    class_counts[cls_id] = class_counts.get(cls_id, 0) + 1
                    total_objects += 1
                    
    df = pd.DataFrame(list(class_counts.items()), columns=["Class ID", "Số lượng Bounding Box"])
    df = df.sort_values(by="Class ID")
    df["Tỷ lệ (%)"] = (df["Số lượng Bounding Box"] / total_objects * 100).round(2)
    
    print(f"\n[+] Tổng số file nhãn: {len(txt_files)}")
    print(f"[+] Tổng số đối tượng biển báo (Bounding Boxes): {total_objects}\n")
    print(df.to_string(index=False))

# ==========================================
# CHẠY PHÂN TÍCH
# ==========================================
if __name__ == "__main__":
    if os.path.exists(DATASET_DIR):
        # 1. Chạy phân tích dạng Thư mục
        df_result = analyze_folder_based_dataset(DATASET_DIR)
        
        # 2. Nếu dataset có thư mục 'labels', chạy phân tích dạng YOLO
        labels_path = os.path.join(DATASET_DIR, "labels")
        if os.path.exists(labels_path):
            analyze_yolo_labels(labels_path)
    else:
        print(f"[-] Đường dẫn '{DATASET_DIR}' không tồn tại. Vui lòng điền đúng đường dẫn!")