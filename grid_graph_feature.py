import torch
import timm
from pycocotools.coco import COCO
import numpy as np
import cv2
from tqdm import tqdm
import h5py

DATA_DIR = "/path/to/COCO2017"  # Thư mục gốc chứa dữ liệu COCO 2017
MODEL_NAME = "swin_base_patch4_window7_224"  # Có thể đổi thành "swin_tiny_patch4_window7_224" nếu cần
BATCH_SIZE = 64  

train_ann_file = "/kaggle/input/2017-2017/annotations_trainval2017/annotations/instances_train2017.json"
val_ann_file   = "/kaggle/input/2017-2017/annotations_trainval2017/annotations/instances_val2017.json"
train_img_dir  = "/kaggle/input/2017-2017/train2017/train2017"
val_img_dir    = "/kaggle/input/2017-2017/val2017/val2017"

# Tải mô hình Swin Transformer đã huấn luyện trước
model = timm.create_model(MODEL_NAME, pretrained=True)
model.eval()  # chế độ eval, không training

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

@torch.no_grad()
def extract_features_batch(image_paths):

    batch_images = []
    batch_ids = []
    for img_path, img_id in image_paths:
        # Đọc ảnh bằng OpenCV (BGR)
        img = cv2.imread(img_path)
        if img is None:
            # Nếu ảnh không đọc được, bỏ qua (có thể in cảnh báo)
            continue
        # Chuyển BGR -> RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Resize ảnh về 224x224
        img_resized = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)
        # Chuyển sang float32 và chuẩn hóa về [0,1]
        img_float = img_resized.astype(np.float32) / 255.0
        # Chuẩn hóa theo thống kê ImageNet
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_norm = (img_float - mean) / std
        # Đổi chiều thành (3, H, W) cho PyTorch
        img_chw = np.transpose(img_norm, (2, 0, 1))
        batch_images.append(img_chw)
        batch_ids.append(img_id)
    if len(batch_images) == 0:
        return []  # không có ảnh hợp lệ trong batch
    # Đóng gói batch thành tensor
    batch_images_np = np.stack(batch_images, axis=0)  # shape: (batch_size, 3, 224, 224)
    batch_tensor = torch.from_numpy(batch_images_np).to(device)
    # Trích xuất đặc trưng patch từ Swin Transformer
    features = model.forward_features(batch_tensor)
    # Xử lý đầu ra để lấy đặc trưng các patch (node) cho mỗi ảnh
    # Kiểm tra và chuyển định dạng output về (B, num_patches, feature_dim)
    if isinstance(features, (list, tuple)):
        # Nếu model với features_only=True trả về list, lấy đặc trưng tầng cuối cùng
        features = features[-1]
    # features có thể ở dạng (B, H, W, C) hoặc (B, C, H, W) tùy mô hình
    if features.dim() == 4:
        if features.shape[-1] > features.shape[1]:
            # Định dạng (B, H, W, C)
            B, H, W, C = features.shape
            feat_vectors = features.reshape(B, H*W, C)  # nối HxW patch thành một chiều
        else:
            # Định dạng (B, C, H, W)
            B, C, H, W = features.shape
            feat_map = features.permute(0, 2, 3, 1)     # chuyển về (B, H, W, C)
            feat_vectors = feat_map.reshape(B, H*W, C)  # nối HxW patch
    else:
        # Trường hợp đầu ra đã phẳng (B, N, C)
        feat_vectors = features
    feat_vectors = feat_vectors.cpu().numpy().astype(np.float32)  # chuyển về CPU numpy float32
    # Ghép ID với vector đặc trưng tương ứng
    results = [(img_id, feat_vectors[idx]) for idx, img_id in enumerate(batch_ids)]
    return results

def process_dataset(coco_ann_file, img_dir, output_h5_file):

    coco = COCO(coco_ann_file)
    img_ids = coco.getImgIds()
    # Mở file HDF5 để ghi (tạo file mới)
    with h5py.File(output_h5_file, "w") as h5f:
        # Duyệt qua các ảnh theo batch
        for i in tqdm(range(0, len(img_ids), BATCH_SIZE), desc=f"Processing {output_h5_file}"):
            batch_ids = img_ids[i : i + BATCH_SIZE]
            # Lấy thông tin file ảnh cho batch
            images_info = coco.loadImgs(batch_ids)
            # Chuẩn bị danh sách (đường dẫn, ID) cho batch
            image_paths = [(f"{img_dir}/{img_info['file_name']}", img_info['id']) 
                           for img_info in images_info]

            batch_results = extract_features_batch(image_paths)

            for img_id, features in batch_results:

                h5f.create_dataset(str(img_id), data=features, dtype='float32')
    print(f"Đã lưu đặc trưng cho {len(img_ids)} ảnh vào {output_h5_file}")

process_dataset(train_ann_file, train_img_dir, "coco_train2017_swin_features.h5")
process_dataset(val_ann_file,   val_img_dir,   "coco_val2017_swin_features.h5")
