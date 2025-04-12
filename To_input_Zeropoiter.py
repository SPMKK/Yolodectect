import os
import argparse
import numpy as np
from PIL import Image
import torch
from ultralytics import YOLO
from sklearn.cluster import KMeans


class DominantColorExtractor:
    def __init__(self, n_colors=1):
        self.n_colors = n_colors

    def extract_from_box(self, image, box):
        """
        Trích xuất màu chủ đạo từ một bounding box.
        box: [x_center, y_center, width, height] (normalized YOLO format)
        """
        W, H = image.size
        x, y, w, h = box
        left = int((x - w / 2) * W)
        top = int((y - h / 2) * H)
        right = int((x + w / 2) * W)
        bottom = int((y + h / 2) * H)

        cropped = image.crop((left, top, right, bottom))
        cropped = cropped.resize((100, 100))  # Resize cho nhanh
        img_np = np.array(cropped).reshape(-1, 3)

        if len(img_np) < self.n_colors:
            return [(0, 0, 0)]

        kmeans = KMeans(n_clusters=self.n_colors, random_state=42, n_init='auto')
        kmeans.fit(img_np)
        dominant = kmeans.cluster_centers_.astype(int)
        return [tuple(c) for c in dominant]

    def get_masked_image(self, image, box, tolerance=30):
        """
        Giữ lại vùng trong bbox có màu gần giống màu chủ đạo, các vùng còn lại chuyển thành trắng.
        """
        W, H = image.size
        x, y, w, h = box
        left = int((x - w / 2) * W)
        top = int((y - h / 2) * H)
        right = int((x + w / 2) * W)
        bottom = int((y + h / 2) * H)

        img_np = np.array(image)
        cropped = img_np[top:bottom, left:right]
        if cropped.size == 0:
            return image  # tránh lỗi nếu bbox lệch

        dominant_color = self.extract_from_box(image, box)[0]

        # Tính khoảng cách đến dominant color
        dist = np.linalg.norm(cropped - np.array(dominant_color), axis=2)
        mask = dist < tolerance  # giữ pixel gần dominant

        # Tạo ảnh kết quả
        new_img = np.ones_like(img_np) * 255  # nền trắng
        new_img[top:bottom, left:right][mask] = cropped[mask]

        return Image.fromarray(new_img)


class PanoramaProcessor:
    def __init__(self, device='cuda'):
        self.device = device

    def _panorama_to_plane(self, panorama_tensor, FOV, output_size, yaw, pitch):
        pano_tensor = panorama_tensor.to(self.device)
        pano_tensor = pano_tensor.permute(2, 0, 1).float() / 255  # [3, H, W]
        pano_h, pano_w = pano_tensor.shape[1:]

        W, H = output_size
        f = (0.5 * W) / np.tan(np.radians(FOV) / 2)
        yaw_r, pitch_r = np.radians(yaw), np.radians(pitch)

        u, v = torch.meshgrid(torch.arange(W, device=self.device), torch.arange(H, device=self.device), indexing='xy')
        x = u - W / 2
        y = H / 2 - v
        z = torch.full_like(x, f)

        norm = torch.sqrt(x ** 2 + y ** 2 + z ** 2)
        x, y, z = x / norm, y / norm, z / norm

        dirs = torch.stack([x, y, z], dim=0)  # [3, H, W]
        dirs = dirs.reshape(3, -1).contiguous()  # [3, H*W]

        # Rotation matrices
        sin_pitch, cos_pitch = np.sin(pitch_r), np.cos(pitch_r)
        sin_yaw, cos_yaw = np.sin(yaw_r), np.cos(yaw_r)

        Rx = torch.tensor([
            [1, 0, 0],
            [0, cos_pitch, -sin_pitch],
            [0, sin_pitch, cos_pitch]
        ], dtype=torch.float32, device=self.device)

        Rz = torch.tensor([
            [cos_yaw, -sin_yaw, 0],
            [sin_yaw, cos_yaw, 0],
            [0, 0, 1]
        ], dtype=torch.float32, device=self.device)

        R = Rz @ Rx
        rotated_dirs = R @ dirs  # [3, H*W]
        x3, y3, z3 = rotated_dirs[0], rotated_dirs[1], rotated_dirs[2]

        # Convert to spherical
        theta = torch.acos(z3.clamp(-1, 1))
        phi = torch.atan2(y3, x3) % (2 * np.pi)

        U = phi * pano_w / (2 * np.pi)
        V = theta * pano_h / np.pi

        # Normalize to [-1, 1]
        U_norm = 2 * (U / pano_w) - 1
        V_norm = 2 * (V / pano_h) - 1
        grid = torch.stack((U_norm, V_norm), dim=-1).reshape(H, W, 2).unsqueeze(0)

        pano_tensor = pano_tensor.unsqueeze(0)
        sampled = torch.nn.functional.grid_sample(pano_tensor, grid, mode='bilinear', align_corners=True)

        sampled_img = (sampled.squeeze(0).permute(1, 2, 0).clamp(0, 1) * 255).byte().cpu().numpy()
        return sampled_img


    def generate_all_views(self, image_path):
        pano = Image.open(image_path).convert('RGB')
        pano_tensor = torch.from_numpy(np.array(pano)).to(self.device)

        result_images = []
        for yaw in np.linspace(0, 360, 16):
            for pitch in [30, 60, 90, 120, 150]:
                view = self._panorama_to_plane(pano_tensor, 100, (640, 640), yaw, pitch)
                result_images.append(view)
        return result_images

class ObjectDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def get_best_view(self, images):
        max_score = 0
        best_img = None

        for i, img in enumerate(images):
            result = self.model(img, verbose=False)[0]

            score = 0
            if hasattr(result, "boxes") and result.boxes and result.boxes.xywh is not None:
                boxes = result.boxes.xywh.cpu().numpy()  # (x_center, y_center, w, h)
                img_h, img_w = img.shape[:2]

                # Loại bbox nằm sát viền
                filtered_boxes = []
                for box in boxes:
                    x_center, y_center, w, h = box
                    x_min = x_center - w / 2
                    y_min = y_center - h / 2
                    x_max = x_center + w / 2
                    y_max = y_center + h / 2

                    margin = 5  # pixel, hoặc 0.05 * width tùy độ chặt

                    if (
                        x_min > margin and y_min > margin and
                        x_max < img_w - margin and y_max < img_h - margin
                    ):
                        filtered_boxes.append(box)

                filtered_boxes = np.array(filtered_boxes)
                if len(filtered_boxes) > 0:
                    total_area = np.sum(filtered_boxes[:, 2] * filtered_boxes[:, 3])
                    num_boxes = filtered_boxes.shape[0]
                    score = total_area + 1000 * num_boxes

                    print(f"View {i}: {num_boxes} valid boxes, total bbox area = {total_area}")
                else:
                    print(f"View {i}: All boxes too close to edge.")
            else:
                print("No boxes found.")

            if score > max_score:
                max_score = score
                best_img = img
                print("This is the new best view.")

        print(f"\nBest score: {max_score}")
        return best_img




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image_name", type=str, help="Panorama image file (e.g. 11_colors.png)")
    parser.add_argument("--model_path", type=str, required=True, help="Path to YOLO model (e.g. best.pt)")
    parser.add_argument("--save_dir", type=str, default="SelectedImages", help="Where to save final selected image")
    parser.add_argument("--dominant_colors", type=int, default=3, help="Number of dominant colors to extract from the best view")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    pano_name = os.path.splitext(os.path.basename(args.image_name))[0]

    processor = PanoramaProcessor(device='cuda')
    views = processor.generate_all_views(args.image_name)

    detector = ObjectDetector(args.model_path)
    best_img_np = detector.get_best_view(views)

    if best_img_np is not None:
        best_img_pil = Image.fromarray(best_img_np)
        save_path = os.path.join(args.save_dir, f"{pano_name}.png")
        best_img_pil.save(save_path)
        print(f"✔ Saved best view to {save_path}")

        # Extract dominant colors from the best view
        color_extractor = DominantColorExtractor(n_colors=args.dominant_colors)
        results = detector.model(best_img_np, verbose=False)[0]

        if results.boxes is not None and len(results.boxes) > 0:
            dominant_colors_list = []
            for i, box in enumerate(results.boxes.xywhn.cpu().numpy()):
                dominant_colors = color_extractor.extract_from_box(best_img_pil, box)
                dominant_colors_list.append(dominant_colors)
                print(f"  Object {i+1}: {dominant_colors}")

                # ➕ Mask ảnh giữ lại vùng màu chính
                masked = color_extractor.get_masked_image(best_img_pil, box, tolerance=30)
                masked_path = os.path.join(args.save_dir, f"{pano_name}_masked_obj{i+1}.png")
                masked.save(masked_path)
                print(f"    ✔ Saved masked object {i+1} to {masked_path}")

            print("\n✔ Masked images and dominant colors extracted successfully.")
        else:
            print("\nNo objects detected in the best view to extract dominant colors.")
    else:
        print("✘ No object detected in any view.")



if __name__ == '__main__':
    main()