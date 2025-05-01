import os
import torch
import dlib
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from models import Generator
from proj2_starter import poisson_blend

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
face_detector = dlib.get_frontal_face_detector()

def load_generator(checkpoint_dir, opts):
    G = Generator(conv_dim=opts.g_conv_dim, norm=opts.norm)
    ckpt_path = os.path.join(checkpoint_dir, "G.pkl")
    if os.path.exists(ckpt_path):
        print(f"Loading Generator from checkpoint: {ckpt_path}")
        G.load_state_dict(torch.load(ckpt_path, map_location=device))
        G.to(device)
        G.eval()
    else:
        raise FileNotFoundError(f"No generator checkpoint found at: {ckpt_path}")
    return G

def preprocess_image(img_path):
    img = Image.open(img_path).convert("RGB").resize((128, 128))
    return img

def blur_face(pil_img):
    img_np = np.array(pil_img)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    faces = face_detector(gray, 1)

    if faces:
        for face in faces:
            x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img_np.shape[1], x2), min(img_np.shape[0], y2)

            face_region = img_np[y1:y2, x1:x2]
            blurred_face = cv2.GaussianBlur(face_region, (25, 25), 30)

            height, width = y2 - y1, x2 - x1
            mask = np.zeros((height, width, 3), dtype=np.uint8)

            center = (width // 2, height // 2)
            axes = (int(width * 0.4), int(height * 0.6))
            cv2.ellipse(mask, center, axes, 0, 0, 360, (255, 255, 255), -1)

            masked_blur = np.where(mask == 255, blurred_face, face_region)
            img_np[y1:y2, x1:x2] = masked_blur
        img_res = Image.fromarray(img_np)
        img_res.save("output0.jpg")

        return Image.fromarray(img_np), (x1, y1, x2, y2)
    return pil_img, None

def pil_to_tensor(pil_img):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])
    return transform(pil_img).unsqueeze(0)

def tensor_to_pil(tensor):
    tensor = tensor.squeeze().detach().cpu()
    tensor = (tensor * 0.5 + 0.5).clamp(0, 1)
    return transforms.ToPILImage()(tensor)

def get_dummy_one_hot(index=0, num_classes=31):
    vec = torch.zeros(num_classes)
    vec[index] = 1
    return vec.unsqueeze(0).to(device)

def run_inference(img_path, G, output_path="output", index=0):
    orig_img = preprocess_image(img_path)
    blurred_img, face_box = blur_face(orig_img)

    input_tensor = pil_to_tensor(blurred_img).to(device)
    with torch.no_grad():
        one_hot_vector = get_dummy_one_hot(index, num_classes=31)
        output_tensor = G(input_tensor, one_hot_vector)

    gen_img = tensor_to_pil(output_tensor)
    gen_np = np.array(gen_img)
    orig_np = np.array(orig_img)
    blend_img = gen_np
    
    gen_res = Image.fromarray(gen_np)
    gen_res.save(output_path + "1.jpg")

    if face_box:
        x1, y1, x2, y2 = face_box
        fg = orig_np
        bg = gen_np
        mask = np.zeros((128, 128, 3), dtype=np.uint8)
        mask[y1:y2, x1:x2] = [255, 255, 255]
        ratio = 1
        mask = cv2.resize(mask, (0, 0), fx=ratio, fy=ratio)

        fg = fg / 255.
        bg = bg / 255.
        mask = (mask.sum(axis=2, keepdims=True) > 0)

        blend_img = poisson_blend(fg, mask, bg)
        blend_img_uint8 = (blend_img * 255).astype(np.uint8) if blend_img.dtype == np.float64 else blend_img.astype(np.uint8)
        gen_np[y1:y2, x1:x2] = orig_np[y1:y2, x1:x2]
        gen2_res = Image.fromarray(gen_np)
        gen2_res.save(output_path + "2.jpg")

    final_img = Image.fromarray(blend_img_uint8)
    final_img.save(output_path + '.jpg')
    print(f"Saved result to {output_path}")

if __name__ == "__main__":
    class Opts:
        g_conv_dim = 32
        norm = 'instance'
        checkpoint_dir = 'checkpoints_stylegan_clip/2000itr'

    opts = Opts()
    generator = load_generator(opts.checkpoint_dir, opts)

    test_image_path = "test.jpg"
    run_inference(test_image_path, generator, output_path="output", index=15)
