import torch
from util.encoder import EncoderVi
from util.encoder import EncoderIr
from util.res_decoder import head_fus
from util.transfer import transfer
from util.mixer import Mixer
from util.taskinteraction import TaskInteraction
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
import os
import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


encoder_1 = EncoderVi()
encoder_2 = EncoderIr()
transfer_vi_to_ir = transfer()
transfer_ir_to_vi = transfer()
decoder_4 = head_fus()
hidden_dim_channel = 1024
hidden_dim_token = 512
mixer = Mixer(512, 512, hidden_dim_channel, hidden_dim_token)
mixer_f = Mixer(512, 512, hidden_dim_channel, hidden_dim_token)
taskinteraction = TaskInteraction(in_channels=512)

encoder_1.load_state_dict(torch.load('save/pos/encoder_1.pth'))
encoder_2.load_state_dict(torch.load('save/pos/encoder_2.pth'))
transfer_vi_to_ir.load_state_dict(torch.load('save/pos/transfer_vi_to_ir.pth'))
transfer_ir_to_vi.load_state_dict(torch.load('save/pos/transfer_ir_to_vi.pth'))

decoder_4.load_state_dict(torch.load('save/pos/decoder_4.pth'))
mixer.load_state_dict(torch.load('save/pos/mixer.pth'))
mixer_f.load_state_dict(torch.load('save/pos/mixer_f.pth'))
taskinteraction.load_state_dict(torch.load('save/pos/task_interaction.pth'))

encoder_1.to(device)
encoder_2.to(device)
transfer_vi_to_ir.to(device)
transfer_ir_to_vi.to(device)
decoder_4.to(device)
mixer.to(device)
mixer_f.to(device)
taskinteraction.to(device)

transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor()
])

# 定义函数用于融合图片
def fusion_images(rgb_path, ir_path, image_path):
    rgb_image = Image.open(rgb_path).convert("RGB")
    ir_image = Image.open(ir_path).convert("L")

    img_array = np.array(rgb_image)
    img_hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)

    rgb_tensor = transform(rgb_image).unsqueeze(0).to(device)
    ir_tensor = transform(ir_image).unsqueeze(0).to(device)

    # 使用模型进行融合
    with torch.no_grad():
        test_vi_features = encoder_1(rgb_tensor)
        test_ir_features = encoder_2(ir_tensor)

        test_vi_allin = transfer_vi_to_ir(test_vi_features, test_ir_features)
        test_ir_allin = transfer_ir_to_vi(test_ir_features, test_vi_features)

        test_seg_features = mixer(test_vi_allin, test_ir_allin)
        test_fus_features = mixer_f(test_vi_allin, test_ir_allin)
        test_fus_features_taski = taskinteraction(test_fus_features, test_seg_features)

        generated_fus_images = decoder_4(test_fus_features_taski)
        ones_1 = torch.ones_like(generated_fus_images)
        zeros_1 = torch.zeros_like(generated_fus_images)
        generated_fus_images = torch.where(generated_fus_images > ones_1, ones_1, generated_fus_images)
        generated_fus_images = torch.where(generated_fus_images < zeros_1, zeros_1, generated_fus_images)

        fus_img_pred = generated_fus_images.cpu().detach().numpy().squeeze()
        fus_img_pred = np.uint8(255.0 * fus_img_pred)

        img_hsv[:, :, 2] = fus_img_pred
        modifited_rgb_img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
        modifited_rgb_img = Image.fromarray(modifited_rgb_img)
        modifited_rgb_img.save(image_path)


dir_vi_test = "datasets/pos/test/vi"
dir_ir_test = "datasets/pos/test/ir"
output_dir = 'fusion/pos'
os.makedirs(output_dir, exist_ok=True)


test_vi_files = [f for f in os.listdir(dir_vi_test) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
for filename in test_vi_files:
    rgb_path = os.path.join(dir_vi_test, filename)
    ir_path = os.path.join(dir_ir_test, filename)
    output_path = os.path.join(output_dir, filename)

    if not os.path.exists(ir_path):
        print(f"Skipping {filename}: IR image not found")
        continue

    fusion_images(rgb_path, ir_path, output_path)
    print(f"Processed test image: {filename}")