import torch
import torch.optim as optim
from util.seg_dataloader_whu import get_train_loader
import torch.nn as nn
from util.encoder import EncoderVi
from util.encoder import EncoderIr
from util.res_decoder import head_seg_whu
from util.res_decoder import head_fus
from util.transfer import transfer
from util.mixer import Mixer
from util.taskinteraction import TaskInteraction
from util.loss_ssim import Fusionloss_ir
from util.dice_loss import dice_loss
from util.imageutil import RGB2YCrCb,YCrCb2RGB
import warnings
warnings.filterwarnings("ignore")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


dir_vi_train = "datasets/whu/train/vi"
dir_ir_train = "datasets/whu/train/ir"
dir_seg_train = "datasets/whu/train/lbl"

decoder_3_path = "save/whu/decoder_3.pth"
decoder_4_path = "save/whu/decoder_4.pth"
mixer_path = "save/whu/mixer.pth"
mixer_f_path = "save/whu/mixer_f.pth"
task_interaction_path = "save/whu/task_interaction.pth"

def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


encoder_1 = EncoderVi()
encoder_2 = EncoderIr()
transfer_vi_to_ir = transfer()
transfer_ir_to_vi = transfer()
decoder_3 = head_seg_whu()
decoder_4 = head_fus()
hidden_dim_channel = 1024
hidden_dim_token = 512
mixer = Mixer(512, 512, hidden_dim_channel, hidden_dim_token)
mixer_f = Mixer(512, 512, hidden_dim_channel, hidden_dim_token)
taskinteraction = TaskInteraction(in_channels=512)

encoder_1.load_state_dict(torch.load('save/whu/encoder_1.pth'))
encoder_2.load_state_dict(torch.load('save/whu/encoder_2.pth'))
transfer_vi_to_ir.load_state_dict(torch.load('save/whu/transfer_vi_to_ir.pth'))
transfer_ir_to_vi.load_state_dict(torch.load('save/whu/transfer_ir_to_vi.pth'))


encoder_1.to(device)
encoder_2.to(device)
transfer_vi_to_ir.to(device)
transfer_ir_to_vi.to(device)
decoder_3.to(device)
decoder_4.to(device)
mixer.to(device)
mixer_f.to(device)
taskinteraction.to(device)


batch_size = 4
epochs = 30
lr=0.0001
optimizer_decoder_3 = optim.Adam(decoder_3.parameters(), lr=lr)
optimizer_decoder_4 = optim.Adam(decoder_4.parameters(), lr=lr)
optimizer_mixer = optim.Adam(mixer.parameters(), lr=lr)
optimizer_mixer_f = optim.Adam(mixer_f.parameters(), lr=lr)
optimizer_taskinteraction = optim.Adam(taskinteraction.parameters(), lr=lr)

train_loader = get_train_loader(dir_vi_train, dir_ir_train, dir_seg_train, batch_size)

decoder_3.train()
decoder_4.train()
mixer.train()
mixer_f.train()
taskinteraction.train()

loss = nn.BCEWithLogitsLoss()
criterion = nn.CrossEntropyLoss()

fusionloss = Fusionloss_ir()

for epoch in range(epochs):
    for batch_idx, (vi_images, ir_images, seg_images) in enumerate(train_loader):
        vi_images, ir_images, seg_images = vi_images.to(device), ir_images.to(device), seg_images.to(device)
        rgb_images_ycrcb = RGB2YCrCb(vi_images)
        opt_images = rgb_images_ycrcb[:, 0:1, :, :]

        optimizer_decoder_3.zero_grad()
        optimizer_decoder_4.zero_grad()
        optimizer_mixer.zero_grad()
        optimizer_mixer_f.zero_grad()
        optimizer_taskinteraction.zero_grad()
        with torch.no_grad():
            vi_features = encoder_1(vi_images)
            ir_features = encoder_2(ir_images)
            vi_allin = transfer_vi_to_ir(vi_features, ir_features)
            ir_allin = transfer_ir_to_vi(ir_features, vi_features)

        seg_features = mixer(vi_allin, ir_allin)
        seg_out = decoder_3(seg_features)

        seg_loss = dice_loss(seg_out, seg_images, num_classes=8) + 0.5 * criterion(seg_out, seg_images)


        fus_features = mixer_f(vi_allin, ir_allin)
        fus_features_taski = taskinteraction(fus_features, seg_features)
        fus_out = decoder_4(fus_features_taski)

        fus_loss = fusionloss(opt_images, ir_images, fus_out)

        total_loss = seg_loss + fus_loss
        total_loss.backward()

        optimizer_decoder_3.step()
        optimizer_mixer.step()
        optimizer_decoder_4.step()
        optimizer_mixer_f.step()
        optimizer_taskinteraction.step()

        if batch_idx % 100 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Step [{batch_idx}/{len(train_loader)}], total_loss: {total_loss.item():.4f}")

save_model(decoder_3, decoder_3_path)
save_model(decoder_4, decoder_4_path)
save_model(mixer, mixer_path)
save_model(mixer_f, mixer_f_path)
save_model(taskinteraction, task_interaction_path)