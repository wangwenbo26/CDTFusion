import torch
import torch.optim as optim
import kornia.losses as losses
from util.data_utils import get_train_loader
import torch.nn.functional as F
import torch.nn as nn
from util.encoder import EncoderVi
from util.encoder import EncoderIr
from util.res_decoder import head_1
from util.res_decoder import head_2
from util.transfer import transfer
from util.discriminator import DomDiscriminator
from util.MeanVarianceLoss import LocalMeanVarianceLoss
import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

batch_size = 4
epochs = 20
lr = 0.0001

dir_vi_train = "datasets/whu/train/vi"
dir_ir_train = "datasets/whu/train/ir"

encoder_1_path = "save/whu/encoder_1.pth"
encoder_2_path = "save/whu/encoder_2.pth"
transfer_vi_to_ir_path = "save/whu/transfer_vi_to_ir.pth"
transfer_ir_to_vi_path = "save/whu/transfer_ir_to_vi.pth"
domain_discriminator_path = "save/whu/domain_discriminator.pth"
decoder_1_path = "save/whu/decoder_1.pth"
decoder_2_path = "save/whu/decoder_2.pth"

def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

train_loader = get_train_loader(dir_vi_train, dir_ir_train, batch_size)

encoder_1 = EncoderVi()
encoder_2 = EncoderIr()
transfer_vi_to_ir = transfer()
transfer_ir_to_vi = transfer()
domain_discriminator = DomDiscriminator()
decoder_1 = head_1()
decoder_2 = head_2()

encoder_1.to(device)
encoder_2.to(device)
transfer_vi_to_ir.to(device)
transfer_ir_to_vi.to(device)
domain_discriminator.to(device)
decoder_1.to(device)
decoder_2.to(device)


optimizer_encoder_1 = optim.Adam(encoder_1.parameters(), lr=lr)
optimizer_encoder_2 = optim.Adam(encoder_2.parameters(), lr=lr)
optimizer_transfer_vi_to_ir = optim.Adam(transfer_vi_to_ir.parameters(), lr=lr)
optimizer_transfer_ir_to_vi = optim.Adam(transfer_ir_to_vi.parameters(), lr=lr)
optimizer_domain_discriminator = optim.Adam(domain_discriminator.parameters(), lr=lr)
optimizer_decoder_1 = optim.Adam(decoder_1.parameters(), lr=lr)
optimizer_decoder_2 = optim.Adam(decoder_2.parameters(), lr=lr)

encoder_1.train()
encoder_2.train()
transfer_vi_to_ir.train()
transfer_ir_to_vi.train()
domain_discriminator.train()
decoder_1.train()
decoder_2.train()

ssim_loss = losses.SSIMLoss(window_size=11, reduction='mean')
bce_loss = nn.BCELoss()
loss_calculator = LocalMeanVarianceLoss(window_size=4)

for epoch in range(epochs):
    lambda_domain = 0.5
    lambda_mean_vari = 0.5
    for batch_idx, (vi_images, ir_images) in enumerate(train_loader):
        vi_images, ir_images = vi_images.to(device), ir_images.to(device)

        encoder_1.eval()
        encoder_2.eval()
        with torch.no_grad():
            vi_features = encoder_1(vi_images)
            ir_features = encoder_2(ir_images)

        domain_vi_labels = torch.ones(vi_features.size(0), 1).to(device)
        domain_ir_labels = torch.zeros(ir_features.size(0), 1).to(device)
        domain_un_labels = (domain_ir_labels + domain_vi_labels) / 2

        domain_discriminator.train()
        for param in domain_discriminator.parameters():
            param.requires_grad = True

        optimizer_domain_discriminator.zero_grad()

        domain_vi_pred = domain_discriminator(vi_features)
        domain_ir_pred = domain_discriminator(ir_features)

        domain_loss_real = bce_loss(domain_vi_pred, domain_vi_labels) + bce_loss(domain_ir_pred, domain_ir_labels)
        domain_loss_real.backward()
        optimizer_domain_discriminator.step()

        encoder_1.train()
        encoder_2.train()
        domain_discriminator.eval()
        for param in domain_discriminator.parameters():
            param.requires_grad = False

        optimizer_encoder_1.zero_grad()
        optimizer_encoder_2.zero_grad()
        optimizer_transfer_vi_to_ir.zero_grad()
        optimizer_transfer_ir_to_vi.zero_grad()
        optimizer_decoder_1.zero_grad()
        optimizer_decoder_2.zero_grad()

        vi_features = encoder_1(vi_images)
        ir_features = encoder_2(ir_images)

        vi_allin = transfer_vi_to_ir(vi_features, ir_features)
        ir_allin = transfer_ir_to_vi(ir_features, vi_features)

        ir_out = decoder_2(vi_allin)
        vi_out = decoder_1(ir_allin)

        reir_loss = F.l1_loss(ir_out, ir_images) + ssim_loss(ir_out, ir_images)
        revi_loss = F.l1_loss(vi_out, vi_images) + ssim_loss(vi_out, vi_images)
        recon_loss = revi_loss + reir_loss


        domain_vi_a_pred = domain_discriminator(vi_allin)
        domain_ir_a_pred = domain_discriminator(ir_allin)

        domain_loss_fake = bce_loss(domain_vi_a_pred, domain_un_labels) + bce_loss(domain_ir_a_pred, domain_un_labels)
        domain_loss = domain_loss_fake


        mean_loss, variance_loss = loss_calculator(ir_allin, vi_allin)
        m_v_loss = mean_loss + variance_loss

        total_loss = recon_loss + lambda_domain * domain_loss + lambda_mean_vari * m_v_loss
        total_loss.backward()

        optimizer_encoder_1.step()
        optimizer_encoder_2.step()
        optimizer_transfer_vi_to_ir.step()
        optimizer_transfer_ir_to_vi.step()
        optimizer_decoder_1.step()
        optimizer_decoder_2.step()

        if batch_idx % 100 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Step [{batch_idx}/{len(train_loader)}], total_loss: {total_loss.item():.4f}")


save_model(encoder_1, encoder_1_path)
save_model(encoder_2, encoder_2_path)
save_model(transfer_vi_to_ir, transfer_vi_to_ir_path)
save_model(transfer_ir_to_vi, transfer_ir_to_vi_path)
save_model(domain_discriminator, domain_discriminator_path)
save_model(decoder_1, decoder_1_path)
save_model(decoder_2, decoder_2_path)
