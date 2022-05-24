import sys

sys.path.append("PerceptualSimilarity\\")
import os
import utils
import torch
import numpy as np
from torch import nn
import torchgeometry
from kornia import color
import torch.nn.functional as F
from torchvision import transforms
import torchaudio.transforms as T


class Dense(nn.Module):
    def __init__(self, in_features, out_features, activation='relu', kernel_initializer='he_normal'):
        super(Dense, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation
        self.kernel_initializer = kernel_initializer

        self.linear = nn.Linear(in_features, out_features)
        # initialization
        if kernel_initializer == 'he_normal':
            nn.init.kaiming_normal_(self.linear.weight)
        else:
            raise NotImplementedError

    def forward(self, inputs):
        outputs = self.linear(inputs)
        if self.activation is not None:
            if self.activation == 'relu':
                outputs = nn.ReLU(inplace=True)(outputs)
        return outputs


class Conv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, activation='relu', strides=1):
        super(Conv2D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.activation = activation
        self.strides = strides

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, strides, int((kernel_size - 1) / 2))
        # default: using he_normal as the kernel initializer
        nn.init.kaiming_normal_(self.conv.weight)

    def forward(self, inputs):
        outputs = self.conv(inputs)
        if self.activation is not None:
            if self.activation == 'relu':
                outputs = nn.ReLU(inplace=True)(outputs)
            else:
                raise NotImplementedError
        return outputs


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, input):
        return input.view(input.size(0), -1)


class StegaStampEncoder(nn.Module):
    def __init__(self, n_frames: int, height: int):
        super(StegaStampEncoder, self).__init__()
        self.n_frames = n_frames
        self.height = height

        self.secret_dense = Dense(100, height * n_frames // 8 // 8, activation='relu', kernel_initializer='he_normal')

        self.conv1 = Conv2D(2, 32, 3, activation='relu')
        self.conv2 = Conv2D(32, 32, 3, activation='relu', strides=2)
        self.conv3 = Conv2D(32, 64, 3, activation='relu', strides=2)
        self.conv4 = Conv2D(64, 128, 3, activation='relu', strides=2)
        self.conv5 = Conv2D(128, 256, 3, activation='relu', strides=2)
        self.up6 = Conv2D(256, 128, 3, activation='relu')
        self.conv6 = Conv2D(256, 128, 3, activation='relu')
        self.up7 = Conv2D(128, 64, 3, activation='relu')
        self.conv7 = Conv2D(128, 64, 3, activation='relu')
        self.up8 = Conv2D(64, 32, 3, activation='relu')
        self.conv8 = Conv2D(64, 32, 3, activation='relu')
        self.up9 = Conv2D(32, 32, 3, activation='relu')
        self.conv9 = Conv2D(66, 32, 3, activation='relu')
        self.residual = Conv2D(32, 1, 1, activation=None)

    def forward(self, inputs):
        secrect, image = inputs
        # secrect = secrect - .5
        # image = image - .5

        secrect = self.secret_dense(secrect)
        secrect = secrect.reshape(-1, 1, self.height // 8, self.n_frames // 8)
        secrect_enlarged = nn.Upsample(scale_factor=(8, 8))(secrect)

        inputs = torch.cat([secrect_enlarged, image], dim=1)
        conv1 = self.conv1(inputs)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)
        up6 = self.up6(nn.Upsample(scale_factor=(2, 2))(conv5))
        merge6 = torch.cat([conv4, up6], dim=1)
        conv6 = self.conv6(merge6)
        up7 = self.up7(nn.Upsample(scale_factor=(2, 2))(conv6))
        merge7 = torch.cat([conv3, up7], dim=1)
        conv7 = self.conv7(merge7)
        up8 = self.up8(nn.Upsample(scale_factor=(2, 2))(conv7))
        merge8 = torch.cat([conv2, up8], dim=1)
        conv8 = self.conv8(merge8)
        up9 = self.up9(nn.Upsample(scale_factor=(2, 2))(conv8))
        merge9 = torch.cat([conv1, up9, inputs], dim=1)
        conv9 = self.conv9(merge9)
        residual = self.residual(conv9)
        return residual


# class SpatialTransformerNetwork(nn.Module):
#     def __init__(self):
#         super(SpatialTransformerNetwork, self).__init__()
#         self.localization = nn.Sequential(
#             Conv2D(3, 32, 3, strides=2, activation='relu'),
#             Conv2D(32, 64, 3, strides=2, activation='relu'),
#             Conv2D(64, 128, 3, strides=2, activation='relu'),
#             Flatten(),
#             Dense(320000, 128, activation='relu'),
#             nn.Linear(128, 6)
#         )
#         self.localization[-1].weight.data.fill_(0)
#         self.localization[-1].bias.data = torch.FloatTensor([1, 0, 0, 0, 1, 0])

#     def forward(self, image):
#         theta = self.localization(image)
#         theta = theta.view(-1, 2, 3)
#         grid = F.affine_grid(theta, image.size(), align_corners=False)
#         transformed_image = F.grid_sample(image, grid, align_corners=False)
#         return transformed_image


class StegaStampDecoder(nn.Module):
    def __init__(self, secret_size=100):
        super(StegaStampDecoder, self).__init__()
        self.secret_size = secret_size
        #         self.stn = SpatialTransformerNetwork()
        self.decoder = nn.Sequential(
            Conv2D(1, 32, 3, strides=2, activation='relu'),
            Conv2D(32, 32, 3, activation='relu'),
            Conv2D(32, 64, 3, strides=2, activation='relu'),
            Conv2D(64, 64, 3, activation='relu'),
            Conv2D(64, 64, 3, strides=2, activation='relu'),
            Conv2D(64, 128, 3, strides=2, activation='relu'),
            Conv2D(128, 128, 3, strides=2, activation='relu'),
            Flatten(),
            Dense(1024, 512, activation='relu'),
            Dense(512, secret_size, activation=None))

    def forward(self, image):
        # image = image - .5
        #         transformed_image = self.stn(image)
        return torch.sigmoid(self.decoder(image))


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            Conv2D(1, 8, 3, strides=2, activation='relu'),
            Conv2D(8, 16, 3, strides=2, activation='relu'),
            Conv2D(16, 32, 3, strides=2, activation='relu'),
            Conv2D(32, 64, 3, strides=2, activation='relu'),
            Conv2D(64, 1, 3, activation=None))

    def forward(self, image):
        x = image - .5
        x = self.model(x)
        output = torch.mean(x)
        return output, x


def get_secret_acc(secret_true, secret_pred):
    if 'cuda' in str(secret_pred.device):
        secret_pred = secret_pred.cpu()
        secret_true = secret_true.cpu()
    secret_pred = torch.round(secret_pred)
    correct_pred = torch.sum((secret_pred - secret_true) == 0, dim=1)
    str_acc = 1.0 - torch.sum((correct_pred - secret_pred.size()[1]) != 0).numpy() / correct_pred.size()[0]
    bit_acc = torch.sum(correct_pred).numpy() / secret_pred.numel()
    return bit_acc, str_acc


def transform_net(image, args, global_step):
    stretch = T.TimeStretch(fixed_rate=1.2)
    masking = T.TimeMasking(time_mask_param=80)
    fr_masking = T.FrequencyMasking(freq_mask_param=80)

    #     image = image.squeeze()

    #     image = stretch(image)
    image = masking(image)
    image = fr_masking(image)

    #     image = image.unsqueeze(1)

    return image


class TrainableLoss(nn.Module):
    def __init__(self, image_loss_c: float = 1.0, secret_loss_c: float = 1.0, ):
        super().__init__()
        self.image_loss_c = nn.Parameter(torch.tensor(image_loss_c))
        self.secret_loss_c = nn.Parameter(torch.tensor(secret_loss_c))


def build_model(encoder: StegaStampEncoder, decoder: StegaStampEncoder, secret_input, image_input, l2_edge_gain,
                borders, secret_size, M, loss_scales, yuv_scales, args, global_step, writer, t_loss):
    test_transform = transform_net(image_input, args, global_step)

    input_warped = torchgeometry.warp_perspective(image_input, M[:, 1, :, :], dsize=(encoder.height, encoder.n_frames), flags='bilinear')
    mask_warped = torchgeometry.warp_perspective(torch.ones_like(input_warped), M[:, 1, :, :], dsize=(encoder.height, encoder.n_frames),
                                                 flags='bilinear')
    input_warped += (1 - mask_warped) * image_input

    residual_warped = encoder((secret_input, input_warped))
    encoded_warped = residual_warped + input_warped

    residual = torchgeometry.warp_perspective(residual_warped, M[:, 0, :, :], dsize=(encoder.height, encoder.n_frames), flags='bilinear')

    if borders == 'no_edge':
        encoded_image = image_input + residual
    else:
        raise ValueError
    # if borders == 'no_edge':
    # D_output_real, _ = discriminator(image_input)
    # D_output_fake, D_heatmap = discriminator(encoded_image)

    transformed_image = transform_net(encoded_image, args, global_step)
    decoded_secret = decoder(transformed_image)
    bit_acc, str_acc = get_secret_acc(secret_input, decoded_secret)

    # normalized_input = image_input * 2 - 1
    # normalized_encoded = encoded_image * 2 - 1
    # lpips_loss = torch.mean(lpips_fn(normalized_input, normalized_encoded))

    cross_entropy = nn.BCELoss()
    if args.cuda:
        cross_entropy = cross_entropy.cuda()
    secret_loss = cross_entropy(decoded_secret, secret_input)

    image_loss = torch.mean((encoded_image - image_input) ** 2)

    # D_loss = D_output_real - D_output_fake
    # G_loss = D_output_fake
    loss = t_loss.image_loss_c ** 2 * image_loss + t_loss.secret_loss_c ** 2 * secret_loss + 2 * torch.log(
        t_loss.image_loss_c * t_loss.secret_loss_c)
    # if not args.no_gan:
    #     loss += loss_scales[3] * G_loss

    writer.add_scalar('loss/image_loss', image_loss, global_step)
    # writer.add_scalar('loss/lpips_loss', lpips_loss, global_step)
    writer.add_scalar('loss/secret_loss', secret_loss, global_step)
    # writer.add_scalar('loss/G_loss', G_loss, global_step)
    writer.add_scalar('loss/loss', loss, global_step)

    writer.add_scalar('metric/bit_acc', bit_acc, global_step)
    writer.add_scalar('metric/str_acc', str_acc, global_step)
    if global_step % 20 == 0:
        writer.add_image('input/image_input', image_input[0], global_step)
        writer.add_image('input/image_warped', input_warped[0], global_step)
        writer.add_image('encoded/encoded_warped', encoded_warped[0], global_step)
        writer.add_image('encoded/residual_warped', residual_warped[0] + 0.5, global_step)
        writer.add_image('encoded/encoded_image', encoded_image[0], global_step)
        writer.add_image('transformed/transformed_image', transformed_image[0], global_step)
        writer.add_image('transformed/test', test_transform[0], global_step)

    return loss, secret_loss, bit_acc, str_acc
