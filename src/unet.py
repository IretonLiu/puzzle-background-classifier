import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import crop
from torchvision.models import vgg16, VGG16_Weights
import torch
from tqdm import tqdm
import numpy as np
from utils import confusion_matrix, accuracy, f1_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DoubleConvolution(nn.Module):
    """
    perform two ReLU-activated 3x3 convolutions, where only the first will change the number of channels in the feature map
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x

    def load_conv_weights(self, vgg, index1, index2):
        # load vgg[index1] to the first conv layer AND vgg[index2] to the second conv layer
        with torch.no_grad():
            self.conv[0].weight.copy_(vgg.features[index1].weight)
            self.conv[0].bias.copy_(vgg.features[index1].bias)
            self.conv[2].weight.copy_(vgg.features[index2].weight)
            self.conv[2].bias.copy_(vgg.features[index2].bias)


class Down(nn.Module):
    """
    this is one of a repeated "unit" which performs a double convolution, and then downsamples (for the contracting path).

    max-pooling is done with 2x2 filter and stride 2 to reduce the dimensions of the feature map
    convolution will double the number of channels, and is done with a 3x3 filter
    """

    def __init__(self, in_channels, out_channels):
        assert in_channels * 2 == out_channels
        super().__init__()
        self.conv = nn.Sequential(
            nn.MaxPool2d(2, 2), DoubleConvolution(in_channels, out_channels)
        )

    def forward(self, x):
        # apply the downsampling
        x = self.conv(x)
        return x

    def load_conv_weights(self, vgg, index1, index2):
        # load the convolutional layer weights
        self.conv[1].load_conv_weights(vgg, index1, index2)


class Up(nn.Module):
    """
    this is one of a repeated "unit" which performs upsampling and then convolution (for the expansive path)

    upsample using transposed convolution with a 2x2 filter -> this also halves the number of channels
    perform a double convolution with 3x3 filters
    """

    def __init__(self, in_channels, out_channels):
        assert in_channels // 2 == out_channels
        super().__init__()

        # deconvolution with 2x2 filter and stride of 2 (is the green arrow)
        self.upsample = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=2, stride=2
        )

        # perform the double convolution
        self.double_convolution = DoubleConvolution(in_channels, out_channels)

    def forward(self, x, feature_map):
        # apply the upsampling, which will halve the number of channels so it matches the feature_map
        x = self.upsample(x)

        # crop the feature map so it has the same spatial dimensions as the current data
        # this is necessary because the convolutions are not padded, so the size decreases
        # get the size difference
        vert_dim = 2
        hori_dim = 3
        dim0_size_diff = feature_map.size()[vert_dim] - x.size()[vert_dim]
        dim1_size_diff = feature_map.size()[hori_dim] - x.size()[hori_dim]

        # crop the feature map
        feature_map = crop(
            feature_map,
            dim0_size_diff // 2,  # top left vertical component
            dim1_size_diff // 2,  # top left horizontal component
            x.size()[vert_dim],  # height of the cropped area
            x.size()[hori_dim],  # width of the cropped area
        )

        # concatenate with the feature map from the corresponding downsampling step
        x = torch.cat([x, feature_map], dim=1)

        # apply the double convolution
        x = self.double_convolution(x)

        return x


class UNet(nn.Module):
    """
    Pytorch implementation of UNet, following the same architecture as
    the on presented in the paper "U-Net: Convolutional Networks for Biomedical Image Segmentation".
    """

    def __init__(self, n_channels, learning_rate=0.001):
        super().__init__()

        self.n_channels = n_channels
        # Initial Transforms
        # is just a double convolution, no downsampling
        self.initial = DoubleConvolution(3, 64)

        # Contracting path
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)

        # Expansive path
        # the number in the name of the layer matches to the down units for the concat of the feature map
        self.up4 = Up(1024, 512)
        self.up3 = Up(512, 256)
        self.up2 = Up(256, 128)
        self.up1 = Up(128, 64)

        # final layer to reduce to a single dimension, which will be the classification
        # map to 2 channels since there are 2 classes
        self.final = nn.Conv2d(64, 1, kernel_size=1)

        #  Set up the optimizer
        self.to(device=device)
        self.optimiser = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def evaluate(self, val_data_loader, threshold=0.5):
        self.eval()

        val_loss = 0
        confusion_matrices = np.zeros((2, 2))
        with tqdm(total=len(val_data_loader), desc=f"Validating", unit="img") as pbar:
            # evaluation - no need to track gradients
            with torch.no_grad():
                for batch in val_data_loader:
                    print(batch)
                    images = batch[0]
                    true_masks = batch[1]

                    assert images.shape[1] == self.n_channels, (
                        f"Network has been defined with {self.n_channels} input channels, "
                        f"but loaded images have {images.shape[1]} channels. Please check that "
                        "the images are loaded correctly."
                    )

                    images = images.to(device=device, dtype=torch.float32)
                    true_masks = true_masks.to(device=device, dtype=torch.float32)

                    pred_masks = self(images)

                    vert_dim = 2
                    hori_dim = 3
                    dim0_size_diff = (
                        true_masks.size()[vert_dim] - pred_masks.size()[vert_dim]
                    )
                    dim1_size_diff = (
                        true_masks.size()[hori_dim] - pred_masks.size()[hori_dim]
                    )

                    # crop the true masks
                    if dim0_size_diff != 0 and dim1_size_diff != 0:
                        true_masks = crop(
                            true_masks,
                            dim0_size_diff // 2,  # top left vertical component
                            dim1_size_diff // 2,  # top left horizontal component
                            pred_masks.size()[vert_dim],  # height of the cropped area
                            pred_masks.size()[hori_dim],  # width of the cropped area
                        )

                    loss = F.mse_loss(pred_masks, true_masks)

                    pbar.update(images.shape[0])
                    val_loss += loss.item()
                    pbar.set_postfix(**{"loss (batch)": loss.item()})

                    # loop through the images and get their confusion matrices
                    pred_masks = pred_masks.to("cpu")
                    true_masks = true_masks.to("cpu")
                    for pred, true in zip(pred_masks, true_masks):
                        confusion_matrices += confusion_matrix(pred, true, threshold)

        self.train()
        print(confusion_matrices)
        print(confusion_matrices / np.sum(confusion_matrices))
        return val_loss, confusion_matrices / np.sum(confusion_matrices)

    def train_model(
        self,
        train_data_loader,
        val_data_loader,
        save_path,
        max_epoch=10,
        current_epoch=0,
        threshold=0.5,
    ):
        self.train()
        for i in range(current_epoch, max_epoch):
            print("Starting epoch {}".format(i))
            epoch_loss = 0
            # https://github.com/milesial/Pytorch-UNet/blob/master/train.py
            # with tqdm(
            #     total=len(train_data_loader), desc=f"Epoch {i}/{max_epoch}", unit="img"
            # ) as pbar:
            #     for batch in train_data_loader:
            #         images = batch[0]
            #         true_masks = batch[1]

            #         assert images.shape[1] == self.n_channels, (
            #             f"Network has been defined with {self.n_channels} input channels, "
            #             f"but loaded images have {images.shape[1]} channels. Please check that "
            #             "the images are loaded correctly."
            #         )

            #         images = images.to(device=device, dtype=torch.float32)
            #         true_masks = true_masks.to(device=device, dtype=torch.float32)

            #         pred_masks = self(images)

            #         vert_dim = 2
            #         hori_dim = 3
            #         dim0_size_diff = (
            #             true_masks.size()[vert_dim] - pred_masks.size()[vert_dim]
            #         )
            #         dim1_size_diff = (
            #             true_masks.size()[hori_dim] - pred_masks.size()[hori_dim]
            #         )

            #         # crop the true masks
            #         if dim0_size_diff != 0 and dim1_size_diff != 0:
            #             true_masks = crop(
            #                 true_masks,
            #                 dim0_size_diff // 2,  # top left vertical component
            #                 dim1_size_diff // 2,  # top left horizontal component
            #                 pred_masks.size()[vert_dim],  # height of the cropped area
            #                 pred_masks.size()[hori_dim],  # width of the cropped area
            #             )

            #         loss = F.mse_loss(pred_masks, true_masks)

            #         self.optimiser.zero_grad()
            #         loss.backward()
            #         self.optimiser.step()

            #         pbar.update(images.shape[0])
            #         epoch_loss += loss.item()
            #         pbar.set_postfix(**{"loss (batch)": loss.item()})

            #         print("", end="", flush=True)

            # # save model and to evaluation on validation data
            # path = f"{save_path}/epoch_{i}.pt"
            # self.save_model(path, i)
            val_loss, confusion_matrix = self.evaluate(val_data_loader, threshold)
            print(f"Epoch {i}:")
            print(f"\ttraining loss = {epoch_loss}")
            print(f"\tvalidation loss = {val_loss}")
            print(f"\tvalidation accuracy = {accuracy(confusion_matrix)}")
            print(f"\tvalidation F1 Score = {f1_score(confusion_matrix)}")
            print("", end="", flush=True)

    def forward(self, x):
        # pass the image through the unet

        # initial
        down1 = self.initial(x)

        # Contracting path
        # names are the end of that unit
        down2 = self.down1(down1)
        down3 = self.down2(down2)
        x = self.down3(down3)
        # middle = self.down4(down4)

        # Expansive path
        # the number in the name of the layer matches to the down units for the concat of the feature map
        # x = self.up4(middle, down4)
        x = self.up3(x, down3)
        x = self.up2(x, down2)
        x = self.up1(x, down1)

        # final layer to get the desired number of classes -> 2 classes represented by a single value
        x = self.final(x)

        # sigmoid activation to map to (0, 1) -> use 0.5 as a threshold
        return torch.sigmoid(x)

    def load_vgg_weights(self):
        # load the pretrained weights for the relevant layers
        """
        all convs are 3x3 filters, except the last conv
        our layers are:
            initial
                conv: 3-64 and 64-64
            contractive path
                maxpool
                conv: 64-128 and 128-128
                maxpool
                conv: 128-256 and 256-256
                maxpool
                conv: 256-512 and 512-512
                maxpool
                conv: 512-1024 and 1024-1024
            expansive path
                up: 1024-512
                conv: 512-512 and 512-512
                up: 512-256
                conv: 256-256 and 256-256
                up: 256-128
                conv: 128-128 and 128-128
                up: 128-64
                conv: 64-64 and 64-64
            final
                conv: 64-2
        """

        print("Loading VGG16 weights...")
        vgg = vgg16(weights=VGG16_Weights.DEFAULT)

        # we can only replace the encoder part of the network with VGG weights
        # initial
        self.initial.load_conv_weights(vgg, 0, 2)
        # down units
        self.down1.load_conv_weights(vgg, 5, 7)
        self.down2.load_conv_weights(vgg, 10, 12)
        self.down3.load_conv_weights(vgg, 17, 19)

        print("Finished loading")

    def save_model(self, path, epoch):
        """
        save the model to the specified dir
        """

        # print("saving model to", path)
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.state_dict(),
                "optimiser_state_dict": self.optimiser.state_dict(),
            },
            path,
        )

        print("model saved to", path)

    def load_model(self, path):
        """
        load the policy network from the specified dir
        """

        print("loading model from", path)
        saved = torch.load(path)
        self.load_state_dict(saved["model_state_dict"])
        self.to(device=device)
        self.optimiser.load_state_dict(saved["optimiser_state_dict"])
        print("model loaded from", path)

        return saved["epoch"]


"""
VGG structture:

VGG(
  (features): Sequential(
    (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): ReLU(inplace=True)
    (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (3): ReLU(inplace=True)
    (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (6): ReLU(inplace=True)
    (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (8): ReLU(inplace=True)
    (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (11): ReLU(inplace=True)
    (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (13): ReLU(inplace=True)
    (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (15): ReLU(inplace=True)
    (16): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (17): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (18): ReLU(inplace=True)
    (19): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (20): ReLU(inplace=True)
    (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (22): ReLU(inplace=True)
    (23): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (24): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (25): ReLU(inplace=True)
    (26): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (27): ReLU(inplace=True)
    (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (29): ReLU(inplace=True)
    (30): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (avgpool): AdaptiveAvgPool2d(output_size=(7, 7))
  (classifier): Sequential(
    (0): Linear(in_features=25088, out_features=4096, bias=True)
    (1): ReLU(inplace=True)
    (2): Dropout(p=0.5, inplace=False)
    (3): Linear(in_features=4096, out_features=4096, bias=True)
    (4): ReLU(inplace=True)
    (5): Dropout(p=0.5, inplace=False)
    (6): Linear(in_features=4096, out_features=1000, bias=True)
  )
)
"""
