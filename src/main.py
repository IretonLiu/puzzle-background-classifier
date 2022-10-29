import glob
import cv2
import time
import os
import glob
from natsort import natsorted
from tqdm import tqdm
from copy import deepcopy

# from matplotlib.pyplot import scatter
import matplotlib.pyplot as plt
import numpy as np
from sklearn import cluster
from sklearn.metrics import accuracy_score
from gmm import GaussianMixtureModel
from scipy.stats import multivariate_normal
from ellipsoid import get_cov_ellipsoid
from sklearn.mixture import GaussianMixture
from classifier import Classifier
from unet import UNet

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms as T

res = (1024, 768)


def read_data():
    # glob all images from data
    image_files = glob.glob("data/images-1024x768/*.png")
    mask_files = glob.glob("data/masks-1024x768/*.png")

    # convert to images using cv2 and convert to float RGB
    images = np.array(
        [cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB) for image in image_files]
    )
    masks = np.array(
        [cv2.cvtColor(cv2.imread(mask), cv2.COLOR_BGR2GRAY) > 0 for mask in mask_files]
    )

    return images, masks


def split_dataset(images, masks):
    # split the images and masks into training validation and test with ration 70:15:15
    training_count = int(round(len(images) * 0.7))
    validation_count = int(round(len(images) * 0.15))
    testing_count = int(round(len(images) * 0.15))

    zipped_images = list(zip(images, masks))
    # Split images into train, test and validation
    train_set, val_set, test_set = random_split(
        zipped_images,
        [training_count, validation_count, testing_count],
        generator=torch.Generator().manual_seed(0),
    )

    return train_set, val_set, test_set


def random_affine(degrees, translate):
    # get the affine transformation matrix

    degree = np.random.uniform(degrees[0], degrees[1])


def augment_images(images, masks, n=10):

    # convert to tensors and rearrange to channels first
    image_tensors = torch.from_numpy(images).permute(0, 3, 1, 2)
    mask_tensors = torch.from_numpy(masks).unsqueeze(1)

    if n <= 0:
        return image_tensors, mask_tensors

    augmented_images = []
    augmented_masks = []
    with tqdm(total=len(images) * n, desc=f"Transforming Images", unit="img") as pbar:
        for image_tensor, mask_tensor in zip(image_tensors, mask_tensors):

            for i in range(n):
                # get the affine transformation parameters
                params = T.RandomAffine.get_params(
                    degrees=(-180, 180),
                    translate=(0.3, 0.3),
                    scale_ranges=None,
                    shears=None,
                    img_size=image_tensor.size(),
                )

                # apply the affine transformation
                aug_image = T.functional.affine(image_tensor, *params)
                aug_mask = T.functional.affine(mask_tensor, *params)

                augmented_images.append(aug_image)
                augmented_masks.append(aug_mask)

                pbar.update()
                print("", end="", flush=True)

    # add the original images
    augmented_images = torch.cat((image_tensors, torch.stack(augmented_images)), dim=0)
    augmented_masks = torch.cat((mask_tensors, torch.stack(augmented_masks)), dim=0)

    return augmented_images, augmented_masks


# def scatter_plot(data, step, gmm=None):
#     # 3d scatter plot of train_data_foreground
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection="3d")
#     ax.scatter(data[::step, 0], data[::step, 1], data[::step, 2])

#     ax.set_xlabel("X Label")
#     ax.set_ylabel("Y Label")
#     ax.set_zlabel("Z Label")
#     if gmm:
#         for j in range(gmm.n_components):
#             # plot the gaussian
#             x, y, z = get_cov_ellipsoid(gmm.covariances_[j], gmm.means_[j])
#             ax.plot_wireframe(x, y, z, color="r", alpha=0.1)
#     plt.show()


def to_feature_vector(images, feature_type):

    image_dim = images[0].shape
    # convert images to feature vectors
    float_images = images.astype(np.float32) / 255.0
    if feature_type == "rgb":
        # flatten the images
        return float_images.reshape(-1, 3)
    elif feature_type == "hsv":
        # convert float images to hsv
        hsv = np.array(
            [cv2.cvtColor(image, cv2.COLOR_RGB2HSV) for image in images]
        ).reshape(-1, 3)
        # normalize the hsv values
        return hsv.astype(np.float32) / np.array([179, 255, 255])

    elif feature_type == "rgb+dog":
        dog = np.array(
            [
                cv2.GaussianBlur(image, (3, 3), 0) - cv2.GaussianBlur(image, (5, 5), 0)
                for image in float_images
            ]
        )
        dog = np.array([(d - np.min(d)) / (np.max(d) - np.min(d)) for d in dog])
        return np.hstack((float_images.reshape(-1, 3), dog.reshape(-1, 3)))
    elif feature_type == "hsv+xy":
        # convert float images to hsv
        hsv = np.array(
            [cv2.cvtColor(image, cv2.COLOR_RGB2HSV) for image in images]
        ).reshape(-1, 3)
        # a np array of the x and y coordinates
        xy = np.array(
            [np.indices(image_dim[:2]).transpose((1, 2, 0)) for image in images]
        ).reshape(-1, 2)
        # normalize the hsv values
        return np.hstack(
            (
                hsv.astype(np.float32) / np.array([179, 255, 255]),
                xy.astype(np.float32) / np.array(image_dim[:2]),
            )
        )

    else:
        raise ValueError("Unknown feature type")


def run_gmm():

    images, masks = read_data()
    train_set, val_set, test_set = split_dataset(images, masks)

    # convert to pixels
    # train_data = np.concatenate([image.reshape(-1, 3)
    #                              for image in train_images])

    # list of feature sets to use
    # features = ["rgb", "rgb+dog", "hsv", "hsv+xy"]
    features = ["rgb+dog"]

    foreground_h_list = [3, 4, 5, 6, 7, 8, 9]
    background_h_list = [2, 3, 4, 5, 6]
    train_images, train_masks = zip(*train_set)
    train_images = np.array(train_images)
    train_masks = np.array(train_masks)
    if 1:

        # fit the model
        for feature in features:
            print("Training with feature: {}".format(feature))
            # convert to feature vectors

            train_data = to_feature_vector(train_images, feature)
            train_data_masks = train_masks.reshape(-1, 1)

            train_data_foreground = train_data[train_data_masks[:, 0]]
            train_data_background = train_data[~train_data_masks[:, 0]]

            for background_h in background_h_list:
                print("Training background GMM with h = {}".format(background_h))
                start_time = time.time()
                gmm_background = GaussianMixtureModel(
                    background_h, train_data_background.shape[1], max_iter=500, seed=4
                )
                # try:
                gmm_background.fit(train_data_background)
                # except:
                #     print(
                #         "Failed to fit background GMM with h = {}".format(
                #             background_h)
                #     )
                #     continue
                print("Training time:", (time.time() - start_time))
                gmm_background.save_model(
                    f"models/gmm/{feature}/background/{background_h}/"
                )
                print(flush=True, end="")

            for foreground_h in foreground_h_list:
                print("Training foreground GMM with h = {}".format(foreground_h))
                start_time = time.time()
                gmm_foreground = GaussianMixtureModel(
                    foreground_h, train_data_foreground.shape[1], max_iter=500, seed=2
                )
                try:
                    gmm_foreground.fit(train_data_foreground)
                except:
                    print("Failed to fit foreground GMM with h = {}".format())
                    continue
                print("Training time:", (time.time() - start_time))
                gmm_foreground.save_model(
                    f"models/gmm/{feature}/foreground/{foreground_h}/"
                )
                print(flush=True, end="")
            print("=============================================================")

    # features = ["rgb"]
    # foreground_h_list = [6, 7, 8, 9]
    # background_h_list = [5, 6]

    if 0:
        best_feature = "None"
        best_background_h = -1
        best_foreground_h = -1
        max_accuracy = 0

        for feature in features:
            train_data = to_feature_vector(train_images, feature)
            train_data_masks = train_masks.reshape(-1, 1)
            train_data_foreground = train_data[train_data_masks[:, 0]]
            classifier = Classifier(train_data, train_data_masks)

            for foreground_h in foreground_h_list:

                gmm_foreground = GaussianMixtureModel(
                    foreground_h, train_data_foreground.shape[1], max_iter=500, seed=2
                )
                gmm_foreground.load_model(
                    f"models/gmm/{feature}/foreground/{foreground_h}/"
                )

                for background_h in background_h_list:

                    gmm_background = GaussianMixtureModel(
                        background_h,
                        train_data_background.shape[1],
                        max_iter=500,
                        seed=4,
                    )

                    gmm_background.load_model(
                        f"models/gmm/{feature}/background/{background_h}/"
                    )

                    # test the model

                    # likelihoods = [gmm_background, gmm_foreground]
                    # protrain_imagesbabilities = classifier.maximum_a_posteriori(
                    #     likelihoods)
                    # predictions = np.argmax(probabilities, axis=0) == 1

                    # accuracy_score = np.sum(
                    #     predictions == train_data_masks[:, 0]) / len(train_data_masks)

                    # print(
                    #     f"Training accuracy for {feature} with {foreground_h} foreground and {background_h} background: {accuracy_score}")
                    # # updated the best model
                    # if accuracy_score > max_accuracy:
                    #     max_accuracy = accuracy_score
                    #     best_feature = feature
                    #     best_foreground_h = foreground_h
                    #     best_background_h = background_h

                    #     print(
                    #         f"Current best model: {best_feature} with {best_foreground_h} foreground and {best_background_h} background: {max_accuracy}")


def unet_get_checkpoint(path):
    # get the name of the latest model
    dirs = glob.glob(f"{path}/*.pt")
    if len(dirs) == 0:
        return None

    return natsorted(dirs)[-1]


def augmentation_wrapper(data, n):
    # separate the current dataset
    imgs = []
    msks = []
    for img, msk in data:
        imgs.append(img)
        msks.append(msk)

    imgs = np.array(imgs)
    msks = np.array(msks)

    # do the augmentation
    imgs, msks = augment_images(imgs, msks, n=n)

    # repackage into a list of 2-tuples
    output = []
    for img, msk in zip(imgs, msks):
        output.append((img, msk))

    return output


def run_unet(save_path, train_set, val_set, test_set, parameters):
    train_set = augmentation_wrapper(train_set, n=parameters["augmentation_size"])
    val_set = augmentation_wrapper(val_set, n=0)
    test_set = augmentation_wrapper(test_set, n=0)
    print("Loaded data", flush=True)

    batch_size = parameters["batch_size"]

    loader_args = dict(
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    train_loader = DataLoader(
        train_set, shuffle=True, batch_size=batch_size, **loader_args
    )
    val_loader = DataLoader(val_set, shuffle=False, batch_size=1, **loader_args)
    unet = UNet(n_channels=3, learning_rate=parameters["lr"])

    # check if the current save path has a checkpoint, if so, load from it
    checkpoint = unet_get_checkpoint(save_path)
    current_epoch = 0
    if checkpoint:
        current_epoch = unet.load_model(checkpoint) + 1
    else:
        unet.load_vgg_weights()

    (
        training_loss,
        validation_loss,
        validation_accuracy,
        validation_f1_score,
    ) = unet.train_model(
        train_loader,
        val_loader,
        save_path,
        max_epoch=15,
        current_epoch=current_epoch,
        threshold=0.5,
    )

    return training_loss, validation_loss, validation_accuracy, validation_f1_score

    # prediction = unet.predict(train_set[0][0])
    # fig, ax = plt.subplots(rows=2, cols=2, figsize=(10, 5))
    # ax = ax.ravel()
    # ax[0].imshow(prediction, cmap="gray")
    # ax[1].imshow(train_set[0][1], cmap="gray")
    # fig.savefig(f"{save_path}/sample_imge.png")


if __name__ == "__main__":
    # run_gmm()
    # run_unet("./models/unet/one", {"lr": 1e-3, "batch_size": 1, "augmentation_size": 0})
    # exit()

    # UNet ------------------------------------------
    # read the images
    images, masks = read_data()
    images = images.astype(np.float32) / 255.0

    train_set, val_set, test_set = split_dataset(images, masks)

    # hyperparameter lists
    learning_rates = [1e-3, 1e-4]
    batch_sizes = [1, 8, 16]
    augmentation_sizes = [0, 5, 10]
    # thresholds = [0.5, 0.75, 0.8, 0.9]

    # best params
    best_validation_accuracy = -float("inf")
    best_accuracy_params = None
    best_validation_f1score = -float("inf")
    best_f1score_params = None

    # do grid search
    for lr in learning_rates:
        # graph a plot per configuration
        fig, ax = plt.subplots(
            nrows=len(augmentation_sizes), ncols=len(batch_sizes), figsize=(20, 20)
        )
        fig.suptitle(f"Plots for a learning rate of {lr}\n(Batch Size changes per row; Augmentation Size changes per column)")

        for i, bs in enumerate(batch_sizes):
            for j, aug_size in enumerate(augmentation_sizes):
                # store the parameters
                params = {"lr": lr, "batch_size": bs, "augmentation_size": aug_size}

                # define the save location and create the folders as required
                save_path = f"./models/unet/params/{lr}_{bs}_{aug_size}"
                os.makedirs(save_path, exist_ok=True)

                # test this configuration
                (
                    training_loss,
                    validation_loss,
                    validation_accuracy,
                    validation_f1_score,
                ) = run_unet(save_path, deepcopy(train_set), deepcopy(val_set), deepcopy(test_set), params)

                # record the best params
                if validation_accuracy > best_validation_accuracy:
                    best_validation_accuracy = validation_accuracy
                    best_accuracy_params = params

                if validation_f1_score > best_validation_f1score:
                    best_validation_f1score = validation_f1_score
                    best_f1score_params = params

                # plot
                ax[i, j].set_title(f"Batch size = {bs}; Augmentation_size = {aug_size}")
                ax[i, j].plot(training_loss, label="Training Loss")
                ax[i, j].plot(validation_loss, label="Validation Loss")
                ax[i, j].plot(validation_accuracy, label="Validation Accuracy")
                ax[i, j].plot(validation_f1_score, label="Validation F1 Score")

        # save the figure
        fig.legend(loc="best")
        fig.savefig(f"hyperparameter_tuning_lr_{lr}.png", format="png")

    print("The best parameters are:")
    print(
        f"Validation Accuracy {best_validation_accuracy} with params {best_accuracy_params}"
    )
    print(
        f"Validation F1 Score {best_validation_f1score} with params {best_f1score_params}"
    )

"""
UNet hyperparameters to tune:
- learning rate
- batch size
- amount of augmentation
- threshold
"""
