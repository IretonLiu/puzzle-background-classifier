import glob
import cv2
import time
import os
import glob
import gc
from natsort import natsorted
from tqdm import tqdm
from copy import deepcopy

from matplotlib.pyplot import scatter
import matplotlib.pyplot as plt
import numpy as np
from gmm import GaussianMixtureModel
from scipy.stats import multivariate_normal
from ellipsoid import get_cov_ellipsoid
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold
from classifier import Classifier
from unet import UNet

import torch
from torchvision.transforms.functional import crop
from torch.utils.data import DataLoader, random_split
from torchvision import transforms as T

from utils import (
    confusion_matrix,
    accuracy,
    false_positive_rate,
    precision,
    recall,
    f1_score,
)

res = (1024, 768)


def read_data():
    # glob all images from data
    image_files = natsorted(glob.glob("data/images-1024x768/*.png"))
    mask_files = natsorted(glob.glob("data/masks-1024x768/*.png"))

    # convert to images using cv2 and convert to float RGB
    images = np.array(
        [cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB)
         for image in image_files]
    )
    masks = np.array(
        [cv2.cvtColor(cv2.imread(mask), cv2.COLOR_BGR2GRAY)
         > 0 for mask in mask_files]
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
    augmented_images = torch.cat(
        (image_tensors, torch.stack(augmented_images)), dim=0)
    augmented_masks = torch.cat(
        (mask_tensors, torch.stack(augmented_masks)), dim=0)

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
                cv2.GaussianBlur(image, (3, 3), 0) -
                cv2.GaussianBlur(image, (5, 5), 0)
                for image in float_images
            ]
        )
        dog = np.array([(d - np.min(d)) / (np.max(d) - np.min(d))
                       for d in dog])
        return np.hstack((float_images.reshape(-1, 3), dog.reshape(-1, 3)))
    elif feature_type == "hsv+xy":
        # convert float images to hsv
        hsv = np.array(
            [cv2.cvtColor(image, cv2.COLOR_RGB2HSV) for image in images]
        ).reshape(-1, 3)
        # a np array of the x and y coordinates
        xy = np.array(
            [np.indices(image_dim[:2]).transpose((1, 2, 0))
             for image in images]
        ).reshape(-1, 2)
        # normalize the hsv values
        return np.hstack(
            (
                hsv.astype(np.float32) / np.array([179, 255, 255]),
                xy.astype(np.float32) / np.array(image_dim[:2]),
            )
        )
    elif feature_type == "all":
        hsv = np.array(
            [cv2.cvtColor(image, cv2.COLOR_RGB2HSV) for image in images]
        ).reshape(-1, 3)
        # a np array of the x and y coordinates
        xy = np.array(
            [np.indices(image_dim[:2]).transpose((1, 2, 0))
             for image in images]
        ).reshape(-1, 2)
        dog = np.array(
            [
                cv2.GaussianBlur(image, (3, 3), 0) -
                cv2.GaussianBlur(image, (5, 5), 0)
                for image in float_images
            ]
        )
        dog = np.array([(d - np.min(d)) / (np.max(d) - np.min(d))
                       for d in dog])

        # normalize the hsv values
        return np.hstack(
            (
                float_images.reshape(-1, 3),
                hsv.astype(np.float32) / np.array([179, 255, 255]),
                xy.astype(np.float32) / np.array(image_dim[:2]),
                dog.reshape(-1, 3),
            )
        )
    else:
        raise ValueError("Unknown feature type")


def post_processing(predictions, kernel_dims=5):
    """Apply morhpolical transformations of closing
    and opening"""
    kernel = np.ones((kernel_dims, kernel_dims), np.uint8)
    for i in range(len(predictions)):

        predictions[i] = cv2.morphologyEx(
            predictions[i], cv2.MORPH_OPEN, kernel)
        predictions[i] = cv2.morphologyEx(
            predictions[i], cv2.MORPH_CLOSE, kernel)

    return predictions.reshape(-1, 1)


def train_gmm(
    train_set,
    features,
    background_h_list,
    foreground_h_list,
):
    train_images, train_masks = zip(*train_set)
    train_images = np.array(train_images)
    train_masks = np.array(train_masks)

    # fit the model
    for feature in features:
        print("Training with feature: {}".format(feature))
        # convert to feature vectors

        train_data = to_feature_vector(train_images, feature)
        train_data_masks = train_masks.reshape(-1, 1)

        train_data_foreground = train_data[train_data_masks[:, 0]]
        train_data_background = train_data[~train_data_masks[:, 0]]

        # train background gmm
        for background_h in background_h_list:
            print("Training background GMM with h = {}".format(background_h))
            start_time = time.time()
            gmm_background = GaussianMixtureModel(
                background_h, train_data_background.shape[1], max_iter=500, seed=4
            )
            gmm_background.fit(train_data_background)
            print("Training time:", (time.time() - start_time))
            gmm_background.save_model(
                f"models/gmm/{feature}/background/{background_h}/"
            )
            print(flush=True, end="")

        # train foreground gmm
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


def validate_gmm(val_set, features, background_h_list, foreground_h_list):
    best_feature = "None"
    best_background_h = -1
    best_foreground_h = -1
    max_accuracy = 0
    val_images, val_masks = zip(*val_set)
    val_images = np.array(val_images)
    val_masks = np.array(val_masks)

    # loop through the features
    for feature in features:
        val_data = to_feature_vector(val_images, feature)

        val_data_masks = val_masks.reshape(-1, 1)

        val_data_foreground = val_data[val_data_masks[:, 0]]
        val_data_background = val_data[~val_data_masks[:, 0]]

        classifier = Classifier(val_data, val_data_masks)

        # loop through all foreground and background h values
        for foreground_h in foreground_h_list:

            gmm_foreground = GaussianMixtureModel(
                foreground_h, val_data_foreground.shape[1], max_iter=500, seed=2
            )
            gmm_foreground.load_model(
                f"models/gmm/{feature}/foreground/{foreground_h}/"
            )

            for background_h in background_h_list:

                gmm_background = GaussianMixtureModel(
                    background_h,
                    val_data_background.shape[1],
                    max_iter=500,
                    seed=4,
                )

                gmm_background.load_model(
                    f"models/gmm/{feature}/background/{background_h}/"
                )

                # predict the masks
                likelihoods = [gmm_background, gmm_foreground]
                probabilities = classifier.maximum_a_posteriori(likelihoods)
                predictions = np.argmax(probabilities, axis=0) == 1
                predictions = post_processing(
                    predictions.reshape(val_masks.shape).astype(np.float32)
                )

                accuracy_score = np.sum(predictions == val_data_masks) / len(
                    val_data_masks
                )

                print(
                    f"validation accuracy for {feature} with {foreground_h} foreground and {background_h} background: {accuracy_score}"
                )
                # updated the best model
                if accuracy_score > max_accuracy:
                    max_accuracy = accuracy_score
                    best_feature = feature
                    best_foreground_h = foreground_h
                    best_background_h = background_h

                    print(
                        f"Current best model: {best_feature} with {best_foreground_h} foreground and {best_background_h} background: {max_accuracy}"
                    )
    print(
        "Best model: {} with {} foreground and {} background: {}".format(
            best_feature, best_foreground_h, best_background_h, max_accuracy
        )
    )
    return best_feature, best_foreground_h, best_background_h


def test_gmm(test_set, best_feature, best_foreground_h, best_background_h):
    # format the test set
    test_images, test_masks = zip(*test_set)
    test_images = np.array(test_images)
    test_masks = np.array(test_masks)

    test_data = to_feature_vector(test_images, best_feature)
    test_data_masks = test_masks.reshape(-1, 1)

    test_data_foreground = test_data[test_data_masks[:, 0]]
    test_data_background = test_data[~test_data_masks[:, 0]]

    classifier = Classifier(test_data, test_data_masks)

    gmm_foreground = GaussianMixtureModel(
        best_foreground_h, test_data_foreground.shape[1], max_iter=500, seed=2
    )
    gmm_foreground.load_model(
        f"models/gmm/{best_feature}/foreground/{best_foreground_h}/"
    )

    gmm_background = GaussianMixtureModel(
        best_background_h,
        test_data_background.shape[1],
        max_iter=500,
        seed=4,
    )

    gmm_background.load_model(
        f"models/gmm/{best_feature}/background/{best_background_h}/"
    )

    # predict the test set
    likelihoods = [gmm_background, gmm_foreground]
    probabilities = classifier.maximum_a_posteriori(likelihoods)
    predictions = np.argmax(probabilities, axis=0) == 1
    predictions = post_processing(
        predictions.reshape(test_masks.shape).astype(np.float32)
    )

    predictions = predictions.reshape((len(test_masks), -1)).astype(np.float32)
    test_masks = test_masks.reshape((len(test_masks), -1))

    # calculate the accuracy image by image
    for prediction, mask in zip(predictions, test_masks):

        c_matrix = confusion_matrix(mask, prediction)
        accuracy_score = accuracy(c_matrix)
        precision_score = precision(c_matrix)
        recall_score = recall(c_matrix)
        print("Confusion matrix:")
        print(c_matrix)
        print("Accuracy score:", accuracy_score)
        print("Precision score:", precision_score)
        print("Recall score:", recall_score)


def gmm_cross_validation(images, masks, best_feature, best_foreground_h, best_background_h):
    # performing 6-fold cross validation
    kf = KFold(n_splits=6)
    average_accuracy = 0
    average_precision = 0
    average_recall = 0
    average_f1 = 0
    for train_index, test_index in kf.split(images):
        # split the data into train and test
        train_images, test_images = images[train_index], images[test_index]
        train_masks, test_masks = masks[train_index], masks[test_index]
        train_set = list(zip(train_images, train_masks))
        test_set = list(zip(test_images, test_masks))

        train_data = to_feature_vector(train_images, best_feature)
        train_data_masks = train_masks.reshape(-1, 1)

        # train
        print("Training GMM...")
        train_data_foreground = train_data[train_data_masks[:, 0]]
        train_data_background = train_data[~train_data_masks[:, 0]]
        gmm_background = GaussianMixture(
            n_components=best_background_h).fit(train_data_background)
        gmm_foreground = GaussianMixture(
            n_components=best_foreground_h).fit(train_data_foreground)

        # test
        print("Testing GMM...")
        test_images, test_masks = zip(*test_set)
        test_images = np.array(test_images)
        test_masks = np.array(test_masks)

        test_data = to_feature_vector(test_images, best_feature)
        test_data_masks = test_masks.reshape(-1, 1)
        classifier = Classifier(test_data, test_data_masks)

        likelihoods = [gmm_background, gmm_foreground]
        probabilities = classifier.maximum_a_posteriori(likelihoods)
        predictions = np.argmax(probabilities, axis=0) == 1
        predictions = post_processing(
            predictions.reshape(test_masks.shape).astype(np.float32)
        )

        # calculate metrics
        c_matrix = confusion_matrix(test_data_masks, predictions)
        accuracy_score = accuracy(c_matrix)
        precision_score = precision(c_matrix)
        recall_score = recall(c_matrix)
        f1 = f1_score(c_matrix)
        print("Confusion matrix:")
        print(c_matrix)
        print("Accuracy score:", accuracy_score)
        print("Precision score:", precision_score)
        print("Recall score:", recall_score)
        print("F1 score:", f1)
        print("=====================")
        average_accuracy += accuracy_score
        average_precision += precision_score
        average_recall += recall_score
        average_f1 += f1

    print("Average accuracy:", average_accuracy / 6)
    print("Average precision:", average_precision / 6)
    print("Average recall:", average_recall / 6)
    print("Average F1:", average_f1 / 6)


def execute_gmm():

    images, masks = read_data()
    train_set, val_set, test_set = split_dataset(images, masks)

    # hyperparameters
    # list of feature sets to use
    features = ["rgb", "rgb+dog", "hsv", "hsv+xy", "all"]

    foreground_h_list = [2, 3, 4, 5, 6, 7, 8]
    background_h_list = [2, 3, 4]
    train_gmm(train_set, features, background_h_list, foreground_h_list)

    # find optimal hyperparameters
    best_feature, best_foreground_h, best_background_h = validate_gmm(
        val_set, features, background_h_list, foreground_h_list)

    # test the model
    test_gmm(test_set, best_feature, best_foreground_h, best_background_h)
    gmm_cross_validation(images[:6], masks[:6], best_feature,
                         best_foreground_h, best_background_h)


def unet_get_checkpoint(path, number):
    # number is the epoch to load
    # get the name of the latest model
    dirs = glob.glob(f"{path}/*.pt")
    if len(dirs) == 0:
        return None

    return natsorted(dirs)[number]


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


def run_unet(save_path, train_set, val_set, test_set, parameters, validate_only=False):
    # train_set = augmentation_wrapper(train_set, n=parameters["augmentation_size"])
    # val_set = augmentation_wrapper(val_set, n=0)
    # test_set = augmentation_wrapper(test_set, n=0)
    # print("Loaded data", flush=True)

    batch_size = 1

    loader_args = dict(
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    train_loader = DataLoader(
        train_set, shuffle=True, batch_size=batch_size, **loader_args
    )
    val_loader = DataLoader(val_set, shuffle=False,
                            batch_size=1, **loader_args)

    unet = UNet(n_channels=3, learning_rate=parameters["lr"])

    # check if the current save path has a checkpoint, if so, load from it
    checkpoint = unet_get_checkpoint(
        save_path, parameters["epoch"] if "epoch" in parameters else -1
    )
    current_epoch = 0
    if checkpoint:
        current_epoch = unet.load_model(checkpoint) + 1
    else:
        unet.load_vgg_weights()

    if not validate_only:
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
            threshold=parameters["threshold"] if "threshold" in parameters else 0.5,
        )

        return training_loss, validation_loss, validation_accuracy, validation_f1_score

    else:
        # only do validation
        loss, matrix = unet.evaluate(
            val_loader, threshold=parameters["threshold"])

        return matrix


def do_unet_hyperparameter_search(run_name, train_set_, val_set_, test_set_):
    # hyperparameter lists
    learning_rates = [1e-3, 5e-4, 1e-4]
    augmentation_sizes = [0, 5, 10]
    # thresholds = [0.5, 0.75, 0.8, 0.9]

    # best params
    # (epoch, value)
    best_validation_accuracy = (0, -float("inf"))
    best_accuracy_params = None
    best_validation_f1score = (0, -float("inf"))
    best_f1score_params = None

    # graph a plot per configuration
    fig, ax = plt.subplots(
        nrows=len(augmentation_sizes), ncols=len(learning_rates), figsize=(20, 20)
    )
    fig.suptitle(
        f"Plots of training and validation performance\n(Augmentation Size changes per row; Learning Rate changes per column)"
    )

    # do grid search
    for i, aug_size in enumerate(augmentation_sizes):
        train_set = augmentation_wrapper(train_set_, n=aug_size)
        val_set = augmentation_wrapper(val_set_, n=0)
        test_set = augmentation_wrapper(test_set_, n=0)
        print(f"Loaded data for augmentation size {aug_size}", flush=True)

        for j, lr in enumerate(learning_rates):
            # store the parameters
            params = {"lr": lr, "augmentation_size": aug_size}

            print(f"\nTraining with {params}")

            # define the save location and create the folders as required
            save_path = f"./models/unet/{run_name}/{lr}_{aug_size}"
            os.makedirs(save_path, exist_ok=True)

            # test this configuration
            (
                training_loss,
                validation_loss,
                validation_accuracy,
                validation_f1_score,
            ) = run_unet(
                save_path,
                deepcopy(train_set),
                deepcopy(val_set),
                deepcopy(test_set),
                params,
            )

            # record the best params
            for k in range(len(validation_accuracy)):
                if validation_accuracy[k] > best_validation_accuracy[1]:
                    best_validation_accuracy = (k, validation_accuracy[k])
                    best_accuracy_params = params

                if validation_f1_score[k] > best_validation_f1score[1]:
                    best_validation_f1score = (k, validation_f1_score[k])
                    best_f1score_params = params

            # plot
            ax[i, j].set_title(
                f"Augmentation_size = {aug_size}; Learning Rate = {lr}")
            ax[i, j].set_xlabel("# Epochs")
            ax[i, j].plot(
                training_loss, label="Training Loss" if i == j == 0 else None)
            ax[i, j].plot(
                validation_loss, label="Validation Loss" if i == j == 0 else None
            )
            ax[i, j].plot(
                validation_accuracy,
                label="Validation Accuracy" if i == j == 0 else None,
            )
            ax[i, j].plot(
                validation_f1_score,
                label="Validation F1 Score" if i == j == 0 else None,
            )

    # save the figure
    fig.legend()
    fig.savefig(
        f"./models/unet/{run_name}/hyperparameter_tuning.png", format="png")

    print("The best parameters are:")
    print(
        f"Validation Accuracy {best_validation_accuracy[1]} with params {best_accuracy_params} after Epoch {best_validation_accuracy[0]}"
    )
    print(
        f"Validation F1 Score {best_validation_f1score[1]} with params {best_f1score_params} after Epoch {best_validation_f1score[0]}"
    )


def do_unet_threshold_tuning(run_name, train_set_, val_set_, test_set_):
    # for each parameter configuration
    # get the validation accuracy and f1 scores for each epoch, for each potential threshold
    # plot the threshold evaluation for the epoch which had the highest threshold
    # AND plot the threshold evaluation for the highest threshold across the epochs
    """
    Validation Accuracy 0.9971325719509939 with params {'lr': 0.0001, 'augmentation_size': 5} after Epoch 13
    Validation F1 Score 0.9955362248449363 with params {'lr': 0.0001, 'augmentation_size': 5} after Epoch 13
    """

    # hyperparameter lists
    lr = 1e-4
    aug_size = 5
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    epoch = 13

    # determine the save path for the model
    model_path = f"./models/unet/15/{lr}_{aug_size}"
    save_path = f"./models/unet/{run_name}"
    os.makedirs(save_path, exist_ok=True)

    val_set = augmentation_wrapper(val_set_, n=0)
    train_set = augmentation_wrapper(train_set_, n=0)

    best_acc = -float("inf")
    best_threshold = None

    # test the different thresholds for the given learning rate and augmentation size
    precision_arr = []
    recall_arr = []
    false_positive_rate_arr = []
    for t in thresholds:
        # construct the parameter list
        params = {
            "lr": lr,
            "augmentation_size": aug_size,
            "threshold": t,
            "epoch": epoch,
        }

        # evaluate this model
        print(f"Evaluating {params}", flush=True)
        matrix = run_unet(
            model_path,
            deepcopy(train_set),
            deepcopy(val_set),
            None,
            params,
            True,
        )

        # store the precision, recall, and false positive rate
        precision_arr.append(precision(matrix))
        recall_arr.append(recall(matrix))
        false_positive_rate_arr.append(false_positive_rate(matrix))
        val_acc = accuracy(matrix)

        print(f"Validation Accuracy = {val_acc}")

        # remember the best accuracy
        if val_acc > best_acc:
            best_acc = val_acc
            best_threshold = t

    # plot an ROC curve
    font = 10
    plt.suptitle(
        f"ROC Curve for Learning Rate = {lr} and Augmentation Size = {aug_size}\n(Max Accuracy of {np.round(best_acc, 4)} at threshold {best_threshold})",
        fontsize=font,
    )
    plt.plot(
        [min(false_positive_rate_arr), max(false_positive_rate_arr)],
        [min(recall_arr), max(recall_arr)],
        label="Random",
    )
    plt.plot(
        false_positive_rate_arr, recall_arr, label=f"model after epoch {epoch} used"
    )
    plt.xlabel("False Positive Rate", fontsize=font)
    plt.ylabel("True Positive rate", fontsize=font)
    plt.legend()
    plt.savefig(f"{save_path}/roc.png", format="png")
    plt.clf()

    # plot a precision-recall curve
    plt.suptitle(
        f"Precision-Recall Curve for Learning Rate = {lr} and Augmentation Size = {aug_size}\n(Max Accuracy of {np.round(best_acc, 4)} at threshold {best_threshold})",
        fontsize=font,
    )
    # plt.plot(
    #     [min(recall_arr), max(recall_arr)],
    #     [min(precision_arr), max(precision_arr)],
    #     label="Random",
    # )
    plt.plot(recall_arr, precision_arr,
             label=f"model after epoch {epoch} used")
    plt.xlabel("Recall", fontsize=font)
    plt.ylabel("Precision", fontsize=font)
    plt.legend()
    plt.savefig(f"{save_path}/precision-recall.png", format="png")
    plt.clf()


def do_unet_k_fold(run_name, images, masks, k, parameters):
    # determine where to save
    save_path = f"./models/unet/kfold/{run_name}"
    os.makedirs(save_path, exist_ok=True)

    # create a KFold model selection object
    kfold = KFold(k)

    # create a list of indices
    indices = [i for i in range(len(images))]

    # graph a plot for accuracy and f1 score
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))
    fig.suptitle(
        f"Plots of Test Accuracy and F1 Score for each of the k-splits (k = {k})"
    )
    ax[0].set_title("Test Accuracy")
    ax[1].set_title("Test F1 Score")
    ax[0].set_xlabel("# Epochs")
    ax[1].set_xlabel("# Epochs")

    # keep track for the average accuracy and f1 score per epoch
    x_vals = [i for i in range(15)]
    total_accuracy = np.zeros(15, dtype=np.float32)
    total_f1_scores = np.zeros(15, dtype=np.float32)

    # now do a train and evaluation on each split
    i = 0
    for train_indices, test_indices in kfold.split(indices):
        # use the indices to construct data sets
        train_set_ = [(images[i], masks[i]) for i in train_indices]
        test_set_ = [(images[i], masks[i]) for i in test_indices]

        # augment the images
        train_set = augmentation_wrapper(
            deepcopy(train_set_), n=parameters["augmentation_size"]
        )
        test_set = augmentation_wrapper(deepcopy(test_set_), n=0)
        print(
            f"Loaded data for augmentation size {parameters['augmentation_size']}",
            flush=True,
        )

        fold_path = f"{save_path}/{i}"
        os.makedirs(fold_path, exist_ok=True)

        # train and evaluate on this combination
        (
            training_loss,
            validation_loss,
            validation_accuracy,
            validation_f1_score,
        ) = run_unet(
            fold_path,
            train_set,
            test_set,
            None,
            parameters,
        )

        # track the avg and f1 scores
        total_accuracy += np.array(validation_accuracy)
        total_f1_scores += np.array(validation_f1_score)

        # plot the f1 score and accuracy
        ax[0].plot(x_vals, validation_accuracy, label=i)
        ax[1].plot(x_vals, validation_f1_score)

        i += 1

    # average the accuracy and f1 scores
    total_accuracy /= k
    total_f1_scores /= k

    ax[0].plot(x_vals, total_accuracy, label="average")
    ax[1].plot(x_vals, total_f1_scores)

    # save the figure
    fig.legend()
    fig.savefig(f"{save_path}/kfold.png", format="png")

    print("The best average performance is:")
    print(
        f"Accuracy = {np.max(total_accuracy)} after Epoch {np.argmax(total_accuracy)}"
    )
    print(
        f"F1 Score = {np.max(total_f1_scores)} after Epoch {np.argmax(total_f1_scores)}"
    )


def do_unet_test(model_path, train_set_, test_set_, parameters):
    # filepath is the complete file path to the model to use for testing and prediction

    train_set = augmentation_wrapper(train_set_, n=0)
    test_set = augmentation_wrapper(test_set_, n=0)

    # get the confusion matrix
    matrix = run_unet(model_path, train_set, deepcopy(
        test_set), None, parameters, True)

    # print some stats
    print(f"Test Accuracy = {accuracy(matrix)}")
    print(f"Test F1 Score = {f1_score(matrix)}")


def do_unet_prediction(model_path, plot_save_name, image, mask, parameters):
    # image and mask should already have been augmented

    # create the model
    unet = UNet(n_channels=3, learning_rate=parameters["lr"])

    # load from some checkpoint
    checkpoint = unet_get_checkpoint(
        model_path, parameters["epoch"] if "epoch" in parameters else -1
    )
    if checkpoint:
        unet.load_model(checkpoint)
    else:
        unet.load_vgg_weights()

    # evaluate the model
    prediction = unet.predict(image, parameters["threshold"])

    # crop the true masks
    mask = mask.squeeze()
    vert_dim = 0
    hori_dim = 1
    dim0_size_diff = mask.size()[vert_dim] - prediction.size()[vert_dim]
    dim1_size_diff = mask.size()[hori_dim] - prediction.size()[hori_dim]

    if dim0_size_diff != 0 and dim1_size_diff != 0:
        mask = crop(
            mask,
            dim0_size_diff // 2,  # top left vertical component
            dim1_size_diff // 2,  # top left horizontal component
            prediction.size()[vert_dim],  # height of the cropped area
            prediction.size()[hori_dim],  # width of the cropped area
        )

    # plot
    fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(20, 5))

    acc = accuracy(
        confusion_matrix(prediction.flatten(), mask.flatten(),
                         parameters["threshold"])
    )
    fig.suptitle(f"Segmentation Accuracy = {np.round(acc, 4)}")

    ax[0].set_title("Image")
    ax[0].imshow(image.permute(1, 2, 0).numpy())
    ax[1].set_title("Predicted Mask")
    ax[1].imshow(prediction.numpy(), cmap="gray")
    ax[2].set_title("True Mask")
    ax[2].imshow(mask.numpy(), cmap="gray")
    ax[3].set_title("Difference")
    ax[3].imshow(
        mask.numpy().astype(np.uint8) - prediction.numpy().astype(np.uint8), cmap="gray"
    )

    for a in ax:
        a.axis("off")

    fig.savefig(plot_save_name, bbox_inches="tight", format="png")


def execute_unet():
    # read in the data for unet
    # --------------------------- DO NOT TOUCH ---------------------------
    images, masks = read_data()
    images = images.astype(np.float32) / 255.0

    train_set_, val_set_, test_set_ = split_dataset(images, masks)
    # --------------------------- DO NOT TOUCH ---------------------------

    # todo: uncomment the below sections depending on what you want to run

    # hyperparameter search
    # do_unet_hyperparameter_search("15", train_set_, val_set_, test_set_)

    # do the threshold tuning
    # do_unet_threshold_tuning("15_thresholds", train_set_, val_set_, test_set_)

    # do k-fold validation
    # do_unet_k_fold(
    #     "15", images, masks, 6, {"lr": 1e-4, "threshold": 0.4, "augmentation_size": 5}
    # )

    # evaluate on the test set
    # do_unet_test(
    #     "./models/unet/15/0.0001_5",
    #     train_set_,
    #     test_set_,
    #     {"lr": 1e-4, "threshold": 0.4, "augmentation_size": 5, "epoch": 13},
    # )

    # do prediction on some images
    test_set = augmentation_wrapper(test_set_, n=0)
    image_save_path = "./models/unet/images"
    os.makedirs(image_save_path, exist_ok=True)
    images_to_examine = [0, 1, 2, 3, 4]
    for i in images_to_examine:
        if i < 0 or i >= len(test_set):
            print(f"Image {i} does not exist. Skipping...")
            continue
        else:
            print(f"Processing Image {i}")

    do_unet_prediction(
        "./models/unet/15/0.0001_5",
        f"{image_save_path}/{i}.png",
        test_set[i][0],
        test_set[i][1],
        {"lr": 1e-4, "threshold": 0.4, "augmentation_size": 5, "epoch": 13},
    )


if __name__ == "__main__":

    # Uncomment the section you want to run

    # execute_gmm()
    # execute_unet()
    print("Uncomment the section you want to run in the main function")
