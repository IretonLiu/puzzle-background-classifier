import glob
import cv2
import matplotlib.pyplot as plt
import numpy as np
from gmm import GaussianMixtureModel
from scipy.stats import multivariate_normal
from ellipsoid import get_cov_ellipsoid
from sklearn.mixture import GaussianMixture

res = (1024, 768)


def read_data():
    # glob all images from data
    image_files = glob.glob('data/images-1024x768/*.png')
    mask_files = glob.glob('data/masks-1024x768/*.png')

    # convert to images using cv2 and convert to float RGB
    images = np.array([cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2HSV).astype(np.float32) / 255.0
                       for image in image_files])
    masks = np.array([cv2.cvtColor(cv2.imread(mask), cv2.COLOR_BGR2GRAY) > 0
                      for mask in mask_files])

    return images, masks


def split_dataset(images, masks):
    # split the images and masks into training validation and test with ration 70:15:15
    training_images = images[:int(len(images) * 0.7)]
    validation_images = images[int(len(images) * 0.7):int(len(images) * 0.85)]
    testing_images = images[int(len(images) * 0.85):]

    training_masks = masks[:int(len(masks) * 0.7)]
    validation_masks = masks[int(len(masks) * 0.7):int(len(masks) * 0.85)]
    testing_masks = masks[int(len(masks) * 0.85):]

    return training_images, validation_images, testing_images, training_masks, validation_masks, testing_masks


def scatter_plot(data, step, gmm=None):
    # 3d scatter plot of train_data_foreground
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(
        data[::step, 0], data[::step, 1], data[::step, 2])

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    if gmm:
        for j in range(gmm.n_components):
            # plot the gaussian
            x, y, z = get_cov_ellipsoid(gmm.covariances_[j], gmm.means_[j])
            ax.plot_surface(x, y, z, color='r', alpha=0.1)
    plt.show()


def main():
    images, masks = read_data()
    train_images, val_images, test_images, train_masks, val_masks, test_masks = split_dataset(
        images, masks)

    # convert to pixels
    train_data = np.concatenate([image.reshape(-1, 3)
                                 for image in train_images])
    train_data_masks = np.concatenate(
        [mask.reshape(-1, 1) for mask in train_masks])

    train_data_foreground = train_data[train_data_masks[:, 0]]
    train_data_background = train_data[~train_data_masks[:, 0]]
    data = np.array([-2, 5, 7, 14, 15])
    gmm = GaussianMixtureModel(
        6, train_data_foreground.shape[1], max_iter=700, seed=2)
    # gmm = GaussianMixtureModel(3, 2, max_iter=200)
    gmm.fit(train_data_foreground[::1000, :])

    # gm = GaussianMixture(n_components=3, covariance_type='full', max_iter=100)
    # gm.fit(train_data_foreground)

    scatter_plot(train_data_foreground, 1000, gmm)


if __name__ == "__main__":
    main()
