import matplotlib as plt 
import torch
import numpy as np
import cv2
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from skimage.feature import hog

KMEANS_ITERATIONS = 100
NR_CLUSTERS = 10
HOG_SIZE = 36
VOCABULARY_SIZE = 300
KEY_POINT_SIZE = 32
EDGETHRESHOLD = 25

sift = cv2.SIFT_create()
orb = cv2.ORB_create(nfeatures=100, edgeThreshold=EDGETHRESHOLD, patchSize = KEY_POINT_SIZE)

def convert_fashion_tensor_to_np(torch_image : torch.Tensor):
    torch_image = (torch_image * 255).type(torch.uint8)
    torch_image = torch_image.permute([0, 2, 3, 1])
    torch_image = torch_image.squeeze(dim=3)
    torch_image = torch_image.detach().numpy()
    return np.array(torch_image)


def compute_descriptors(img : np.array):
    keypoints = orb.detect(img, None)
    if (len(keypoints)) == 0:
        height, width = img.shape
        out = []
        for i in range(0, height, 16):
          for j in range(0, width, 16):
            out.append(np.array(hog(img[i:i+16, j:j+16], orientations=9, pixels_per_cell=(8, 8),
              cells_per_block=(2, 2), feature_vector=True)))
        return np.vstack(out)
    descriptors = np.zeros(shape=(len(keypoints), HOG_SIZE))
    for idx, kp in enumerate(keypoints):
        y, x, size = round(kp.pt[0]), round(kp.pt[1]), round(kp.size)
        select_window = img[x - int(size / 2)  : x + int(size / 2), y - int(size / 2) : y + int(size / 2)]
        descriptors[idx] = np.array(hog(select_window[0:16, 0:16], orientations=9,
                               pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True))
    return descriptors

def compute_vocabulary(dataloader, nr_features):
  LIMIT = 100
  vocabulary = []
  kmeans = KMeans(n_clusters=nr_features, max_iter=KMEANS_ITERATIONS)
  for idx, (images, _) in enumerate(dataloader):
    descriptors = []
    for img in convert_fashion_tensor_to_np(images):
      descriptors.append(compute_descriptors(img))
    descriptors = np.vstack(descriptors)
    descriptors = descriptors[torch.randperm(len(descriptors))]
    vocabulary.append(kmeans.fit(descriptors[:LIMIT]).cluster_centers_)
    if (idx % 100 == 0):
       print("a ajuns la epoca:", idx)
  return np.vstack(vocabulary)

def return_features_image(img : np.array, vocabulary):
    out = np.zeros(shape=(vocabulary.shape[0],))
    dists = cdist(compute_descriptors(img), vocabulary, metric='euclidean')
    bin_assignment = np.argmin(dists, axis=1)
    for bin_idx in bin_assignment:
        out[bin_idx] += 1
    return out


def create_initial_features(dataloader, vocabulary):
    features = []
    all_labels = []
    for (images, labels) in dataloader:
        current_features = np.zeros(shape=(len(images), vocabulary.shape[0]))
        for idx, img in enumerate(convert_fashion_tensor_to_np(images)):
            current_features[idx] = return_features_image(img, vocabulary)
        features.append(current_features)
        all_labels.append(labels.detach().numpy())
    return np.vstack(features), np.concatenate(all_labels)


def split_data_in_training_validation(X_train, y_train, fashion_keys, split_factor):
    training = []
    validation = []
    training_labels = []
    validation_labels = []
    for key in fashion_keys:
        indices = y_train == key
        current_values = X_train[indices]
        training_size = int(len(current_values) * (1 - split_factor))
        training.append(X_train[indices][:training_size])
        validation.append(X_train[indices][training_size:])
        training_labels += [key] * training_size
        validation_labels += [key] * (len(current_values) - training_size)
    training = torch.tensor(np.vstack(training), dtype=torch.uint8)
    validation = torch.tensor(np.vstack(validation), dtype=torch.uint8)
    training_labels = torch.tensor(np.array(training_labels), dtype=torch.uint8)
    validation_labels = torch.tensor(np.array(validation_labels), dtype=torch.uint8)
    r_train = torch.randperm(training.shape[0])
    r_val = torch.randperm(validation.shape[0])
    return training[r_train], training_labels[r_train], validation[r_val], validation_labels[r_val]



def compute_attributes(vocabulary, train_dataloader, test_dataloader, fashion_keys, split_factor):
    initials_train_features, all_train_labels = create_initial_features(train_dataloader, vocabulary)
    print(initials_train_features.shape, all_train_labels.shape)
    X_test, y_test = create_initial_features(test_dataloader, vocabulary)
    print(X_test.shape, y_test.shape)
    X_train, y_train, X_validation, y_validation = split_data_in_training_validation(initials_train_features, all_train_labels, fashion_keys, split_factor)
    return X_train, y_train, X_validation, y_validation, X_test, y_test


