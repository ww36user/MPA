from sklearn.svm import SVC
import clip
import random
import warnings
from sklearn.metrics.pairwise import cosine_similarity
import torchvision.transforms as transforms
import torch.nn.functional as F
from torchvision.transforms import functional as F
import numpy as np
from sklearn.linear_model import LogisticRegression
from io_utils import parse_args_test
import torch

warnings.filterwarnings("ignore", category=Warning)

# -------------------------------------------------------------
# Apply center crop and resize transformation to support images
# -------------------------------------------------------------
def apply_center_crop_resize(x_support, crop_size=200, resize_size=224):
    center_crop = transforms.CenterCrop(crop_size)
    resize = transforms.Resize(resize_size)
    # Apply center crop and resize sequentially to each image tensor
    cropped_resized_images = torch.stack([resize(center_crop(img)) for img in x_support])
    return cropped_resized_images


# -------------------------------------------------------------
# Calculate average feature difference between classes
# -------------------------------------------------------------
def calculate_difference(features, labels):
    # Get all unique class labels
    unique_labels = np.unique(labels)
    # Group features by class
    class_features = {label: features[labels == label] for label in unique_labels}
    # Compute class mean feature (prototype) for each class
    class_averages = {label: np.mean(features, axis=0) for label, features in class_features.items()}
    num_classes = len(class_averages)

    # Initialize similarity matrix
    similarity_matrix = np.zeros((num_classes, num_classes))
    labels_list = list(class_averages.keys())

    # Compute cosine similarity between every pair of class prototypes
    for i, label1 in enumerate(labels_list):
        for j, label2 in enumerate(labels_list):
            if i <= j:
                similarity_matrix[i, j] = cosine_similarity(
                    class_averages[label1].reshape(1, -1),
                    class_averages[label2].reshape(1, -1)
                )[0, 0]

    # Normalize similarities to [0, 1]
    max_similarity = np.max(similarity_matrix)
    min_similarity = np.min(similarity_matrix)
    normalized_similarities = (similarity_matrix - min_similarity) / (max_similarity - min_similarity)

    # Compute average inter-class difference
    average_difference = 1 - np.mean(normalized_similarities[np.triu_indices(num_classes, k=1)])
    return average_difference


# -------------------------------------------------------------
# Generate uncertain dummy classes for data augmentation
# -------------------------------------------------------------
def generate_uncertain_classes(out_support_LR, y, num_dummy_per_class, factor):
    unique_labels = np.unique(y)
    centroids = {}

    # Calculate centroid (mean feature) for each class
    for label in unique_labels:
        class_indices = np.where(y == label)[0]
        class_embeddings = out_support_LR[class_indices]
        centroids[label] = np.mean(class_embeddings, axis=0)

    dummy_labels = list(range(0, 5))
    X_dummies = []
    y_dummies = []

    # Create interpolated dummy samples between adjacent classes
    for i in range(len(dummy_labels)):
        label_1 = dummy_labels[i]
        label_2 = dummy_labels[(i + 1) % len(dummy_labels)]
        C_1 = centroids.get(label_1)
        C_2 = centroids.get(label_2)

        # Interpolate between two centroids with noise
        for alpha in np.linspace(0.2, 0.8, num=num_dummy_per_class):
            C_dummy = alpha * C_1 + (1 - alpha) * C_2
            X_dummy = np.random.randn(1, 768 * 2)  # Add random Gaussian noise
            final = C_dummy * factor + X_dummy * (1 - factor)
            X_dummies.append(final)
            y_dummies.append(label_1)

    # Combine dummy samples into arrays
    X_dummy_combined = np.array(X_dummies)
    y_dummy = np.tile(range(5, 10), 5)
    y_dummy.sort()

    # Reshape dummy data to match input dimensions
    new_shape = (X_dummy_combined.shape[0] * X_dummy_combined.shape[1], X_dummy_combined.shape[2])
    X_dummy_combined = X_dummy_combined.reshape(new_shape)

    # Append dummy data to the original support set
    out_support_LR = np.concatenate((out_support_LR, X_dummy_combined), axis=0)
    y = np.concatenate((y, y_dummy), axis=0)
    return out_support_LR, y


# -------------------------------------------------------------
# Set random seed for reproducibility
# -------------------------------------------------------------
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# -------------------------------------------------------------
# Transductive refinement: augment support set using confident query samples
# -------------------------------------------------------------
def Tr(support_set, query_set, pred, K):
        softmax = torch.nn.Softmax(dim=1)
        pred = softmax(pred)
        score, index = pred.max(1)  # Get predicted class and confidence score

        all_class_score = []
        all_class_index = []

        # Group query samples by predicted class
        for j in range(5):
            current_class_score = []
            current_class_index = []
            for i in range(75):
                if index[i] == j:
                    current_class_score.append(score[i])
                    current_class_index.append(i)
            all_class_score.append(current_class_score)
            all_class_index.append(current_class_index)

        tr_support_set = []

        # Add top-K confident query samples to each class
        for i in range(5):
            current_class_index = all_class_index[i]
            if len(current_class_index) == 0:
                current_support_set = support_set[i]  # No query added
            elif len(current_class_index) <= K:
                current_query_image = query_set[current_class_index]
                current_support_set = torch.cat((support_set[i], current_query_image), 0)
            else:
                current_class_score = all_class_score[i]
                # Sort scores and select top-K
                current_class_scores_on_cpu = [score.cpu().numpy() for score in current_class_score]
                current_class_score_index = np.argsort(current_class_scores_on_cpu)
                current_class_index = np.array(current_class_index)[current_class_score_index[-K:].tolist()]
                current_query_image = query_set[current_class_index]
                current_support_set = torch.cat((support_set[i], current_query_image), 0)
            tr_support_set.append(current_support_set)

        # Combine all refined support sets
        tr_support_set_all = torch.cat((tr_support_set[0], tr_support_set[1],
                                        tr_support_set[2], tr_support_set[3],
                                        tr_support_set[4]), 0)

        # Generate corresponding ground-truth labels
        tr_support_set_gt = [0] * len(tr_support_set[0]) + [1] * len(tr_support_set[1]) + \
                            [2] * len(tr_support_set[2]) + [3] * len(tr_support_set[3]) + \
                            [4] * len(tr_support_set[4])
        tr_support_set_gt = np.array(tr_support_set_gt)
        return tr_support_set_all, tr_support_set_gt


# -------------------------------------------------------------
# Rotate each image in the batch by a given angle
# -------------------------------------------------------------
def rotate_batch(x_support, rotate_num):
    batch_size = x_support.size(0)
    rotated_images = torch.empty_like(x_support)
    # Apply rotation to each image tensor individually
    for i in range(batch_size):
        rotated_images[i] = F.rotate(x_support[i], rotate_num)
    return rotated_images

def test(novel_loader, model, params):
    iter_num = len(novel_loader)  # Total number of test episodes
    acc_all_LR = []  # To store accuracy for each episode

    with torch.no_grad():  # Disable gradient computation for efficiency
        for i, (x, label, label_name, label_name2,
                label_des, label_des2, label_des3, label_des4, label_des5) in enumerate(novel_loader):

            # Split the batch into query and support sets
            x_query = x[:, params.n_support:, :, :, :].contiguous().view(params.n_way * params.n_query, *x.size()[2:]).cuda()
            x_support = x[:, :params.n_support, :, :, :].contiguous().view(params.n_way * params.n_support, *x.size()[2:]).cuda()  # (25, 3, 224, 224)

            # Combine all textual descriptions
            text_des = label_name + label_name2 + label_des + label_des2 + label_des3 + label_des4 + label_des5

            # --- Horizontal flipping for augmentation ---
            flipped_images = torch.empty_like(x_support)
            for i in range(x_support.size(0)):
                flipped_images[i] = F.hflip(x_support[i])

            x_query_flipped_images = torch.empty_like(x_query)
            for i in range(x_query.size(0)):
                x_query_flipped_images[i] = F.hflip(x_query[i])

            # --- Center crop and resize augmentations ---
            center_cropped_x_support = apply_center_crop_resize(x_support, crop_size=170)
            center_cropped_x_support2 = apply_center_crop_resize(x_support, crop_size=200)
            center_cropped_x_support3 = apply_center_crop_resize(x_support, crop_size=120)
            center_cropped_x_support_flipped = apply_center_crop_resize(flipped_images, crop_size=170)
            center_cropped_x_support2_flipped = apply_center_crop_resize(flipped_images, crop_size=200)
            center_cropped_x_support3_flipped = apply_center_crop_resize(flipped_images, crop_size=120)

            # --- Color jitter augmentation ---
            color_jitter = transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2)
            transformed_images = []
            for i in range(x_support.shape[0]):
                img_pil = transforms.ToPILImage()(x_support[i])
                img_transformed = color_jitter(img_pil)
                img_transformed_tensor = transforms.ToTensor()(img_transformed)
                transformed_images.append(img_transformed_tensor)
            color_jitter_x_support = torch.stack(transformed_images)

            transformed_images_flipped = []
            for i in range(flipped_images.shape[0]):
                img_pil = transforms.ToPILImage()(flipped_images[i])
                img_transformed = color_jitter(img_pil)
                img_transformed_tensor = transforms.ToTensor()(img_transformed)
                transformed_images_flipped.append(img_transformed_tensor)
            color_jitter_x_support_flipped = torch.stack(transformed_images_flipped)

            # --- Rotation augmentations ---
            rotated_image_45 = rotate_batch(x_support, 45)
            rotated_image_90 = rotate_batch(x_support, 90)
            rotated_image_180 = rotate_batch(x_support, 180)
            rotated_image_270 = rotate_batch(x_support, 270)
            rotated_image_315 = rotate_batch(x_support, 315)

            rotated_image_45_flipped = rotate_batch(flipped_images, 45)
            rotated_image_90_flipped = rotate_batch(flipped_images, 90)
            rotated_image_180_flipped = rotate_batch(flipped_images, 180)
            rotated_image_270_flipped = rotate_batch(flipped_images, 270)
            rotated_image_315_flipped = rotate_batch(flipped_images, 315)

            # Create label vector for support samples
            y = np.tile(range(params.n_way), params.n_support)
            temp_y = y

            # Concatenate all augmentations for support set
            aug_images = torch.cat((rotated_image_45.cpu(), rotated_image_90.cpu()), dim=0)
            aug_images = torch.cat((aug_images, rotated_image_180.cpu()), dim=0)
            aug_images = torch.cat((aug_images, rotated_image_270.cpu()), dim=0)
            aug_images = torch.cat((aug_images, rotated_image_315.cpu()), dim=0)
            aug_images = torch.cat((aug_images, center_cropped_x_support.cpu()), dim=0)
            aug_images = torch.cat((aug_images, center_cropped_x_support2.cpu()), dim=0)
            aug_images = torch.cat((aug_images, center_cropped_x_support3.cpu()), dim=0)
            aug_images = torch.cat((aug_images, color_jitter_x_support.cpu()), dim=0)

            # Same for flipped images
            aug_images_flipped = torch.cat((rotated_image_45_flipped.cpu(), rotated_image_90_flipped.cpu()), dim=0)
            aug_images_flipped = torch.cat((aug_images_flipped, rotated_image_180_flipped.cpu()), dim=0)
            aug_images_flipped = torch.cat((aug_images_flipped, rotated_image_270_flipped.cpu()), dim=0)
            aug_images_flipped = torch.cat((aug_images_flipped, rotated_image_315_flipped.cpu()), dim=0)
            aug_images_flipped = torch.cat((aug_images_flipped, center_cropped_x_support_flipped.cpu()), dim=0)
            aug_images_flipped = torch.cat((aug_images_flipped, center_cropped_x_support2_flipped.cpu()), dim=0)
            aug_images_flipped = torch.cat((aug_images_flipped, center_cropped_x_support3_flipped.cpu()), dim=0)
            aug_images_flipped = torch.cat((aug_images_flipped, color_jitter_x_support_flipped.cpu()), dim=0)

            # Tokenize text descriptions for CLIP
            text_des = clip.tokenize(text_des).to(device)

            # --- Feature extraction using CLIP ---
            with torch.no_grad():
                out_support = model.encode_image(x_support)
                out_support_flipped = model.encode_image(flipped_images)
                out_query = model.encode_image(x_query)
                out_query_flipped = model.encode_image(x_query_flipped_images)
                aug_images = model.encode_image(aug_images.cuda())
                aug_images_flipped = model.encode_image(aug_images_flipped.cuda())
                out_classes_text = model.encode_text(text_des)

            # Concatenate features for multimodal fusion (image + flipped + text)
            out_support = torch.cat((out_support, out_support_flipped), dim=1)
            out_query = torch.cat((out_query, out_query_flipped), dim=1)
            aug_images = torch.cat((aug_images, aug_images_flipped), dim=1)
            out_classes_text = torch.cat((out_classes_text, out_classes_text), dim=1)

            # Convert to NumPy arrays for classifier training
            _, c = out_support.size()
            out_support_LR3 = out_classes_text.cpu().numpy()
            out_support_LR2 = aug_images.cpu().numpy()
            out_support_LR = out_support.cpu().numpy()

            # Combine all features (support + augmentations + text)
            out_support_LR = np.concatenate((out_support_LR, out_support_LR2), axis=0)
            out_support_LR = np.concatenate((out_support_LR, out_support_LR3), axis=0)
            out_query_LR = out_query.cpu().numpy()

            # Concatenate labels for all augmentations
            y.sort()
            y = np.concatenate((y, temp_y), axis=0)  # 45
            y = np.concatenate((y, temp_y), axis=0)  # 90
            y = np.concatenate((y, temp_y), axis=0)  # 180
            y = np.concatenate((y, temp_y), axis=0)  # 270
            y = np.concatenate((y, temp_y), axis=0)  # 315
            y = np.concatenate((y, temp_y), axis=0)  # center1
            y = np.concatenate((y, temp_y), axis=0)  # center2
            y = np.concatenate((y, temp_y), axis=0)  # center3
            y = np.concatenate((y, temp_y), axis=0)  # color_jitter

            # Add textual labels
            y_text = np.tile(range(params.n_way), 1)
            y = np.concatenate((y, y_text), axis=0)  # name
            y = np.concatenate((y, y_text), axis=0)  # name2
            y = np.concatenate((y, y_text), axis=0)  # des1
            y = np.concatenate((y, y_text), axis=0)  # des2
            y = np.concatenate((y, y_text), axis=0)  # des3
            y = np.concatenate((y, y_text), axis=0)  # des4
            y = np.concatenate((y, y_text), axis=0)  # des5

            # --- Compute inter-class difference (used for dummy class generation) ---
            factor = calculate_difference(out_support_LR, y)

            # Generate uncertain dummy classes based on class centroids
            out_support_LR, y = generate_uncertain_classes(out_support_LR=out_support_LR, y=y,
                                                           num_dummy_per_class=5, factor=factor)

            # Train a linear SVM classifier on all support data
            classifier = SVC(kernel='linear', probability=True, max_iter=1000).fit(X=out_support_LR, y=y)

            # Predict probabilities for query samples
            pred = classifier.predict_proba(out_query_LR)

            # --- Iterative prototype refinement using logistic regression ---
            out_support_LR = out_support.view(params.n_way, params.n_support, c)
            for k in range(7):  # Iterative fine-tuning rounds
                pred = torch.from_numpy(pred).cuda()
                tr_support_set, tr_support_set_gt = Tr(out_support_LR, out_query, pred, 10)
                tr_support_set = tr_support_set.cpu().numpy()
                classifier = LogisticRegression(max_iter=1000).fit(X=tr_support_set, y=tr_support_set_gt)
                pred = classifier.predict_proba(out_query_LR)

            # --- Compute top-1 accuracy ---
            y_query = np.repeat(range(params.n_way), params.n_query)
            pred = torch.from_numpy(pred).cuda()
            topk_scores, topk_labels = pred.data.topk(1, 1, True, True)
            topk_ind = topk_labels.cpu().numpy()
            top1_correct = np.sum(topk_ind[:, 0] == y_query)
            correct_this, count_this = float(top1_correct), len(y_query)
            acc_all_LR.append((correct_this / count_this * 100))

    # --- Compute final average accuracy and confidence interval ---
    acc_all = np.asarray(acc_all_LR)
    acc_mean = np.mean(acc_all)
    acc_std = np.std(acc_all)
    print('acc : %4.2f%% +- %4.2f%%' % (acc_mean, 1.96 * acc_std / np.sqrt(iter_num)))


if __name__ == '__main__':
    params = parse_args_test()  # Parse test configuration parameters
    setup_seed(params.seed)  # Set random seed for reproducibility
    print("use llm:gpt-4")

    # Load test dataset
    import test_dataset.test_dataset_text_gpt4 as test_dataset
    datamgr = test_dataset.Eposide_DataManager(
        data_path=params.current_data_path,  # Dataset path
        num_class=params.current_class,      # Number of classes in dataset
        image_size=params.image_size,        # Image resolution
        n_way=params.n_way,                  # Number of classes per episode
        n_support=params.n_support,          # Number of support samples per class
        n_query=params.n_query,              # Number of query samples per class
        n_eposide=params.test_n_eposide      # Number of test episodes
    )
    novel_loader = datamgr.get_data_loader(aug=False)  # Load data without augmentation

    # Select device for inference
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load CLIP model (from github: https://github.com/openai/CLIP)
    model_list = ["ViT-L/14", "ViT-B/16", "ViT-B/32", "RN101"]
    model, preprocess = clip.load(model_list[0], device=device)

    '''
    # URLs for reference CLIP model weights
    _MODELS = {
    "RN101": "https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt",
    "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
    "ViT-B/16": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
    "ViT-L/14": "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
    }
    '''
    model.cuda()
    model.eval()  # Set model to evaluation mode

    # Run test function
    test(novel_loader, model, params)
