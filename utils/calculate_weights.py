import os
import numpy as np
from tqdm import tqdm

def calculate_weights_labels(dataset, dataloader, num_class):
    # create an instance from the data loader
    z = np.zeros((num_class,))
    print("Calculating classes weights")
    for sample in tqdm(dataloader):
        y = sample["label"]
        y = y.detach().cpu().numpy()
        mask = (y>=0) & (y<num_class)
        labels = y[mask].astype(np.uint8)
        count_l = np.bincount(labels, minlength=num_class)
        z += count_l

    home_dir = "/Users/nopz/Devel/mastery/segmentation/deeplabv3"
    tqdm(dataloader).close()
    total_frequency = np.sum(z)
    class_weights = []
    for frequency in z:
        class_weight = 1 / (np.log(1.02 + (frequency/total_frequency)))
        class_weights.append(class_weight)
    ret = np.array(class_weights)
    class_weight_path = os.path.join(home_dir, dataset+"_classes_weights.npy")
    np.save(class_weight_path, ret)

    return ret
