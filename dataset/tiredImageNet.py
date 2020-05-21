import os
import torch


class tiredImageNet(object):
    """
    To build tiredImageNet dataset.
    Dataset statistics:
    # 351 * 600 (train) + 97 * 600 (val) + 160 * 600 (test)
    """
    dataset_dir = '/home/miaoxinyuan/datasets/tiered_imagenet/'

    def __init__(self):
        super(tiredImageNet, self).__init__()
        self.train_dir = os.path.join(self.dataset_dir, 'train')
        self.val_dir = os.path.join(self.dataset_dir, 'val')
        self.test_dir = os.path.join(self.dataset_dir, 'test')

        self.train, self.train_labels2idx, self.train_labelIDs = self.process(
            self.train_dir)
        self.val, self.val_labels2idx, self.val_labelIDs = self.process(
            self.val_dir)
        self.test, self.test_labels2idx, self.test_labelIDs = self.process(
            self.test_dir)

        num_total_cats = len(self.train_labelIDs) + len(self.val_labelIDs) + len(self.test_labelIDs)
        num_total_imgs = len(self.train + self.val + self.test)
        print("=> tiredImageNet loaded")
        print("Dataset statistics:")
        print("  ------------------------------")
        print("  subset   | # cats | # images")
        print("  ------------------------------")
        print("  train    | {:5d} | {:8d}".format(len(self.train_labelIDs), len(self.train)))
        print("  val      | {:5d} | {:8d}".format(len(self.val_labelIDs),   len(self.val)))
        print("  test     | {:5d} | {:8d}".format(len(self.test_labelIDs),  len(self.test)))
        print("  ------------------------------")
        print("  total    | {:5d} | {:8d}".format(num_total_cats, num_total_imgs))
        print("  ------------------------------")

    def process(self, dir_path):
        categories = sorted(os.listdir(dir_path))  # class names [351]
        cats2label = {cat: label for label, cat in enumerate(
            categories)}  # {'cats0':0, 'cats1':1,...'cats350':350,...}

        datapoints = []
        labels = []
        for cat in categories:
            for img_path in sorted(os.listdir(os.path.join(dir_path, cat))):
                if '.jpg' not in img_path:
                    continue
                label = cats2label[cat]
                datapoints.append(
                    (os.path.join(dir_path, cat, img_path), label))
                labels.append(label)
        # labels [0, 0, 0,..., 350, 350] len: 351*600

        labels2idx = {}
        for idx, label in enumerate(labels):
            if label not in labels2idx:
                labels2idx[label] = []
            labels2idx[label].append(idx)

        labelIDs = sorted(labels2idx.keys())
        return datapoints, labels2idx, labelIDs
