import torch
from torchvision import transforms as T
from torch.utils.data import DataLoader
from dataset.miniImageNet import miniImageNet
from dataset.tiredImageNet import tiredImageNet
from dataset.fewshot_dataset import FewShotDataset_train, FewShotDataset_eval
from args.args_miniImageNet import argument_parser
parser = argument_parser()
args = parser.parse_args()
class DataManager(object):
    """
    DataManager
    """

    def __init__(self, args):
        super(DataManager, self).__init__()
        print("Initializing dataset {}".format(args.dataset))
        self.args = args
        if args.dataset=='miniImageNet':
            self.dataset = miniImageNet()
        elif args.dataset=='tiredImageNet':
            self.dataset = tiredImageNet()
        self.transform_train = T.Compose([
            T.Resize((84, 84), interpolation=3),
            T.RandomCrop(84, padding=8),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            T.RandomErasing(0.5)
        ])
        self.transform_eval = T.Compose([
            T.Resize((84, 84), interpolation=3),
            T.ToTensor(),
            T.Normalize(mean = [0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.FewShotTrainset = FewShotDataset_train(
            datapoints=self.dataset.train,
            labels2idx=self.dataset.train_labels2idx,
            labelIDs=self.dataset.train_labelIDs,
            N_way=self.args.way,
            K_shot=self.args.shot,
            nTestNovel=self.args.nTrainNovel,
            epoch_size=self.args.num_episode_train,
            transform=self.transform_train)
        self.FewShotTestset = FewShotDataset_eval(
            datapoints=self.dataset.test,
            labels2idx=self.dataset.test_labels2idx,
            labelIDs=self.dataset.test_labelIDs,
            N_way=self.args.way,
            K_shot=self.args.shot,
            nTestNovel=self.args.nTestNovel,
            epoch_size=self.args.num_episode_test,
            transform=self.transform_eval)
        self.FewShotValset = FewShotDataset_eval(
            datapoints=self.dataset.val,
            labels2idx=self.dataset.val_labels2idx,
            labelIDs=self.dataset.val_labelIDs,
            N_way=self.args.way,
            K_shot=self.args.shot,
            nTestNovel=self.args.nTestNovel,
            epoch_size=args.num_episode_val,
            transform=self.transform_eval)
    def create_dataloader(self, phase='train', mode = 'few-shot'):
        if mode=='few-shot':
            if phase=='train':
                return DataLoader(
                    dataset=self.FewShotTrainset,
                    batch_size=self.args.train_batch,
                    pin_memory=True,
                    num_workers=self.args.workers
                )
            if phase=='val':
                return DataLoader(
                    dataset=self.FewShotValset,
                    batch_size=self.args.test_batch,
                    pin_memory=True,
                    num_workers=self.args.workers
                )
            if phase=='test':
                return DataLoader(
                    dataset=self.FewShotTestset,
                    batch_size=self.args.test_batch,
                    pin_memory=True,
                    num_workers=self.args.workers
                )
            else:
                raise KeyError("Unknown phase: {}".format(phase))
        else:
            #TODO standard dataloader
            pass

if __name__ == '__main__':

    dm = DataManager(args)
    test_loader = dm.create_dataloader(phase='train', mode='few-shot')
    # for (images_spt, labels_spt, images_qry, labels_qry, cids_qry) in test_loader:
    #     print(cids_qry)
    #     print(labels_qry)
    (images_spt, labels_spt, images_qry, labels_qry, cids_qry) = iter(test_loader).next()
    print(labels_spt)
    print(labels_qry)
