import argparse

def argument_parser():
    parser = argparse.ArgumentParser(description='Train TAMN')
    # ************************************************************
    # Datasets (general)
    # ************************************************************
    parser.add_argument('--dataset', type=str, default='tiredImageNet')

    parser.add_argument('--workers', default=4, type=int,
                        help="number of data loading workers (default: 4)")
    parser.add_argument('--height', type=int, default=84,
                        help="height of an image (default: 84)")
    parser.add_argument('--width', type=int, default=84,
                        help="width of an image (default: 84)")
    parser.add_argument('--train_categories', type=int, default=351)

    # ************************************************************
    # Optimization options
    # ************************************************************
    parser.add_argument('--optim', type=str, choices=['Adam', 'SGD'], default='SGD',
                        help="optimization algorithm")
    parser.add_argument('--lr', '--learning-rate', default=0.05, type=float,
                        help="initial learning rate")
    parser.add_argument('--max_epoch', default=80, type=int,
                        help="maximum epochs to run")
    parser.add_argument('--stepsize', default=[60], nargs='+', type=int,
                        help="stepsize to decay learning rate")
    parser.add_argument('--LUT_lr', default=[(20, 0.05), (40, 0.01), (60, 0.001), (80, 0.0001)],
                        help="multistep to decay learning rate")
    parser.add_argument('--train_batch', default=4, type=int,
                        help="train batch size")
    parser.add_argument('--test_batch', default=4, type=int,
                        help="test batch size")

    # ************************************************************
    # Miscs
    # ************************************************************
    parser.add_argument('--save_dir', type=str, default='./result/tiredImageNet/tamn/seed23/5w1s/')
    parser.add_argument('--gpu', default='0', type=str)
    parser.add_argument('--run_path', type=str, default=None)

    # ************************************************************
    # FewShot settting
    # ************************************************************
    parser.add_argument('--scale_cls', type=int, default=7,help='temperature coefficient')
    parser.add_argument('--epos', type=int, default=3, help='position of the enhancer if the enhancer exists')
    parser.add_argument('--way', type=int, default=5,
                        help='number of novel categories')
    parser.add_argument('--shot', type=int, default=1,
                        help='number of training examples per novel category.')
    parser.add_argument('--nTrainNovel', type=int, default=6 * 5,
                        help='number of test examples for all the novel category when training')
    parser.add_argument('--nTestNovel', type=int, default=15 * 5,
                        help='number of test examples for all the novel category when evaluating')
    parser.add_argument('--num_episode_train', type=int, default=13980,
                        help='number of tasks per epoch when training')
    parser.add_argument('--num_episode_val', type=int, default=600,
                        help='number of tasks per epoch when validating')
    parser.add_argument('--num_episode_test', type=int, default=2000,
                        help='number of tasks per epoch when testing')
    parser.add_argument('--seed', type=int, default=23)
    
    # ************************************************************
    # Network architecture
    # ************************************************************
    parser.add_argument('--method', type=str, choices=['tamn', 'lgm_net', 'can', 'dn4'], default='tamn')
    parser.add_argument('--use_global', dest='use_global', action='store_true')
    parser.add_argument('--use_enhancer', dest='use_enhancer', action='store_true')
    parser.add_argument('--use_matt', dest='use_matt', action='store_true')
    return parser