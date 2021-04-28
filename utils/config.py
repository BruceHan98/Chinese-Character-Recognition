import argparse


parser = argparse.ArgumentParser()

parser.add_argument("--mode", type=str, default="train")
parser.add_argument("--train_data", type=str, default="data/train_data", help="path of train data")
parser.add_argument("--test_data", type=str, default="data/test_data", help="path of test data")
parser.add_argument("--model_dir", type=str, default="checkpoints", help="path to save checkpoints")
parser.add_argument("--model_path", type=str, default="checkpoints/epoch-4-2021-04-28-15-10.pth",
                    help="pretrained model path")

parser.add_argument("--use_gpu", action='store_true', default=True)
parser.add_argument("--epoch_num", type=int, default=10, help="total train epochs")
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--num_workers", type=int, default=8)
parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
parser.add_argument("--lr_decay_freq", type=int, default=2, help="learning rate decay frequency")
parser.add_argument("--weight_decay", type=float, default=0.0005)

parser.add_argument("--val_start", type=int, default=2, help="start validation epoch")
parser.add_argument("--val_freq", type=int, default=2, help="validation frequency")
parser.add_argument("--test_start", type=int, default=5, help="start test epoch")
parser.add_argument("--test_freq", type=int, default=5, help="test frequency")
parser.add_argument("--print_freq", type=int, default=20, help="print frequency")

parser.add_argument("--class_num", type=int, default=4032, help="total num of classes")
parser.add_argument("--img_size", type=int, default=60)
parser.add_argument("--pad_size", type=int, default=4)


config = parser.parse_args()
