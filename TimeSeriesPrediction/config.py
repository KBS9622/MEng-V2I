import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--feature', type=str, default='gps_speed', help='extract which feature for prediction')
parser.add_argument('--network', type=str, default='lstm', choices=['rnn', 'lstm', 'gru', 'seq2seq'])
parser.add_argument('--transfer_learning', type=bool, default=False, help='transfer learning')

parser.add_argument('--which_data', type=str, default='12_sep_oct_nov_nov_dec.csv', help='which data to use')
parser.add_argument('--combined_path', type=str, default='./combined/', help='combined data path')
parser.add_argument('--weights_path', type=str, default='./results/weights/', help='weights path')
parser.add_argument('--numpy_path', type=str, default='./results/numpy/', help='numpy files path')
parser.add_argument('--plots_path', type=str,  default='./results/plots/', help='plots path')

parser.add_argument('--valid_start', type=str, default='2017-12-12 00:00:00', help='validation start date')
parser.add_argument('--test_start', type=str, default='2017-12-23 00:00:00', help='test start date')

parser.add_argument('--num_epochs', type=int, default=500, help='total epoch')
parser.add_argument('--print_every', type=int, default=100, help='print statistics for every n iteration')
parser.add_argument('--window', type=int, default=20, help='the number of window')

parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--lr_decay_rate', type=float, default=0.5, help='decay learning rate')
parser.add_argument('--lr_decay_every', type=int, default=1000, help='decay learning rate for every n epoch')
parser.add_argument('--lr_scheduler', type=str, default='cosine', help='learning rate scheduler', choices=['step', 'plateau', 'cosine'])

parser.add_argument('--input_size', type=int, default=1, help='input_size')
parser.add_argument('--hidden_size', type=int, default=2, help='hidden_size')
parser.add_argument('--num_layers', type=int, default=1, help='num_layers')
parser.add_argument('--num_classes', type=int, default=1, help='num_classes')

parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
parser.add_argument('--train_split', type=float, default=0.8, help='train_split')
parser.add_argument('--test_split', type=float, default=0.1, help='test_split')

parser.add_argument('-f')
config = parser.parse_args()


if __name__ == "__main__":
    print(config)