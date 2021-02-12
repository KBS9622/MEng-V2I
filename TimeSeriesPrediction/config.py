import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=1234, help='fix seed for reproducibility')
parser.add_argument('--batch_size', type=int, default=256, help='mini-batch size')
parser.add_argument('--num_epochs', type=int, default=10, help='total epoch')
parser.add_argument('--window', type=int, default=180, help='the number of window, unit : second')

parser.add_argument('--plot', type=bool, default=False, help='plot graph or not')
parser.add_argument('--preprocess', type=bool, default=True, help='remove outliers')
parser.add_argument('--resample', type=bool, default=True, help='resample')

parser.add_argument('--feature', type=str, default='speed', help='extract which feature for prediction')
parser.add_argument('--network', type=str, default='lstm', choices=['dnn', 'cnn', 'rnn', 'lstm', 'gru', 'recursive', 'attentional'])
parser.add_argument('--transfer_learning', type=bool, default=False, help='transfer learning')

parser.add_argument('--which_data', type=str, default='12_sep_oct_nov_nov_dec.csv', help='which data to use')
parser.add_argument('--combined_path', type=str, default='./combined/', help='combined data path')
parser.add_argument('--weights_path', type=str, default='./results/weights/', help='weights path')
parser.add_argument('--numpy_path', type=str, default='./results/numpy/', help='numpy files path')
parser.add_argument('--plots_path', type=str,  default='./results/plots/', help='plots path')

parser.add_argument('--valid_start', type=str, default='2017-12-12 00:00:00', help='validation start date')
parser.add_argument('--test_start', type=str, default='2017-12-23 00:00:00', help='test start date')

parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--lr_decay_rate', type=float, default=0.5, help='decay learning rate')
parser.add_argument('--lr_decay_every', type=int, default=100, help='decay learning rate for every n epoch')
parser.add_argument('--lr_scheduler', type=str, default='cosine', help='learning rate scheduler', choices=['step', 'plateau', 'cosine'])

parser.add_argument('--input_size', type=int, default=1, help='input_size')
parser.add_argument('--hidden_size', type=int, default=10, help='hidden_size')
parser.add_argument('--num_layers', type=int, default=1, help='num_layers')
parser.add_argument('--output_size', type=int, default=1, help='output_size')
parser.add_argument('--bidirectional', type=bool, default=False, help='bidirectional or not')
parser.add_argument('--key', type=int, default=8, help='key')
parser.add_argument('--query', type=int, default=8, help='query')
parser.add_argument('--value', type=int, default=8, help='value')

parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])

parser.add_argument('-f')
config = parser.parse_args()


if __name__ == "__main__":
    print(config)