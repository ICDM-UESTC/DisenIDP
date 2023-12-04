import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-preprocess', type=bool, default=False, help="preprocess dataset")    #when you first run code, you should set it to true.

##### model parameters
parser.add_argument('-data_name', type=str, default='twitter', choices=['weibo22', 'memes', 'twitter', 'douban'], help="dataset")
parser.add_argument('-epoch', type=int, default=200)
parser.add_argument('-max_lenth', type=int, default=200)
parser.add_argument('-batch_size', type=int, default=128)
parser.add_argument('-posSize', type=int, default=8, help= "the position embedding size")
parser.add_argument('--embSize', type=int, default=64, help='embedding size')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--layer', type=float, default=3, help='the number of layer used')
parser.add_argument('--beta', type=float, default= 0.001, help='ssl graph task maginitude')  
parser.add_argument('--beta2', type=float, default=0.005, help='ssl cascade task maginitude')  
parser.add_argument('--window', type=int, default=10, help='window size')  
parser.add_argument('-n_warmup_steps', type=int, default=1000)
parser.add_argument('-dropout', type=float, default=0.2)

#####data process
parser.add_argument('-train_rate', type=float, default=0.8)
parser.add_argument('-valid_rate', type=float, default=0.1)

###save model
parser.add_argument('-save_path', default= "./checkpoint/")
parser.add_argument('-patience', type=int, default=5, help="control the step of early-stopping")


