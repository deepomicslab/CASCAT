import os
import argparse
import yaml
from utils.data_loader import *
from infer import InferExperiment
from cluster import Experiment


def update_args(args):
    if os.path.exists(args.yml_path):
        with open(args.yml_path, 'r') as f:
            dict = yaml.load(f, Loader=yaml.FullLoader)
        for key in dict.keys():
            args.__dict__[key] = dict[key]
    return args


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yml_path', type=str, default='./config/tree3.yml')
    parser.add_argument('--mode', type=str, default='infer')
    parser.add_argument('--verbose', type=bool, default=False)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--plot', type=str, default='tree_mode',
                        help='tree_mode, emb, st_emb, pesodutime, st_pesodutime, ground_truth')
    args = parser.parse_args()
    args = update_args(args)
    return args


def run_train(args):
    if args.learned_graph not in args.clu_dir:
        args.clu_dir = os.path.join(args.clu_dir, args.learned_graph)
    exp = Experiment(args)
    args.emb_path, ari, ami, args.job_dir = exp.train(verbose=args.verbose)
    print('Percent:{:.2f} Mean ARI: {:.5f}, Mean AMI: {:.5f}'.format(args.percent, np.mean(ari), np.mean(ami)))
    return args


def run_infer(args):
    infer_exp = InferExperiment(args)
    IM, KT, SR = infer_exp.infer()
    if args.plot is not None:
        infer_exp.plot(args.plot, show=True, colors='tab10')
    return IM, KT, SR


if __name__ == '__main__':
    args = parse_arguments()
    if args.mode == 'train':
        run_train(args)
        print(f"Job dir: {args.job_dir}")
    else:
        if not os.path.exists(args.job_dir):
            print(f"Job dir: {args.job_dir} not found.")
        if 'trial0_ARI-inf.npy' in os.listdir(args.job_dir):
            args.emb_path = os.path.join(args.job_dir, 'trial0_ARI-inf.npy')
        else:
            if not os.path.exists(args.emb_path):
                print(f"Embedding path: {args.emb_path} not found.")
        IM, KT, SR = run_infer(args)
