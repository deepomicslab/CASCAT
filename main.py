import os
import argparse
import yaml
import numpy as np
import pandas as pd
import datetime
from infer import InferExperiment
from cluster import Experiment


def update_args(args):
    if os.path.exists(args.YAML):
        with open(args.YAML, 'r') as f:
            dict = yaml.load(f, Loader=yaml.FullLoader)
        for key in dict.keys():
            args.__dict__[key] = dict[key]
    return args


def run_train(args, verbose=True):
    if args.learned_graph not in args.clu_dir:
        args.clu_dir = os.path.join(args.clu_dir, args.learned_graph)
    best_aris, best_amis, seed_records, ari_records, ami_records, best_emb = [], [], [], [], [], None
    for trial in range(args.ntrials):
        print(f"Trial {trial}/{args.ntrials}")
        args.seed = trial
        exp = Experiment(args)
        seeds, aris, amis = exp.train(verbose=verbose)
        seed_records.extend(seeds)
        ari_records.extend(aris)
        ami_records.extend(amis)
        idx = np.argmax(aris)
        best_aris.append(aris[idx])
        best_amis.append(amis[idx])
    df_record = pd.DataFrame({'trial': seed_records, 'ARI': ari_records, 'AMI': ami_records})
    df_record.to_csv(f'{args.job_dir}/training_record.csv', index=False)
    idx = np.argmax(ari_records)
    best_seed, best_ari, best_ami = seed_records[idx], ari_records[idx], ami_records[idx]
    args.emb_path = f'{args.job_dir}/trial{str(best_seed)}_ARI{best_ari:.5f}.npy'
    print('Percent:{:.2f} Best ARI: {:.5f}, Best AMI: {:.5f}'.format(args.percent, np.mean(best_aris),
                                                                     np.mean(best_amis)))


def run_infer(args):
    infer_exp = InferExperiment(args)
    IM, OT, KT, SR = infer_exp.infer()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    infer_exp.adata.write(os.path.join(args.output_dir, 'data_processed.h5ad'))
    return IM, OT, KT, SR


def run_sorted_infer(args):
    dir = args.clu_dir + '/'
    for root, dirs, files in os.walk(dir):
        target_dir = dir + dirs[0]
        args.job_dir = target_dir
        for root, dirs, files in os.walk(target_dir):
            f = [f.split('ARI') for f in files if f.endswith('.npy')]
            f = sorted(f, key=lambda x: float(x[1].split('.')[0]), reverse=True)[0][0]
            target_f = [ff for ff in files if ff.startswith(f)][0]
            args.emb_path = os.path.join(target_dir, target_f)
            break
        break
    print("Infer from:", args.emb_path)
    infer_exp = InferExperiment(args)
    IM, OT, KT, SR = infer_exp.infer()
    return IM, OT, KT, SR


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--YAML', type=str, default='./config/tree1.yml')
    parser.add_argument('--mode', type=str, default='train', help='train or infer')
    parser.add_argument('--verbose', type=bool, default=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--plot', type=str, default='emb',
                        help='tree_mode, emb, st_emb, pesodutime, st_pesodutime, ground_truth')
    args = parser.parse_args()
    args = update_args(args)
    return args


if __name__ == '__main__':
    args = parse_arguments()
    if args.mode == 'train':
        args.job_dir = args.clu_dir + '/' + str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        run_train(args)
        print(f"Job dir: {args.job_dir}")
    if args.mode == 'train':
        IM, OT, KT, SR = run_infer(args)
    else:
        IM, OT, KT, SR = run_sorted_infer(args)
    print(IM, OT, KT, SR)
