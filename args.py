import argparse


def args():
    args = argparse.ArgumentParser()

    args.add_argument('--feature_dimension',default=10,type=int)
    args.add_argument('--max_rul',default=125,type=int)
    args.add_argument('--time_length',default=50,type=int)
    args.add_argument('--batch_size',default=64,type=int)
    args.add_argument('--data_sub',default=1,type=int)
    args.add_argument('--epoch', default=51, type = int)
    # args.add_argument('--epoch', default=1, type=int)
    # args.add_argument('--show_interval',default=5, type = int)
    args.add_argument('--show_interval', default=1, type=int)
    args.add_argument('--save_name',default=None, type = str)
    args.add_argument('--num_nodes',default=7, type = int)
    args.add_argument('--sub_idx',default=1, type = int)

    # for HTGNN
    args.add_argument('--model_name', type=str, default='thgnn')
    # args.add_argument('--device', type=int, default=0)
    args.add_argument('--device', type=int, default=0)
    args.add_argument('--lr', type=float, default=5e-4)
    # args.add_argument('--lr', type=float, default=1e-4)
    # args.add_argument('--hid_dim', type=int, default=64)
    args.add_argument('--hid_dim', type=int, default=16)
    args.add_argument('--cor_embed_dim', type=int, default=16)
    args.add_argument('--num_rnn_layers', type=int, default=1)
    args.add_argument('--experiName', type=str, default='het_bottle')
    args.add_argument('--data_name', type=str, default='CMPS')
    # args.add_argument('--data_name', type=str, default='NCMPS')
    args.add_argument('--num_node_type', type=int, default=8)
    args.add_argument('--typeFunc_dim', type=int, default=16)
    # args.add_argument('--typeFunc_dim', type=int, default=4)
    args.add_argument('--seed', type=int, default=1)

    args.add_argument('--bottle', type=int, default=1)
    # args.add_argument('--bottle', type=int, default=0)
    args.add_argument('--sig', type=int, default=1)
    args.add_argument('--lm', type=int, default=1, help='whether do layer norm')
    args.add_argument('--film_reg', type=int, default=0)
    args.add_argument('--reg_coef', type=float, default=0.001)

    args.add_argument('--l2reg', type=float, default=0)
    # args.add_argument('--l2reg', type=float, default=5e-3)


    return args.parse_args()
