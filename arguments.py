import argparse

def get_parser():
    parser = argparse.ArgumentParser(description='TTP-MORL')
    # GENERAL
    parser.add_argument('--test-instance-name',
                        type=str,
                        default="bar-n100-1",
                        help="instance's name for real testing")
    parser.add_argument('--test-num-vehicles',
                        type=int,
                        default=3,
                        help="num vehicles for test instance")
    parser.add_argument('--title',
                        type=str,
                        default="init-sop",
                        help="title for experiment")
    parser.add_argument('--device',
                        type=str,
                        default='cpu',
                        help='device to be used cpu or cuda (for gpu)')
    parser.add_argument('--max-epoch',
                        type=int,
                        default=1000,
                        help='maximum epoch training')
    parser.add_argument('--seed',
                        type=int,
                        default=1,
                        help='seed for random generator')
    parser.add_argument('--batch-size',
                        type=int,
                        default=256,
                        help="dataloader batch size")
    parser.add_argument('--num-training-samples',
                        type=int,
                        default=1000000,
                        help="dataset training num samples")
    parser.add_argument('--num-validation-samples',
                        type=int,
                        default=100000,
                        help="dataset validation num samples")
    parser.add_argument('--max-grad-norm',
                        type=int,
                        default=1,
                        help="gradient clipping")
    
    #PHN
    parser.add_argument('--ray-hidden-size',
                        type=int,
                        default=128,
                        help="phn ray hidden size")
    parser.add_argument('--num-ray',
                        type=int,
                        default=8,
                        help="num of rays")
    parser.add_argument('--omega',
                        type=int,
                        default=10,
                        help="max patience")
    
    parser.add_argument('--ld',
                        type=float,
                        default=0,
                        help="ld cosine sim penalty")

    # agent base or AM
    parser.add_argument('--lr',
                        type=float,
                        default=1e-4,
                        help="learning rate")
    parser.add_argument('--n-heads',
                        type=int,
                        default=8,
                        help="num heads multihead attention")
    parser.add_argument('--n-gae-layers',
                        type=int,
                        default=3,
                        help="num layers of encoder")
    parser.add_argument('--embed-dim',
                        type=int,
                        default=128,
                        help="embedding dim across model")
    parser.add_argument('--gae-ff-hidden',
                        type=int,
                        default=128,
                        help="size of hidden ff in gae")
    parser.add_argument('--tanh-clip',
                        type=int,
                        default=10,
                        help="tanh clip")
    return parser
