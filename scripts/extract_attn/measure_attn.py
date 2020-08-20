"""Script to measure attention on a corpus and save different metrics"""

import numpy as np
import extract_model as ex
import argparse
import datetime
from opennmt import models
import os

def main(quantiles):
    model = ex.get_model(args.M, args.C, model_type=models.TransformerBase)
    print(f"Reading {args.corp}.")
    ds = ex.get_dataset(model, args.corp,mode='inference' )
    L,H = model.num_layers, model.num_heads
    stack = []
    print("Measuring attention...")
    for i, data in enumerate(ds):
        _, attn  = model(data, return_attn=True, training=False)
        L2,H2,S,S2 = attn.shape
        stack.append(attn.numpy().reshape((L, H, S * S)))
        if args.T:
            print(data["tokens"])
            print("Attn : ", attn.shape)
            assert L2==L
            assert H2==H
            assert S2==S
            print()
            if args.T2:
                break
        if not i%(args.max//50) :
            print_log(i, data["tokens"], stack)
        if args.part is not None and not i % args.part :
            save_stats(np.concatenate(stack, axis=2), args.dest+f".{i}", quantiles)
        if i>args.max:
            break
    total = np.concatenate(stack, axis=2)
    print(f"Done : Measured attention on {total.shape[2]} token pairs.")
    save_stats(total, args.dest, quantiles)


def save_stats(A : np.ndarray, dest, quant = (0.1,0.25,0.5,0.75,0.9)):
    try:
        os.mkdir(dest)
    except OSError:
        pass
    mean = A.mean(axis=2)
    std = A.std(axis=2)
    quantiles = np.quantile(A, quant, axis = 2)
    np.save(dest+"/mean", mean)
    np.save(dest+"/std", std)
    np.save(dest+"/quantiles", quantiles)
    np.save(dest+"/quantiles.list", quant)


def print_log(i, s, stack):
    l,h, _ = stack[0].shape
    print(datetime.datetime.now(), i, f"size ~ 1e {int(np.log10(l*h*sum(d.shape[2] for d in stack)))}")
    print(" ".join( [t.decode() for t in s.numpy().flat] ) )
    print(flush=True)


quant = (0.1,0.25,0.5,0.75,0.9) #0.1, 0.5 and 0.9 are used in inject_tasks

if __name__ == '__main__':
    parser = argparse.ArgumentParser(f"A script to compute statistics on the attention of the heads of a model.\n"
                                     f"Computes the mean, std, and {quant} quantiles")
    parser.add_argument("corp",help="Tokens" )
    parser.add_argument("dest", help="Folder to save results to")
    parser.add_argument("-M", help = "Model", required=True)
    parser.add_argument("-C", help="Model configuration", required=True)
    parser.add_argument("-T", help="Testing", action="store_true")
    parser.add_argument("-T2", help="Stronger testing", action="store_true")
    parser.add_argument("--part", help="Save partial results this often", type = int, default=None)
    parser.add_argument("--max", help="Stops at this many iterations", type=int, default=20_000)

    args = parser.parse_args()

    args.T = args.T or args.T2
    main(quant)




