import argparse
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
from attn_result import AttnResult

dependencies = ["nsubj", "obj", "amod","advmod","nmod", "det"]


def eval_heads_cats(result : AttnResult, dep, cat, pos_lab):
    """Compare scores for a given deps, for all labels in the given category
     pos lab in (0,1) indicates whether the studied dep is for the head or the tail of the dep"""
    dirs = ["->", "<-"]
    pos_text = ["tail", "head"][pos_lab]
    score = result.score_dep[..., 1] / result.score_dep[..., 0]  # score : (times where attn captured {dep})/(nb of {dep})
    occurences = result.score_dep[0, 0, :, 0, 0]  # count occurences of each dependencies
    baseline_pos = result.distr_dep.max(axis=1) / occurences  # Baseline :
    # (times where w_(n+i) is {dep} of w_n)/{times where w_n has a dep}
    baseline_random = result.random_baseline
    L, H, _, _ = score.shape
    dep_i = dep_list.index(dep)
    for j in range(2):
        fig, axes = plt.subplots(2, L // 2, sharey=True, sharex=True)
        axes.resize((L * 2,))
        for l in range(L):
            ax: plt.Subplot = axes[l]
            dl = 0.5*(len(result.cats[cat])+2)
            ax.bar(np.arange(H)*dl, score[l, :, dep_i, j], label='Whole data')
            delta = 0.5
            for lab in result.cats[cat]:
                ax.bar(np.arange(H)*dl+delta, result.cats_scores[cat][lab][l,:,dep_i,j, pos_lab] /result.cats_occurences[cat][lab][dep_i, pos_lab]
                       , label=f"{lab} ({pos_text})")
                delta+=0.5
            ax.set_xticks(np.arange(H)*dl)
            ax.set_xticklabels(np.arange(H))
            ax.set_title(f"Layer {l}")
            ax.set_xlabel("Head")
            ax.hlines(baseline_pos[dep_i], 0, H*dl, label="Positional high baseline")
            ax.hlines(baseline_random[dep_i], 0, H*dl, label="Base score for random choice", colors="green")

        plt.legend()
        fig.suptitle(f"Attention scores\n{dep}\n{dirs[j]}")
        fig.set_size_inches((14, 7), forward=False)
        fig.savefig(f"{args.dest}/score_{dep}_{cat}_{pos_text}_{j}.png")
        plt.close(fig)

def bar_pos_freq(result : AttnResult):
    """Plots the distribution of head position relative to tail for each dep."""
    occurences = result.score_dep[0,0,:,0,0]
    pos_freq = np.apply_along_axis(lambda x: x / occurences, axis=0, arr=pos)
    fig, axes = plt.subplots(6,1, sharey= "all", sharex="all")
    for i, ax in enumerate(axes):
        ax.bar(np.arange(-delta, delta+1),pos_freq[i,:])
        ax.set_xlabel(dep_list[i])
        ax.set_xticks(np.arange(-delta, delta+1))
    fig.suptitle("Distribution des dependances syntaxiques")
    fig.set_size_inches((14, 14), forward=False)
    fig.savefig(f"{args.dest}/distr_dep.png")

#to evaluate a head:
# passing = np.apply_fnc_axis attn[l,h,:,i,1] > baseline
def plot_score(attn : AttnResult ):
    """Plots the attention-on-dependecies score, layer by layer.
    Not very useful"""
    score = attn.score_dep[...,1] /attn.score_dep[...,0]
    L,H,_,_ = score.shape
    d1,d2 = rect(2*H)
    for l in range(L):
        fig, axes = plt.subplots(d1,d2 , sharey=True)
        axes.resize((2*H,))
        for h in range(H):
            for i in range(2):
                ax = axes[h*2+i]
                ax.bar(np.arange(D), score[l,h,:,i])
                ax.set_xticks(np.arange(D))
                ax.set_xticklabels(dep_list)
                ax.set_title(f"Head {h}"+("->" if not i else "<-"))
                plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                         rotation_mode="anchor")
        fig.suptitle(f"Attention scores\nLayer {l}")
        fig.set_size_inches((14, 7), forward=False)
        fig.savefig( f"{args.dest}/score_layer{l}.png")

def plot_score_dep(result : AttnResult, writer):
    "Plots the attention-on-dependecies score, dep by dep"
    #score, baseline_pos, baseline_random
    dirs = ["->", "<-"]
    score = result.score_dep[...,1] /result.score_dep[...,0]#score : (times where attn captured {dep})/(nb of {dep})
    occurences = result.score_dep[0,0,:,0,0]# count occurences of each dependencies
    baseline_pos =  result.distr_dep.max(axis=1)/occurences #Baseline :
    # (times where w_(n+i) is {dep} of w_n)/{times where w_n has a dep}
    baseline_random = result.random_baseline
    L, H, _, _ = score.shape
    for i,d in enumerate(dep_list):
        bl = baseline_pos[i]
        writer.write(f"{d} : baseline {bl}\n")
        for j in range(2):
            fig, axes = plt.subplots(2,L//2, sharey=True, sharex=True)
            axes.resize((L*2,))
            for l in range(L):
                ax :plt.Subplot= axes[l]
                ax.bar(np.arange(H), score[l,:,i,j])
                ax.set_xticks(np.arange(H))
                ax.set_xticklabels(np.arange(H))
                ax.set_title(f"Layer {l}")
                ax.set_xlabel("Head")
                ax.hlines(baseline_pos[i], 0, H, label = "Positional high baseline")
                ax.hlines(baseline_random[i], 0,H, label= "Base score for random choice", colors="green")
                for h in range(H):
                    if score[l,h,i,j] > 1.1 *bl:
                        writer.write(f"{l},{h} ({dirs[j]}) passes Voita with {score[l,h,i,j]}\n")
                    elif score[l,h,i,j] > bl:
                        writer.write(f"{l},{h} ({dirs[j]}) above baseline with {score[l,h,i,j]}\n")
            plt.legend()
            fig.suptitle(f"Attention scores\n{d}\n{dirs[j]}")
            fig.set_size_inches((14, 7), forward=False)
            fig.savefig(f"{args.dest}/score_{d}_{j}.png")
            plt.close(fig)
        writer.write("\n")


def eval_heads_pos(result : AttnResult):
    """Plots the score of all heads for +/- 1 relative position"""
    nmax = result.N_words - 2* result.N_sent #*2 because 1) we dont count neighbours of <s> and 2) we dont count <s>
    score = result.score_pos/nmax
    L,H,_ = score.shape
    for i in (0,1):
        fig = plt.figure()
        plt.bar(np.arange(L*H), score[:,:,i].flat)
        ax = plt.gca()
        ax.set_xticks(np.arange(L*H))
        ax.set_xticklabels([f"{l}\n{h}" for l in range(L) for h in range(H)])
        ax.hlines(0.9,0, L*H)
        ax.hlines(result.distr_baseline[i*2-1], 0, L*H, label = "Random baseline", color='red')
        ax.set_ylabel("%")
        fig.suptitle(f"Positional score in {i*2-1}")
        fig.legend()
        fig.set_size_inches((14, 7), forward=False)
        fig.savefig(f"{args.dest}/score_positional_{i*2-1}.png")
        plt.close(fig)

def eval_heads_rare(result : AttnResult):
    """Plots the score of all heads for attention on rarest tokens"""
    nmax = result.N_words - result.N_sent
    score = result.score_rare/nmax
    baseline = result.rarity_baseline
    L,H = score.shape
    fig = plt.figure()
    plt.bar(np.arange(L * H), score.flat)
    ax = plt.gca()
    ax.set_xticks(np.arange(L * H))
    ax.set_xticklabels([f"{l}\n{h}" for l in range(L) for h in range(H)])
    ax.hlines(0.9, 0, L * H)
    ax.hlines(baseline, 0, L*H, label="Random baseline", color="red")
    ax.set_ylabel("%")
    fig.suptitle(f"Rarity score")
    fig.set_size_inches((14, 7), forward=False)
    fig.savefig(f"{args.dest}/score_rare.png")
    plt.close(fig)

def plot_attn_distrib(result : AttnResult):
    """Plots the distribution of the most attendend token as a function of its relative position"""
    attn_distr = result.distr_attn/(result.N_words - result.N_sent)
    fig = plt.figure()
    plt.bar(np.arange(-delta, delta+1), attn_distr*100, label="Relative position of most attended token")
    plt.plot(np.arange(-delta, delta+1), result.distr_baseline*100, label="Random baselines")
    ax = plt.gca()
    ax.set_xticks(np.arange(-delta, delta+1))
    ax.set_xticks(np.arange(-delta, delta + 1))
    ax.set_ylabel("%")
    plt.title("Distribution du tokens le plus attendu")
    plt.legend()
    fig.set_size_inches((14, 7), forward=False)
    fig.savefig(f"{args.dest}/attn_distrib.png")
    plt.close(fig )

def factor_iter(N):
    for i in range(1,int(np.sqrt(N))+1):
        if not N%i:
            yield i
        i+=1

def rect(N):
    """Returns i,j such that i*j = N and i,j is roughly a nice rectangle"""
    r = 1/3
    m = 1
    i_res=1
    for i in factor_iter(N):
        r2 = i/(N/i)
        if abs(r2-r) > m:
            return i, N//i
        else:
            i_res = i
            m = abs(r2-r)
    return i_res, N//i


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("result", help="Pickled AttnResult object")
    parser.add_argument("dest", help="Folder to save results to")
    args = parser.parse_args()

    try :
        os.mkdir(args.dest)
    except OSError:
        pass

    with open(args.result, "rb") as reader:
        attn_result : AttnResult = pickle.load(reader)
    print(f"{attn_result.N_sent} sentences processed")
    attn = attn_result.score_dep #[L,H,D,2,2]
    print("Attention : ", attn.shape)
    pos = attn_result.distr_dep #[D, 2 delta+1]
    print("Distribution of dep: ", pos.shape)
    dep_list = attn_result.dependencies

    D, delta = pos.shape
    delta = (delta -1 )//2
    assert D == len(dep_list)

    # plots graphs for heads scores with regard to labels
    eval_heads_cats(attn_result, "nsubj", "nb",0)
    eval_heads_cats(attn_result, "amod", "gen",1)
    eval_heads_cats(attn_result, "amod", "nb",1)
    eval_heads_cats(attn_result, "amod", "gen", 0)
    eval_heads_cats(attn_result, "amod", "nb", 0)
    eval_heads_cats(attn_result, "nsubj", "gen",0)
    eval_heads_cats(attn_result, "nsubj", "voice", 1)
    eval_heads_cats(attn_result, "obj", "voice", 1)

    bar_pos_freq(attn_result)
    with open(args.dest+"/scores.txt", "w") as writer :
        plot_score_dep(attn_result, writer)
    eval_heads_pos(attn_result)
    eval_heads_rare(attn_result)
    plot_attn_distrib(attn_result)

    print("Finished")



