#from offlinerllib.utils.functional import discounted_cum_sum
import torch

def discounted_cum_sum(seq, discount):
    seq = seq.copy()
    for t in reversed(range(len(seq)-1)):
        seq[t] += discount * seq[t+1]
    return seq