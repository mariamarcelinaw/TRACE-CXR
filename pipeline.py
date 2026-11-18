"""TRACE-CXR archived-score utilities (pure Python, CPU, no fitting).

These routines are a consolidated *audit implementation*, not a byte-for-byte
copy of the historical PyTorch training/inference implementation. Restricted
medical images and checkpoint binaries are not distributed.
"""
from __future__ import annotations
import csv
import math
from pathlib import Path
from statistics import mean, median, stdev


def rows(path: str | Path):
    with Path(path).open(encoding='utf-8', newline='') as stream:
        yield from csv.DictReader(stream)


def sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    ex = math.exp(x)
    return ex / (1.0 + ex)


def logit(p: float, epsilon: float = 1e-12) -> float:
    p = min(1.0 - epsilon, max(epsilon, p))
    return math.log(p / (1.0 - p))


def calibrate(p: float, temperature: float) -> float:
    if temperature <= 0:
        raise ValueError('Temperature must be positive')
    return sigmoid(logit(p) / temperature)


def gate_features(p_inception: float, p_swin: float) -> tuple[float, ...]:
    for value in (p_inception, p_swin):
        if not (0 <= value <= 1) or not math.isfinite(value):
            raise ValueError('Branch probability must be finite in [0,1]')
    def h(p):
        p = min(1 - 1e-12, max(1e-12, p))
        return -p * math.log(p) - (1-p) * math.log(1-p)
    return (p_inception, p_swin, h(p_inception), h(p_swin), abs(p_inception - p_swin))


def fuse(p_inception: float, p_swin: float, method: str, controls: dict,
         gate_alpha: float | None = None) -> float:
    gate_features(p_inception, p_swin)
    if method == 'inception_v3': return p_inception
    if method == 'swin_b': return p_swin
    if method == 'mean_fusion': return (p_inception + p_swin)/2
    if method == 'const_alpha':
        alpha = float(controls['const_alpha_inception'])
        if not 0 <= alpha <= 1: raise ValueError('Constant alpha outside [0,1]')
        return alpha*p_inception + (1-alpha)*p_swin
    if method == 'gated_fusion':
        if gate_alpha is None: raise ValueError('gated_fusion requires frozen gate output')
        if not 0 <= gate_alpha <= 1: raise ValueError('Gate alpha outside [0,1]')
        return gate_alpha*p_inception + (1-gate_alpha)*p_swin
    if method == 'logit_stack':
        stack = controls['logit_stack']
        return sigmoid(stack['intercept'] + stack['coef_logit_inception']*logit(p_inception)
                       + stack['coef_logit_swin']*logit(p_swin))
    raise ValueError(f'Unknown method: {method}')


def auc(y: list[int], scores: list[float]) -> float:
    if len(y) != len(scores) or not y: raise ValueError('y and scores length mismatch')
    if set(y) - {0, 1}: raise ValueError('Labels must be 0 or 1')
    pos = sum(y); neg = len(y)-pos
    if not pos or not neg: raise ValueError('ROC AUC undefined for single class')
    pairs = sorted(zip(scores, y), key=lambda a:a[0]); rank_sum=0.0; i=0
    while i < len(pairs):
        j=i+1
        while j < len(pairs) and pairs[j][0] == pairs[i][0]: j += 1
        rank_sum += sum(label for _,label in pairs[i:j]) * ((i+1+j)/2)
        i=j
    return (rank_sum - pos*(pos+1)/2)/(pos*neg)


def binary_metrics(y: list[int], probability: list[float], threshold: float) -> dict:
    if len(y)!=len(probability) or not y: raise ValueError('Mismatched arrays')
    if not 0 <= threshold <= 1: raise ValueError('Threshold outside [0,1]')
    if not all(math.isfinite(p) and 0 <= p <= 1 for p in probability):
        raise ValueError('Invalid probabilities')
    decisions=[int(p >= threshold) for p in probability]
    tp=sum(1 for a,b in zip(y,decisions) if a==b==1)
    tn=sum(1 for a,b in zip(y,decisions) if a==b==0)
    fp=sum(1 for a,b in zip(y,decisions) if a==0 and b==1)
    fn=sum(1 for a,b in zip(y,decisions) if a==1 and b==0)
    se=tp/(tp+fn) if tp+fn else None
    sp=tn/(tn+fp) if tn+fp else None
    epsilon=1e-12
    nll=-mean(a*math.log(max(epsilon,p))+(1-a)*math.log(max(epsilon,1-p)) for a,p in zip(y,probability))
    return {'n':len(y),'tp':tp,'tn':tn,'fp':fp,'fn':fn,'roc_auc':auc(y,probability),
            'sensitivity':se,'specificity':sp,'balanced_accuracy':(se+sp)/2 if se is not None and sp is not None else None,
            'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0,
            'brier':mean((a-p)**2 for a,p in zip(y,probability)),'nll':nll}


def reliability_bins(y: list[int], probability: list[float], n_bins: int = 15) -> list[dict]:
    """Equal-width reliability bins (same binning as ECE15)."""
    if len(y)!=len(probability) or not y: raise ValueError('Mismatched arrays')
    groups=[[] for _ in range(n_bins)]
    for label,p in zip(y,probability):
        groups[min(int(p*n_bins),n_bins-1)].append((label,p))
    out=[]
    for k,g in enumerate(groups):
        out.append({'bin':k+1,'n_images':len(g),
                    'mean_predicted':mean(p for _,p in g) if g else None,
                    'observed_rate':mean(a for a,_ in g) if g else None})
    return out


def ece_from_bins(bins: list[dict], n_total: int) -> float:
    return sum(b['n_images']/n_total*abs(b['observed_rate']-b['mean_predicted']) for b in bins if b['n_images'])


def summarize_timings(path: str | Path) -> dict:
    data=list(rows(path))
    if not data: raise ValueError('Empty timing CSV')
    if 'candidate' in data[0]:
        groups={}
        for r in data:
            key=r['candidate']; groups.setdefault(key,[]).append(float(r['total_end_to_end_ms']))
        if any((not math.isfinite(x) or x < 0) for v in groups.values() for x in v):
            raise ValueError('Invalid latency')
        return {k:{'n':len(v),'mean_ms':mean(v),'median_ms':median(v),
                   'sample_sd_ms':stdev(v) if len(v)>1 else None} for k,v in sorted(groups.items())}
    if 'total' in data[0]:
        vals=[float(r['total']) for r in data]
        return {'historical_two_backbone':{'n':len(vals),'mean_ms':mean(vals),
                'median_ms':median(vals),'sample_sd_ms':stdev(vals) if len(vals)>1 else None}}
    raise ValueError('Unsupported timing format')
