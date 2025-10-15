import datetime
from operator import itemgetter
import scipy.stats

def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))

    return str(datetime.timedelta(seconds=elapsed_rounded))

def computeMetric(validationQPP,validationMAP):

    predictedQPP = []
    ActualMAP = []
    for query, QPPScore in validationQPP.items():
        predictedQPP.append(float(round(QPPScore,4)))
        ActualMAP.append(float(validationMAP[query]))
    pearsonr, pearsonp = scipy.stats.pearsonr(ActualMAP, predictedQPP)
    kendalltau = scipy.stats.kendalltau(ActualMAP, predictedQPP)
    spearmanr = scipy.stats.spearmanr(ActualMAP, predictedQPP)

    return pearsonr, pearsonp, kendalltau.correlation, kendalltau.pvalue, spearmanr.correlation, spearmanr.pvalue
