def compute_recall(act, pred, topk=10):
    act_set = set(act)
    pred_set = set(pred[:topk])
    recall = len(act_set & pred_set) / float(len(act_set))
    return recall


def compute_mrr(act, pred):
    mrr = 0
    for a in act:
        if a in pred:
            mrr += 1 / (pred.index(a) + 1)
    mrr = mrr / len(act)
    return mrr
