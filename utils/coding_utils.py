from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, confusion_matrix

def print_args(parser, args_parsed):
    message = ''
    message += '-------------------------------- Args ------------------------------\n'
    for k, v in sorted(vars(args_parsed).items()):
        comment = ''
        default = parser.get_default(k)
        if v != default:
            comment = '\t[default: %s]' % str(default)
        message += '{:>35}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '-------------------------------- End ----------------------------------'
    print(message)

class AverageMeter():
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def performance(preds, trues):
    accuracy = accuracy_score(trues, preds)
    f1_macro = f1_score(trues, preds, average='macro')

    cm = confusion_matrix(trues, preds)
    accuracy_per_class = cm.diagonal() / cm.sum(axis=1)
    kappa = cohen_kappa_score(trues, preds)
    performance_dict = {'acc': accuracy,
                        **{f'acc_class_{i}': acc for i, acc in enumerate(accuracy_per_class.tolist())},
                         'f1': f1_macro, 'cm': cm, "kappa": kappa}
    return performance_dict

