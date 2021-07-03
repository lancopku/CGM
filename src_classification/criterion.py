import torch
import torch.nn.functional as F

PAD = 0


def compute_right_num(outputs, targets):
    right_num = 0
    pre_total = 0
    rec_total = 0
    for output, target in zip(outputs, targets):
        output_result = transform_decode_sequence(output)
        target_result = transform_decode_sequence(target)
        pre_total += len(set(output_result))
        rec_total += len(target_result)
        right_num += len(set(output_result).intersection(set(target_result)))
    return right_num, pre_total, rec_total


def transform_decode_sequence(sequence):
    result = []
    for w in sequence:
        if w > 2:  # not in the range of padding, start, end
            result.append(w.item())
        elif w == 2:  # encounters END
            return result
    return result


def compute_loss(hidden_outputs, targets, mask):
    # hidden_outputs: (span, node, label)
    assert hidden_outputs.size(1) == targets.size(1) and hidden_outputs.size(0) == targets.size(0)
    outputs = hidden_outputs.contiguous().view(-1, hidden_outputs.size(2))
    targets = targets.contiguous().view(-1)
    weight = torch.ones(outputs.size(-1))
    weight[PAD] = 0
    weight = weight.to(outputs.device)
    loss = F.nll_loss(torch.log(outputs), targets, weight=weight, reduction='sum')
    return loss
