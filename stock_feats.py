"""
Input
ohlcvs: torch.FloatTensor, (N, T, 5), OHLCV for N stocks in T time units

Output
feats: torch.FloatTensor, (N, T, num_feats)

difference, rate_of_change:
    计算当前时间t与t-interval_size时刻的差值/ROC
    t<interval_size时，计算t与0时刻的差值
moving_average, z_score, percentiles:
    计算从t-window_size+1时刻到当前时间t这一窗口内的值
    t<window_size-1时，计算从0到t时刻这一窗口内的值
exponential_moving_average:
    0到window_size-1，使用simple moving average
    从window_size起，根据前一时刻EMA计算当前EMA
    (https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/ema)
    (https://www.investopedia.com/ask/answers/122314/what-exponential-moving-average-ema-formula-and-how-ema-calculated.asp)
"""

import torch

def compute_feats(input_feats, eps=1e-6):
    def log(inputs):
        return torch.log(inputs + eps)

    def rate_of_change(inputs, interval_sizes):
        feats = []
        N, T, _ = inputs.size()
        for interval_size in interval_sizes:
            assert interval_size < T
            _feats = inputs.clone()
            _feats[:,interval_size:,:] = (_feats[:,interval_size:,:] - _feats[:,:-interval_size,:]) / (_feats[:,:-interval_size,:] + eps)
            _feats[:,:interval_size,:] = 0
            feats.append(_feats)
        return torch.cat(feats, dim=-1)

    def difference(inputs, interval_sizes):
        feats = []
        N, T, _ = inputs.size()
        for interval_size in interval_sizes:
            assert interval_size < T
            _feats = inputs.clone()
            _feats[:,interval_size:,:] = _feats[:,interval_size:,:] - _feats[:,:-interval_size,:]
            _feats[:,:interval_size,:] = 0
            feats.append(_feats)
        return torch.cat(feats, dim=-1)

    def moving_average(inputs, window_sizes):
        feats = []
        N, T, _ = inputs.size()
        for window_size in window_sizes:
            _feats = inputs.clone()
            for t in range(T):
                window = inputs[:,max(t-window_size+1,0):t+1,:]
                _feats[:,t,:] = window.mean(dim=-2)
            feats.append(_feats)
        return torch.cat(feats, dim=-1)

    def exponential_moving_average(inputs, window_sizes):
        feats = []
        N, T, _ = inputs.size()
        for window_size in window_sizes:
            assert window_size <= T
            _feats = inputs.clone()
            multiplier = 2 / (window_size + 1)
            for t in range(window_size):
                window = inputs[:,:t+1,:]
                _feats[:,t,:] = window.mean(dim=-2)
            for t in range(window_size, T):
                _feats[:,t,:] = inputs[:,t,:] * multiplier + _feats[:,t-1,:] * (1 - multiplier)
            feats.append(_feats)
        return torch.cat(feats, dim=-1)

    def z_scores(inputs, window_sizes):
        feats = []
        N, T, _ = inputs.size()
        for window_size in window_sizes:
            _feats = inputs.clone()
            for t in range(T):
                window = inputs[:,max(t-window_size+1,0):t+1, :]
                _mean = window.mean(dim=-2)
                _std = window.std(dim=-2)
                _feats[:,t,:] -= _mean
                _feats[:,t,:] /= (_std + eps)

            _feats[_feats != _feats] = 0.0 # Nan
            feats.append(_feats)
        return torch.cat(feats, dim=-1)

    def percentiles(inputs, window_sizes):
        feats = []
        N, T, _ = inputs.size()
        for window_size in window_sizes:
            _feats = inputs.clone()
            for t in range(T):
                window = inputs[:,max(t-window_size+1,0):t+1, :]
                _feats[:,t,:] = (window <= _feats[:,t:t+1,:]).sum(dim=-2) / window.size(-2)
            feats.append(_feats)
        return torch.cat(feats, dim=-1)

    interval_sizes = [1, 2, 4, 8]
    window_sizes = [2, 4, 8, 10]
    ohlcvs = input_feats[:,:,:5]
    # from ipdb import set_trace; set_trace()
    feats = torch.cat([
        ohlcvs,
        input_feats[:,:,5:],
        log(ohlcvs),
        difference(ohlcvs, interval_sizes),
        rate_of_change(ohlcvs, interval_sizes),
        moving_average(ohlcvs, window_sizes),
        exponential_moving_average(ohlcvs, window_sizes),
        z_scores(ohlcvs, window_sizes),
        percentiles(ohlcvs, window_sizes),
    ], dim=-1)
    # feature number = 5, 3, 7
    return feats

