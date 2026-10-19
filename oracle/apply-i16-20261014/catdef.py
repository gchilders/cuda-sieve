def cat(k):
    f, l = k
    if f != 'bench_kernels.cu': return None
    if 784 <= l <= 819: return 'entry'
    if 820 <= l <= 919: return 'norm init'
    if 621 <= l <= 639: return 'small: block tier'
    if 640 <= l <= 670: return 'small: warp tier'
    if 671 <= l <= 676: return 'small: thread tier'
    if 920 <= l <= 926: return 'small: call'
    if 927 <= l <= 940: return 'bucket apply'
    if 941 <= l <= 1041 or 700 <= l <= 725: return 'threshold scan'
    return None
