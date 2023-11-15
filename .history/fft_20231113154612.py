import math

def fft(data, nn, isign):
    n = nn << 1
    j = 0
    for i in range(0, n, 2):
        if j > i:
            data[j], data[i] = data[i], data[j]  # SWAP
            data[j+1], data[i+1] = data[i+1], data[j+1]  # SWAP
        m = n >> 1
        while m >= 2 and j >= m:
            j -= m
            m >>= 1
        j += m

    mmax = 2
    while n > mmax:
        istep = mmax << 1
        theta = isign * (2 * math.pi / mmax)
        wtemp = math.sin(0.5 * theta)
        wpr = -2.0 * wtemp * wtemp
        wpi = math.sin(theta)
        wr = 1.0
        wi = 0.0
        for m in range(1, mmax, 2):
            for i in range(m, n, istep):
                j = i + mmax
                tempr = wr * data[j] - wi * data[j+1]
                tempi = wr * data[j+1] + wi * data[j]
                data[j] = data[i] - tempr
                data[j+1] = data[i+1] - tempi
                data[i] += tempr
                data[i+1] += tempi
            wr, wi = wr * wpr - wi * wpi + wr, wi * wpr + wtemp * wpi + wi
        mmax = istep

    # Apply normalization if it's a forward FFT (isign == 1)
    if isign == 1:
        factor = 1.0 / nn
        for i in range(n):
            data[i] *= factor