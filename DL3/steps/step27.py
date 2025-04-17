import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from dezero.core_simple import Variable
from dezero.utils import plot_dot_graph


# 테일러 급수 기반 my_sin 함수
def my_sin(x, threshold=0.001):
    y = 0
    i = 0
    while True:
        c = (-1) ** i / np.math.factorial(2 * i + 1)
        t = c * x ** (2 * i + 1)
        y = y + t

        if np.abs(t.data) < threshold:
            break

        i += 1
    return y


# x = 3π/4
x = Variable(np.array(3 * np.pi / 4))
x.name = 'x'

# 계산
y = my_sin(x)
y.name = 'y'
y.backward()

# 계산 그래프 저장
plot_dot_graph(y, verbose=False, to_file='mysin.png')

# 값 출력
print("my_sin(x) =", y.data)
print("x.grad =", x.grad)  # 기대값: cos(3π/4) ≈ -0.707
