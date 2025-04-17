# step26.py
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from dezero.core_simple import Variable
from dezero.utils import plot_dot_graph

# goldstein 함수 정의
def goldstein(x, y):
    z = (1 + (x + y + 1)**2 * (19 - 14*x + 3*x**2 - 14*y + 6*x*y + 3*y**2)) * \
        (30 + (2*x - 3*y)**2 * (18 - 32*x + 12*x**2 + 48*y - 36*x*y + 27*y**2))
    return z

# 변수 생성
x = Variable(np.array(1.0))
y = Variable(np.array(1.0))

# 순전파
z = goldstein(x, y)

# 역전파
z.backward()

# 이름 붙이기
x.name = 'x'
y.name = 'y'
z.name = 'z'

# 계산 그래프 이미지 저장
plot_dot_graph(z, verbose=False, to_file='goldstein.png')
