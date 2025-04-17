import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from dezero.core_simple import Variable

# 로젠브록 함수 정의
def rosenbrock(x0, x1):
    return (1 - x0)**2 + 100 * (x1 - x0**2)**2

# 초기값 설정
x0 = Variable(np.array(0.0))
x1 = Variable(np.array(2.0))

# 하이퍼파라미터
lr = 0.05   # 학습률
iters = 1000  # 반복 횟수

for i in range(iters):
    y = rosenbrock(x0, x1)
    x0.cleargrad()
    x1.cleargrad()
    y.backward()

    # 경사하강법: 현재 값 - 학습률 * 기울기
    x0.data -= lr * x0.grad
    x1.data -= lr * x1.grad

    if i % 100 == 0 or i == iters - 1:
        print(f"Iter {i}: x0 = {x0.data:.4f}, x1 = {x1.data:.4f}, y = {y.data:.6f}")

print("\n최솟값 추정:")
print(f"x0 = {x0.data}, x1 = {x1.data}")
