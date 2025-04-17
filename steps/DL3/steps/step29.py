import numpy as np
from dezero.core_simple import Variable

# 초기값
x = Variable(np.array(2.0))
iters = 0

while True:
    iters += 1
    y = (x - 1) ** 2  # f(x)

    x.cleargrad()
    y.backward()  # f'(x)

    grad = x.grad
    hess = 2.0  # f''(x) = 2

    x.data -= grad / hess  # 뉴튼 방법

    print(f"Iter {iters}: x = {x.data:.10f}, grad = {grad:.10f}")

    if np.abs(grad) < 1e-6:  # 수렴 조건
        break

print(f"\n최솟값에 수렴한 x = {x.data}")
print(f"수렴까지 걸린 반복 횟수: {iters}")
