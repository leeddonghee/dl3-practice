import numpy as np
from dezero import Variable
import dezero

x = Variable(np.array(np.pi / 4))
y = x
for i in range(4):
    y = dezero.functions.sin(y)

y.backward(create_graph=True)

grads = [x.grad]
for i in range(3):
    gx = grads[-1]
    x.cleargrad()
    gx.backward(create_graph=True)
    grads.append(x.grad)

for i, grad in enumerate(grads):
    print(f'{i+1}차 미분: {grad.data}')
