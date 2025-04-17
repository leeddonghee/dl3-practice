from dezero import Variable

x = Variable((2.0))
y = x * x
y.backward()

print(y)        
print(x.grad)   
