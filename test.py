import math

list = [-15.9714, 39.7226, -10.2047, 8.3652, -34.2642, -18.0605, -19.1563, 34.7959, 22.5679, -27.9340]
sum_of_exps = sum([math.exp(x) for x in list])
print(sum_of_exps)

for i in range(len(list)):
    elem0 = math.log(math.exp(list[i])/sum_of_exps)
    print(elem0)

def allinone(lista):
    resultado = [math.log(math.exp(x)/sum([math.exp(x) for x in lista])) for x in lista]
    return resultado

print(allinone(list))