import math

eps = 0.7
momentum = 0.3
pio = [0, 0]
pei = [[0,0], [0,0]]

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def MSE(ideal, result):
    return (ideal - result)**2

def proceed(enters, inside, enterinside, insideout):
    net = 0
    for a in range(len(inside)):
        inside[a] = 0
    for a in range(len(enters)):
        for j in range(len(enterinside[a])):
            inside[j] += enterinside[a][j] * enters[a]
    for a in range(len(inside)):
        inside[a] = sigmoid(inside[a])
    for a in range(len(inside)):
        net += inside[a] * insideout[a]
    net = sigmoid(net)
    return net

def backProp(res, ideal, enters, inside, enterinside, insideout):
    deltaout = (ideal - res) * (1 - res) * res
    for r in range(len(insideout)):
        deltainside = (1 - inside[r]) * inside[r] * deltaout * insideout[r]
        grad = inside[r] * deltaout
        change = eps * grad + momentum * pio[r]
        pio[r] = change
        insideout[r] += change
        for i in range(len(enterinside[r])):
            grad = enters[i] * deltainside
            change = eps * grad + momentum * pei[i][r]
            enterinside[i][r] += change

enters = [0, 0]
inside = [0, 0]
enterinside = [[0.45, 0.78], [0.12, 0.13]]
insideout = [1.5, -2.3]
learnset = [[1, 0], [0, 1], [0, 0], [1, 1]]
answers = [1, 1, 0, 0]

for k in range(1000):
    for i in range(len(learnset)):
        for j in range(len(enters)):
            enters[j] = learnset[i][j]
        res = proceed(enters, inside, enterinside, insideout)
        error = MSE(answers[i], res)
        backProp(res, answers[i], enters, inside, enterinside, insideout)

for i in range(len(learnset)):
    for j in range(len(enters)):
        enters[j] = learnset[i][j]
    res = proceed(enters, inside, enterinside, insideout)
    print (res)
    if res > 0.5:
        print(1)
    else:
        print(0)
