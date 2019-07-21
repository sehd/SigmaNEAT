# from population import Population
# from config import Config


def minimumBribes(q):
    delta = 0
    anticipating = 1
    for i in range(1, len(q)+1):
        if(q[i-1] == anticipating):
            anticipating += 1
            continue
        d = q[i-1]-i
        if(d > 2):
            print("Too chaotic")
            return
        if(d != 0):
            delta += abs(d)
    print(delta)


#                    1  2  3  4  5  6  7  8
#                    1  2  3  3  4  4  4  4
print(minimumBribes([1, 2, 5, 3, 7, 8, 6, 4]))


# pop = Population()
# threadsperblock = 32
# blockspergrid = (Config.params["maxGenerationCount"] +
#                  (threadsperblock - 1)) // threadsperblock

# if(Config.system["useGpu"]):
#     pop.Run[blockspergrid, threadsperblock]()
# else:
#     pop.Run()
