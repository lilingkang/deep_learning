from ast import arg
import multiprocessing as mp


def task(q, num):
    res = 0
    for i in range(num):
        res += i
    q.put(res)


if __name__ == "__main__":
    q = mp.Queue()
    p1 = mp.Process(target=task, args=(q,100))
    p2 = mp.Process(target=task, args=(q,101))
    p1.start()
    p2.start()
    p1.join()
    p2.join()
    res1 = q.get()
    res2 = q.get()
    print(res1)
    print(res2)