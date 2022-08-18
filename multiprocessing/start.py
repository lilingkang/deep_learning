import multiprocessing as mp


def task(a, b):
    print("Hello, multiprocessing!")


if __name__ ==  '__main__':
    p1 = mp.Process(target=task, args=(1,2))
    p1.start()
