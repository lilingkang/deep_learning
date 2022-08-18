import multiprocessing as mp


def job(x):
    return x*x


def multicore():
    pool = mp.Pool(processes=3)  # 使用3个核
    res = pool.map(job, range(10))
    print(res)


if __name__ == "__main__":
    multicore()