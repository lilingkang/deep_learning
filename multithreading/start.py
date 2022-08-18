import threading as td


def thread_job():
    print("Hello, multithreading! %s" % td.current_thread())


def main():
    t1 = td.Thread(target=thread_job)
    t1.start()
    print(td.active_count())
    print(td.enumerate())
    print(td.current_thread())


if __name__ == "__main__":
    main()
