from time import sleep, process_time, perf_counter, thread_time, time
import threading

if __name__ == '__main__':
    i = 0
    while i <= 30:
        ptStart = process_time()
        pfStart = perf_counter()
        thStart = thread_time()
        tStart = time()
        sleep(1)
        print(perf_counter())
        print(f"time: {format((time() - tStart), '.6f')}")
        print(f"processtime: {format((process_time() - ptStart), '.6f')}")
        print(f"perfcounter:{format((perf_counter() - pfStart), '.6f')}")
        print("threads: ", threading.active_count())
        print(f"threadtime:{format((thread_time() - thStart), '.6f')} \n")
        i += 1
    print(f"END processtime: {process_time()}")
    print(f"END perfcounter:{perf_counter()}")