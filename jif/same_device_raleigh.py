import socket
import threading
import os
import datasets
import fire

from .__main__ import train


def train_parallel(
    n_threads: int = 2,
    **kwargs
):
    def relative_to(i, j):
        return (i - j - 1) % n_threads
    
    n_needed_ports = n_threads * (n_threads - 1)
    free_ports = []
    for _ in range(n_needed_ports):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("0.0.0.0", 0))
        free_ports.append(s.getsockname()[1])
        s.close()
    # free_ports = [13370 + i for i in range(n_threads * (n_threads - 1))]
    
    datasets.disable_progress_bars()
    os.environ["TQDM_DISABLE"] = "1"
    def train_thread(thread_id):
        return train(**(kwargs | {
            "seed": thread_id, "params_seed": 0, "quiet": thread_id != 0,
            "raleigh_ports": [free_ports[(n_threads - 1) * thread_id + i] for i in range(n_threads - 1)],
            "raleigh_friends": [("127.0.0.1", free_ports[(n_threads - 1) * thread_id + relative_to(thread_id, i)]) for i in range(n_threads) if i != thread_id]
        }))

    threads = [threading.Thread(target=train_thread, args=(i,)) for i in range(n_threads)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()


if __name__ == "__main__":
    fire.Fire(train_parallel)
