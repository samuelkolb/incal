import errno
import multiprocessing
import os
import signal
import subprocess
from multiprocessing import Queue, Manager, Process, cpu_count
from multiprocessing.pool import Pool
from subprocess import TimeoutExpired
from traceback import print_exc
from typing import Optional, Union, Any

from pebble import ProcessPool

from .observe import Observer, dispatch
from concurrent.futures import TimeoutError as OtherTimeoutError


class Update:
    SENTINEL = "sentinel"
    STARTED = "started"
    DONE = "done"
    TIMEOUT = "timeout"
    FAILED = "failed"

    def __init__(self, status, index, command, meta):
        self.status = status
        self.index = index
        self.command = command
        self.meta = meta


class ParallelObserver(Observer):
    @dispatch
    def observe(self, update):
        # type: (Update) -> None
        raise NotImplementedError()


def observe(observer, queue, count=None):
    # type: (ParallelObserver, Queue, Optional[int]) -> None
    to_see = None if count is None else set(range(count))
    while to_see is None or len(to_see) > 0:
        update = queue.get()  # type: Union[Update, str]
        if update == Update.SENTINEL:
            return
        elif isinstance(update, Update):
            if update.status == Update.DONE or update.status == Update.TIMEOUT or update.status == Update.FAILED:
                if to_see:
                    to_see.remove(update.index)
            try:
                observer.observe(update)
            except Exception:
                print_exc()
        else:
            raise ValueError("Invalid update {}".format(update))


def run_command(command, timeout=None):
    return worker((-1, None, command, timeout, None))


def run_function(f, *args, timeout=None, **kwargs):
    worker((-1, None, (f, args, kwargs), timeout, None))


def worker(args):
    i, meta, command, timeout, queue = args  # type: (int, Any, Any, int, Queue)
    # TODO Capture output?

    if isinstance(command, str):
        is_string = True
    else:
        try:
            is_string = len(command) > 0 and not callable(command[0])
        except TypeError:
            raise ValueError("Command must be either string, args or function-args pair")

    if is_string:
        with subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              start_new_session=True) as process:
            try:
                if queue:
                    queue.put(Update(Update.STARTED, i, command, meta))
                out, err = process.communicate(timeout=timeout)
                if queue:
                    queue.put(Update(Update.DONE, i, command, meta))
                return out.decode(), err.decode()
            except TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGINT)  # send signal to the process group
                except OSError as e:
                    if e.errno != errno.ESRCH:
                        if e.errno == errno.EPERM:
                            os.waitpid(-process.pid, 0)
                    else:
                        raise e
                finally:
                    if queue:
                        queue.put(Update(Update.TIMEOUT, i, command, meta))

                process.communicate()
    else:
        assert isinstance(command, (tuple, list))
        if len(command) < 2:
            command = command + ([],)
        if len(command) < 3:
            command = command + (dict(),)

        f, args, kwargs = command

        p = multiprocessing.Process(target=f, args=args, kwargs=kwargs)
        if queue:
            queue.put(Update(Update.STARTED, i, command, meta))

        p.start()
        p.join(timeout)

        if p.is_alive():
            p.terminate()
            p.join()
            if queue:
                queue.put(Update(Update.TIMEOUT, i, command, meta))

        else:
            if queue:
                queue.put(Update(Update.DONE, i, command, meta))


def run_commands(commands, processes=None, timeout=None, meta=None, observer=None):
    pool = Pool(processes=processes)
    manager, queue = None, None
    if observer:
        manager = Manager()
        queue = manager.Queue()

    if meta:
        commands = [(i, meta, command, timeout, queue) for i, (command, meta) in enumerate(zip(commands, meta))]
    else:
        commands = [(i, meta, command, timeout, queue) for i, command in enumerate(commands)]

    r = pool.map_async(worker, commands)

    if observer:
        observe(observer, queue, len(commands))

    r.wait()
