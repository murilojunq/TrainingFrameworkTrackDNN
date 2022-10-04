import time

class Timer:
    """
    Simple timer to measure how long block of code takes

    :param
        block_name : printout name for the codeblock being timed
    """

    def __init__(self, block_name):
        self._code_block_name = block_name
        self._start_time = None

    def __enter__(self ):
        self.start()
        return self

    def __exit__(self, *exc_info):
        self.stop()

    def start(self):
        self._start_time = time.perf_counter()

    def stop(self):
        runtime = round(time.perf_counter() - self._start_time, 4)
        print(f"{self._code_block_name}: Time elapsed: {runtime} s")
        self._start_time = None