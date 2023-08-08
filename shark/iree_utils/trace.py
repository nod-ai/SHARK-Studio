# Copyright 2023 The Nod Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Tuple

import os
import threading
import time


def _enable_detail_trace() -> bool:
    return os.getenv("SHARK_DETAIL_TRACE", "0") == "1"


class DetailLogger:
    """Context manager which can accumulate detailed log messages.

    Detailed log is only emitted if the operation takes a long time
    or errors.
    """

    def __init__(self, timeout: float):
        self._timeout = timeout
        self._messages: List[Tuple[float, str]] = []
        self._start_time = time.time()
        self._active = not _enable_detail_trace()
        self._lock = threading.RLock()
        self._cond = threading.Condition(self._lock)
        self._thread = None

    def __enter__(self):
        self._thread = threading.Thread(target=self._run)
        self._thread.start()
        return self

    def __exit__(self, type, value, traceback):
        with self._lock:
            self._active = False
            self._cond.notify()
        if traceback:
            self.dump_on_error(f"exception")

    def _run(self):
        with self._lock:
            timed_out = not self._cond.wait(self._timeout)
        if timed_out:
            self.dump_on_error(f"took longer than {self._timeout}s")

    def log(self, msg):
        with self._lock:
            timestamp = time.time()
            if self._active:
                self._messages.append((timestamp, msg))
            else:
                print(f"  +{(timestamp - self._start_time) * 1000}ms: {msg}")

    def dump_on_error(self, summary: str):
        with self._lock:
            if self._active:
                print(f"::: Detailed report ({summary}):")
                for timestamp, msg in self._messages:
                    print(
                        f"  +{(timestamp - self._start_time) * 1000}ms: {msg}"
                    )
            self._active = False
