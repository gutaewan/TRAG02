# trag/workers/base_worker.py
from __future__ import annotations

import threading
import time
from typing import Optional, Callable

class BaseWorker(threading.Thread):
    def __init__(
        self,
        name: str,
        interval_seconds: int,
        run_once_fn: Callable[[], None],
        daemon: bool = True,
    ):
        super().__init__(name=name, daemon=daemon)
        self.interval_seconds = interval_seconds
        self.run_once_fn = run_once_fn
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def stopped(self) -> bool:
        return self._stop_event.is_set()

    def run(self):
        while not self.stopped():
            try:
                self.run_once_fn()
            except Exception as e:
                # worker 내부 예외는 상위에서 로깅하도록 run_once_fn에서 처리 권장
                pass
            # interval sleep (중간 stop 반영)
            for _ in range(int(self.interval_seconds)):
                if self.stopped():
                    break
                time.sleep(1)