import threading
from typing import Dict, Any
 
 #TODO double check this
class ParameterServer:

    def __init__(self):
        self._lock = threading.Lock()
        self._params: Dict[str, Any] = {}
        self._version: int = 0

    def update(self, state_dict):
        with self._lock:
            self._params = {k: v.cpu().clone() for k, v in state_dict.items()}
            self._version += 1

    def get(self):
        with self._lock:
            return self._version, {k: v.clone() for k, v in self._params.items()}
