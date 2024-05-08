from __future__ import annotations

from collections import defaultdict
import csv
from optparse import Option


class DataLogger:
    def __init__(self, file_path=None):
        self._save_file = file_path
        self._log_data: defaultdict(list) = defaultdict(list)

    def reset(self):
        self._log_data: defaultdict(list) = defaultdict(list)

    def log(self, data: defaultdict):
        for key, value in data.items():
            self._log_data[key].append(value)

    def print(self):
        for key, value in self._log_data.items():
            print(f"{key} size: {len(value)}")

    def save(self, path=None):
        if self._save_file is None and path is None:
            raise Exception("File path is not set.")

        if path is not None:
            self._save_file = path

        with open(self._save_file, "w") as f:
            writer = csv.writer(f)
            writer.writerow(self._log_data.keys())
            writer.writerows(zip(*self._log_data.values()))

    def mean(self, key: str) -> Option[float]:
        if self._log_data[key] is None:
            return None

        if len(self._log_data[key]) == 0:
            return 0.0
        else:
            ave = sum(self._log_data[key]) / len(self._log_data[key])
            return ave

    def sum(self, key: str) -> Option[float]:
        if self._log_data[key] is None:
            return None

        if len(self._log_data[key]) == 0:
            return 0.0
        else:
            ave = sum(self._log_data[key])
            return ave

    def max(self, key: str) -> Option[float]:
        if self._log_data[key] is None:
            return None

        if len(self._log_data[key]) == 0:
            return 0.0
        else:
            max_val = max(self._log_data[key])
            return max_val

    def min(self, key: str) -> Option[float]:
        if self._log_data[key] is None:
            return None

        if len(self._log_data[key]) == 0:
            return 0.0
        else:
            min_val = min(self._log_data[key])
            return min_val
