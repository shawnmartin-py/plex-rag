from enum import Enum

from app.repositories.json import JsonMediaItems
from app.repositories.sql import SqlMediaItems


class RepoType(Enum):
    SQL = SqlMediaItems
    JSON = JsonMediaItems

    def __call__(self):
        return self.value()
