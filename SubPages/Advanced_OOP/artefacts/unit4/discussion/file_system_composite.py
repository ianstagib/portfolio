"""Task 3: the Composite pattern for files and folders.

``FileSystemComponent`` is the component interface. ``File`` is a leaf and
``Folder`` is a composite that holds any components, including other folders.
Client code calls ``size()`` or ``display()`` on either without checking
which it has, and a folder answers by asking its children.

>>> root = Folder("project")
>>> docs = root.add(Folder("docs"))
>>> _ = docs.add(File("report.docx", 120))
>>> _ = root.add(File("main.py", 4))
>>> root.size()
124
>>> print(root.display())
project/ (124 KB)
  docs/ (120 KB)
    report.docx (120 KB)
  main.py (4 KB)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator

__all__ = ["FileSystemComponent", "File", "Folder"]


class FileSystemComponent(ABC):
    """What files and folders have in common."""

    def __init__(self, name: str) -> None:
        if not name or "/" in name or name in {".", ".."}:
            raise ValueError(f"invalid name {name!r}")
        self.name = name
        self.parent: Folder | None = None

    @abstractmethod
    def size(self) -> int:
        """Size in KB."""

    @abstractmethod
    def display(self, indent: int = 0) -> str: ...

    def path(self) -> str:
        return self.name if self.parent is None else f"{self.parent.path()}/{self.name}"


class File(FileSystemComponent):
    """A leaf: it has no children."""

    def __init__(self, name: str, size_kb: int) -> None:
        super().__init__(name)
        if size_kb < 0:
            raise ValueError("size cannot be negative")
        self._size = size_kb

    def size(self) -> int:
        return self._size

    def display(self, indent: int = 0) -> str:
        return f"{'  ' * indent}{self.name} ({self._size} KB)"


class Folder(FileSystemComponent):
    """A composite: it holds files and folders and delegates to them."""

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._children: dict[str, FileSystemComponent] = {}

    def add(self, item: FileSystemComponent) -> FileSystemComponent:
        if item.parent is not None:
            raise ValueError(f"{item.name!r} is already in {item.parent.path()!r}")
        if item.name in self._children:
            raise ValueError(f"{self.path()!r} already contains {item.name!r}")
        ancestor: Folder | None = self
        while ancestor is not None:  # refuse cycles, which would never terminate
            if ancestor is item:
                raise ValueError("a folder cannot be placed inside itself")
            ancestor = ancestor.parent
        self._children[item.name] = item
        item.parent = self
        return item

    def remove(self, name: str) -> FileSystemComponent:
        item = self._children.pop(name)
        item.parent = None
        return item

    def __iter__(self) -> Iterator[FileSystemComponent]:
        return iter(self._children.values())

    def size(self) -> int:
        return sum(child.size() for child in self)

    def display(self, indent: int = 0) -> str:
        lines = [f"{'  ' * indent}{self.name}/ ({self.size()} KB)"]
        lines += [child.display(indent + 1) for child in self]
        return "\n".join(lines)
