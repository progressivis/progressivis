# Management of quality indicators in ProgressiVis
import abc


class QualityLiteral(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def quality(self, val: float) -> float:
        pass


class QualityL1(QualityLiteral):
    def __init__(self) -> None:
        self.previous: float | None = None

    def quality(self, val: float) -> float:
        try:
            ret = -abs(self.previous - val)  # type: ignore
        except TypeError:
            ret = 0
        self.previous = val
        return ret
