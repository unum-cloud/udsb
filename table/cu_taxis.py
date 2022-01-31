
import cudf

from pa_taxis import PaTaxis


class CuTaxis(PaTaxis):
    """
        cuDF adaptation for on-GPU acceleration.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(backend=cudf, **kwargs)


if __name__ == '__main__':
    CuTaxis().log()
