
import cudf

from via_pandas import ViaPandas


class ViaCuDF(ViaPandas):
    """
        CuDF adaptation for on-GPU acceleration.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(backend=cudf, **kwargs)


if __name__ == '__main__':
    ViaCuDF().log()
