
import cudf

import pa_taxis


class CuTaxis(pa_taxis.PaTaxis):

    def __init__(self) -> None:
        super().__init__(backend=cudf)


if __name__ == '__main__':
    CuTaxis().log()
