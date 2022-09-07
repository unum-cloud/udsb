import os
import warnings
os.environ['RAY_DISABLE_IMPORT_WARNING'] = '1'  # nopep8
warnings.filterwarnings('ignore')  # nopep8

import ray
import modin.pandas
from modin.config import Engine
ray.init(include_dashboard=False)  # nopep8
Engine.put('ray')  # nopep8

from via_pandas import ViaPandas
import dataset


class ViaModin(ViaPandas):

    def __init__(self, **kwargs) -> None:
        super().__init__(modin.pandas, **kwargs)

    def _cleanup(self):
        # Modin doesn't natively support masking and defaults to Pandas.
        self.df['passenger_count'] = self.df['passenger_count'].mask(
            self.df['passenger_count'].lt(1), 1)


if __name__ == '__main__':
    dataset.test_engine(ViaModin())
