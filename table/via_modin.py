import ray
import modin.pandas
from modin.config import Engine

from via_pandas import ViaPandas

ray.init(include_dashboard=False)
Engine.put('ray')


class ViaModin(ViaPandas):

    def __init__(self, **kwargs) -> None:
        super().__init__(modin.pandas, **kwargs)

    def cleanup(self):
        # Modin doesn't natively support masking and defaults ot Pandas.
        self.df['passenger_count'] = self.df['passenger_count'].mask(
            self.df['passenger_count'].lt(1), 1)


if __name__ == '__main__':
    ViaModin().log()
