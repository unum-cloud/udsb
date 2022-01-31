import ray
import modin.pandas
from modin.config import Engine

from pa_taxis import PaTaxis

ray.init()
Engine.put('ray')


class MoTaxis(PaTaxis):

    def __init__(self, **kwargs) -> None:
        super().__init__(modin.pandas, **kwargs)

    def cleanup(self):
        # Modin doesn't natively support masking and defaults ot Pandas.
        self.df['passenger_count'] = self.df['passenger_count'].mask(
            self.df['passenger_count'].lt(1), 1)


if __name__ == '__main__':
    MoTaxis().log()
