import pytest
import warnings
warnings.filterwarnings("ignore")

from data.fish_database import FishDatabase

@pytest.mark.run(order=1)
def test_construct_fish_database():
    fd = FishDatabase()
    assert fd is not None
    assert len(fd) == len(fd.stores)