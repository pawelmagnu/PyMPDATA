from MPyDATA.mpdata_factory import MPDATAFactory
from MPyDATA.options import Options
import numpy as np


class TestMPDATA1D:
    def test_fct_init(self):
        # Arrange
        state = np.array([1, 2, 3])
        C = 0
        opts = Options(fct=True, n_iters=2)
        sut = MPDATAFactory.uniform_C_1d(state, C, opts)
        sut.prev.fill_halos()

        # Act
        sut.fct_init()

        # Assert
        np.testing.assert_equal(np.array([3]*5), sut.psi_max.data[1:-1])
        np.testing.assert_equal(np.array([1]*5), sut.psi_min.data[1:-1])

    def test_TODO(self):
        state = np.array([0, 1, 0])
        C = 1

        mpdata = MPDATAFactory.uniform_C_1d(state, C, Options())
        nt = 3

        conserved = np.sum(mpdata.curr.get())
        for _ in range(nt):
            mpdata.step()

        assert np.sum(mpdata.curr.get()) == conserved
