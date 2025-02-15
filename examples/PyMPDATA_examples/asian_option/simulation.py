import numba
import numpy as np
from PyMPDATA_examples.asian_option.options import OPTIONS

from PyMPDATA import Options, ScalarField, Solver, Stepper, VectorField
from PyMPDATA.boundary_conditions import Constant, Extrapolated, Periodic


class Simulation:
    @staticmethod
    def _factory(
        *,
        options: Options,
        advectee: np.ndarray,
        advector_s: float,
        advector_a: np.ndarray,
        boundary_conditions,
    ):
        stepper = Stepper(
            options=options, n_dims=len(advectee.shape), non_unit_g_factor=False
        )
        # a_dim_advector = np.multiply.outer(
        #     np.ones(advectee.shape[0] + 1, dtype=options.dtype), advector_a
        # )
        x_dim_advector = np.full(
            (advectee.shape[0], advectee.shape[1] + 1), advector_s, dtype=options.dtype
        )
        print(f"{x_dim_advector.shape=}", f"{advector_a.shape=}")
        advector_values = (advector_a, x_dim_advector)
        # print(advector_values)
        return Solver(
            stepper=stepper,
            advectee=ScalarField(
                advectee.astype(dtype=options.dtype),
                halo=options.n_halo,
                boundary_conditions=boundary_conditions,
            ),
            advector=VectorField(
                advector_values,
                halo=options.n_halo,
                boundary_conditions=boundary_conditions,
            ),
        )

    def __init__(self, settings):
        self.settings = settings

        sigma2 = pow(settings.sigma, 2)
        dx_opt = abs(
            settings.C_opt / (0.5 * sigma2 - settings.r) * settings.l2_opt * sigma2
        )
        dt_opt = pow(dx_opt, 2) / sigma2 / settings.l2_opt

        # adjusting dt so that nt is integer
        self.dt = settings.T
        self.nt = 0
        while self.dt > dt_opt:
            self.nt += 1
            self.dt = settings.T / self.nt

        # adjusting dx to match requested l^2
        dx = np.sqrt(settings.l2_opt * self.dt) * settings.sigma

        self.dt = self.dt / 10
        self.nt = self.nt * 10
        print(f"{self.nt=}, {self.dt=}")

        # calculating actual u number and lambda
        self.C = -(0.5 * sigma2 - settings.r) * (-self.dt) / dx

        self.l2 = dx * dx / sigma2 / self.dt

        # adjusting nx and setting S_beg, S_end
        S_beg = settings.S_match
        self.nx = 1
        while S_beg > settings.S_min:
            self.nx += 1
            S_beg = np.exp(np.log(settings.S_match) - self.nx * dx)

        # self.na = 15
        # self.na = self.nx # TODO: why?
        # self.na = 42
        # self.da = dx

        self.ix_match = self.nx

        S_end = settings.S_match
        while S_end < settings.S_max:
            self.nx += 1
            S_end = np.exp(np.log(S_beg) + (self.nx - 1) * dx)

        # asset price
        self.S = np.exp(np.log(S_beg) + np.arange(self.nx) * dx)

        self.A = self.S
        self.dy = dx
        self.ny = self.nx
        # self.A, self.da = np.linspace(S_beg, S_end, self.na, retstep=True)
        print(f"{self.S.shape=}, {self.A.shape=}")

        # a advector
        # geometric = True
        # arithmetic = False
        # assert geometric ^ arithmetic
        # if geometric:
        #     # meshgrid for the advector, a
        #     c_a_dummy = np.zeros((self.na, self.nx))
        #     for i in range(self.na):
        #         c_a_dummy[i, :] = np.log(self.S / self.A[i])
        #     self.C_a = c_a_dummy
        # # if arithmetic: # TODO
        # #     self.C_a = self.S - self.A
        # self.C_a *= (-self.dt) / self.da / settings.T
        # # self.A_courant_multiplier = (-self.dt) / self.da / settings.T
        # # self.C_a = self.C_a.T
        # try:
        #     assert np.max(np.abs(self.C_a)) < 1
        # except AssertionError:
        #     print(f"{np.max(np.abs(self.C_a))=}")
        #     raise
        # meshgrid
        self.S_mesh, self.A_mesh = np.meshgrid(self.S, self.A)
        print(f"{self.S_mesh.shape=}, {self.A_mesh.shape=}")

        self.mu_coeff = (0.5 / self.l2, 0)
        # self.mu_coeff = (0, 0)
        # self.solver = {}
        # self.solvers[1] = self._factory(
        #     advectee=settings.payoff(self.A_mesh),
        #     advector=self.C,
        #     options=Options(n_iters=1, non_zero_mu_coeff=True),
        #     boundary_conditions=(Extrapolated(), Extrapolated()),
        #     time_to_maturity=settings.T,
        #     advectee_x_values=self.S,
        # )

        self.x = np.log(S_beg) + np.arange(self.nx) * dx
        self.y = np.log(S_beg) + np.arange(self.ny + 1) * dx
        self.A_values = np.exp(np.log(S_beg) + np.arange(self.ny) * dx)
        self.S_values = np.exp(np.log(S_beg) + np.arange(self.nx) * dx)
        a_dim_advector = np.zeros((self.nx, self.ny + 1))
        for i in range(self.ny + 1):
            a_dim_advector[:, i] = (
                ((self.x - (self.y[i] - dx / 2)) / settings.T) * -(self.dt) / self.dy
            )

        advectee = settings.terminal_value(
            np.exp(np.log(S_beg) + np.arange(self.ny) * dx)
        )
        print(f"{advectee.shape=}, {np.max(advectee)=}, {np.min(advectee)=}")
        advector_value_s = self.C
        # advector_a = self.C_a
        options = Options(**OPTIONS)
        # boundary_conditions = (Periodic(), Constant(2000))
        # boundary_conditions = (Extrapolated(), Extrapolated())
        # boundary_conditions = (Periodic(), Periodic())
        boundary_conditions = (Extrapolated(dim=0), Extrapolated(dim=-1))

        stepper = Stepper(
            options=options, n_dims=len(advectee.shape), non_unit_g_factor=False
        )

        # a_dim_advector_linspace = np.linspace(S_beg, S_end + self.dy, self.ny + 1)

        # for i in range(self.ny + 1):
        #     a_dim_advector[i, :] = (
        #         np.log(self.S / a_dim_advector_linspace[i])
        #         * (-self.dt)
        #         / self.dy
        #         / settings.T
        #     )
        # a_dim_advector[i, :] = (self.S / a_dim_advector_linspace[i]) * (-self.dt) / self.da / settings.T

        # try:
        #     assert np.max(np.abs(a_dim_advector)) < 1
        #     print(f"{np.max(np.abs(a_dim_advector))=}")
        # except AssertionError:
        #     print(f"{a_dim_advector.shape=}")
        #     print(f"{np.max(np.abs(a_dim_advector))=}")
        x_dim_advector = np.full(
            (advectee.shape[0] + 1, advectee.shape[1]),
            advector_value_s,
            # 0,
            dtype=options.dtype,
        )

        # x dim advector should have values in the x direction
        # a dim advector should have values in the a direction
        # x_dim_advector = np.zeros((self.nx + 1, self.ny))
        # for i in range(self.nx + 1):
        #     x_dim_advector[i, :] = advector_value_s
        try:
            assert np.max(np.abs(x_dim_advector)) < 1
            print(f"{np.max(np.abs(a_dim_advector))=}, {a_dim_advector.shape}")
            print(f"{np.max(np.abs(x_dim_advector))=}, {x_dim_advector.shape}")
        except AssertionError:
            print(f"{np.max(np.abs(x_dim_advector))=}")

        print(f"{x_dim_advector.shape=}", f"{a_dim_advector.shape=}")
        # advector_values = (a_dim_advector, x_dim_advector)
        advector_values = (x_dim_advector, a_dim_advector)

        self.solver = Solver(
            stepper=stepper,
            advectee=ScalarField(
                advectee.astype(dtype=options.dtype),
                halo=options.n_halo,
                boundary_conditions=boundary_conditions,
            ),
            advector=VectorField(
                advector_values,
                halo=options.n_halo,
                boundary_conditions=boundary_conditions,
            ),
        )

        # self.solver = self._factory(
        #     advectee=settings.terminal_value(self.A_mesh),
        #     advector_s=self.C,
        #     advector_a=self.C_a,
        #     options=Options(**OPTIONS),
        #     boundary_conditions=(Periodic(), Periodic()),
        # )

    def run(self, n_iters: int):
        # for i in range(self.nt):
        #     print(f"{i=}")
        #     self.solver.advance(1, self.mu_coeff)
        # # self.solver.advance(self.nt, self.mu_coeff)
        self.solver.advance(1, self.mu_coeff)
        return self.solver.advectee.get().copy()

    # def terminal_value(self):
    #     return self.solvers[1].advectee.get()
