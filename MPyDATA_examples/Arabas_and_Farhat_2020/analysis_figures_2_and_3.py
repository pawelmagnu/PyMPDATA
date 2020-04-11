from MPyDATA_examples.Arabas_and_Farhat_2020.simulation import Simulation
from MPyDATA_examples.Arabas_and_Farhat_2020.setup1_european_corridor import Setup
from joblib import Parallel, delayed, parallel_backend
import numpy as np

from MPyDATA_examples.utils.error_norms import L2


def compute(log2_l2_opt: float, log2_C_opt: float):
    simulation = Simulation(Setup(l2_opt=2 ** log2_l2_opt, C_opt=2 ** log2_C_opt))
    output = []
    for n_iters in (1, 2):
        simulation.run(n_iters)
        output.append({
            "n_iters": n_iters,
            "log2_C": np.log2(simulation.C),
            "log2_C_opt": log2_C_opt,
            "log2_l2": np.log2(simulation.l2),
            "log2_l2_opt": log2_l2_opt,
            "err2": error_L2_norm(simulation.solvers, simulation.setup, simulation.S, simulation.nt, simulation.nx,
                                  n_iters)
        })
    return output


def convergence_in_space(num=8):
    with parallel_backend('threading', n_jobs=-2):
        data = Parallel(verbose=10)(
            delayed(compute)(log2_l2_opt, log2_C_opt)
            for log2_C_opt in np.linspace(-9.5, -6, num=num)
            for log2_l2_opt in range(1, 4)
        )
        result = {}
        for pair in data:
            for datum in pair:
                label = f" $\lambda^2\\approx2^{{{datum['log2_l2_opt']}}}$"
                key = ("upwind" + label, "MPDATA" + label)[datum["n_iters"]-1]
                if key not in result:
                    result[key] = ([], [])
                result[key][0].append(datum["log2_C"])
                result[key][1].append(datum["err2"])
        return result


def convergence_in_time(num=13):
    with parallel_backend('threading', n_jobs=-2):
        data = Parallel(verbose=10)(
            delayed(compute)(log2_l2_opt, log2_C_opt)
            for log2_C_opt in np.log2((.01, .005, .0025))
            for log2_l2_opt in np.linspace(1.1, 3.5, num=num)
        )
        result = {}
        for pair in data:
            for datum in pair:
                label = f" $C\\approx{2**(datum['log2_C_opt']):.4f}$"
                key = ("upwind" + label, "MPDATA" + label)[datum['n_iters']-1]
                if key not in result:
                    result[key] = ([], [])
                result[key][0].append(datum['log2_l2'])
                result[key][1].append(datum['err2'])
        return result


def error_L2_norm(solvers, setup, S, nt, nx, n_iters: int):
    numerical = solvers[n_iters].curr.get()
    analytical = setup.analytical_solution(S)
    return L2(numerical, analytical, nt)
