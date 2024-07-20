from util.run_open_loop import run_single
from simulation_parameters import SimulationParameters
import matplotlib.pyplot as plt
import os
import farms_pylog as pylog

def exercise3():

    pylog.info("Ex 3")
    log_path = './logs/exercise3/'
    os.makedirs(log_path, exist_ok=True)

    all_pars = SimulationParameters(
        n_iterations=3001,
        log_path=log_path,
        compute_metrics=3,
        return_network=True,
        headless=False
    )

    controller = run_single(
        all_pars
    )


if __name__ == '__main__':
    exercise3()

