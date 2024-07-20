

from simulation_parameters import SimulationParameters
from util.run_open_loop import run_multiple
import numpy as np
import farms_pylog as pylog
import os


def exercise4():

    pylog.info("Ex 4")
    pylog.info("Implement exercise 4")
    log_path = './logs/exercise4/'
    os.makedirs(log_path, exist_ok=True)

    nsim = 10 #2,4,8

    pars_list = [
        SimulationParameters(
            simulation_i=i*nsim,
            n_iterations=3001,
            log_path=log_path,
            video_record=False,
            compute_metrics=2,
            I=I,
            headless=True,
            print_metrics=False
        )
        #for i, I in enumerate(np.linspace(4, 22, nsim))  ---- To get the plots only in the stable region
        for i, I in enumerate(np.linspace(0, 30, nsim))
    ]

    controller = run_multiple(pars_list, num_process=8)


if __name__ == '__main__':
    exercise4()

