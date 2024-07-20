from util.run_closed_loop import run_multiple
from simulation_parameters import SimulationParameters
import os
import numpy as np
import farms_pylog as pylog
import matplotlib.pyplot as plt

def exercise8():

    pylog.info("Ex 8")
    pylog.info("Implement exercise 8")
    log_path = './logs/exercise8/'
    os.makedirs(log_path, exist_ok=True)

    nsim = 4

    pylog.info(
        "Running multiple simulations in parallel from a list of SimulationParameters")
    
    # Corrected list comprehension to include noise_sigma
    pars_list = [
        SimulationParameters(
            method="noise",
            simulation_i=i*nsim+j,
            n_iterations=3001,
            log_path=log_path,
            video_record=False,
            compute_metrics=2,
            noise_sigma=noise_sigma,
            w_stretch=w_stretch,
            headless=True,
            print_metrics=True
        )
        for i, noise_sigma in enumerate(np.linspace(0, 30, nsim))
        for j, w_stretch in enumerate(np.linspace(0, 10, nsim))
    ]

    controller=run_multiple(pars_list, num_process=16)

if __name__ == '__main__':
    exercise8()
