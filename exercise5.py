

from simulation_parameters import SimulationParameters
from util.run_closed_loop import run_multiple
from util.run_closed_loop import run_single
from plot_results import plot_multiple_ex5, plot_MC, plot_CPG,plot_mean_trajectory
from plotting_common import plot_trajectory
import matplotlib.pyplot as plt

import numpy as np
import farms_pylog as pylog
import os


def exercise5():

    pylog.info("Ex 5")
    pylog.info("Implement exercise 5")
    log_path = './logs/exercise5/'
    os.makedirs(log_path, exist_ok=True)

    '''
    #SINGLE SIM
    all_pars = SimulationParameters(
        n_iterations=7001,
        log_path=log_path,
        compute_metrics=3,
        return_network=True,
        headless=True,
        Idiff = 2 # To add turning change Idiff from 0 to 4
    )

    controller = run_single(
        all_pars
    )
    plot_MC(controller)
    plot_CPG(controller)

    plt.figure("trajectory_single")
    plot_trajectory(controller)
    plot_mean_trajectory(controller)
    
    '''
    # MULTIPLE SIM
    nsim = 8

    pars_list = [
        SimulationParameters(
            simulation_i=i,#*nsim,
            n_iterations=3001,
            log_path=log_path,
            video_record=False,
            compute_metrics=2,
            Idiff=Idiff,
            headless=True,
            print_metrics=False
        )
        for i, Idiff in enumerate(np.linspace(0, 4, nsim))
    ]

    controller = run_multiple(pars_list, num_process=6)
    plot_multiple_ex5(nsim,log_path)
    
    


if __name__ == '__main__':
    exercise5()
    plt.show()

