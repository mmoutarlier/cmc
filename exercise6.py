

from simulation_parameters import SimulationParameters
from util.run_closed_loop import run_multiple
from util.run_closed_loop import run_single
import numpy as np
import farms_pylog as pylog
import os

from plot_results import plot_CPG,plot_MC,plot_SS,plot_multiple_ex6
from plotting_common import plot_time_histories,plot_time_histories_multiple_windows
import matplotlib.pyplot as plt


def exercise6():
    
    pylog.info("Ex 6")
    pylog.info("Implement exercise 6")
    log_path = './logs/exercise6/'
    os.makedirs(log_path, exist_ok=True)

    '''
    ## SINGLE SIM ##
    all_pars = SimulationParameters(
        n_iterations=3001,
        log_path=log_path,
        compute_metrics=3,
        return_network=True,
        headless=False,
        w_stretch = 10 # To introduce feedback change w_stretch from 0 to 15 (w_stretch==gss)
    )

    controller = run_single(
        all_pars
    )
    plot_MC(controller)
    plot_CPG(controller)
    plot_SS(controller)
    labels = [str(i) for i in range(5,14)]
    plot_time_histories(controller.times,controller.joints_positions[:,5:14],offset = 0.1,title='joints_pos_vs_time',ylabel='angular position',labels = labels)
    '''
    
    ## MULTIPLE SIM ##
    nsim = 8#2,4,15,30

    pars_list = [
        SimulationParameters(
            simulation_i=i,
            n_iterations=10001,
            log_path=log_path,
            video_record=False,
            compute_metrics=3,
            headless=True,
            print_metrics=True,
            w_stretch = W_stretch
        )
        for i, W_stretch in enumerate(np.linspace(0, 15, nsim))
    ]

    controller = run_multiple(pars_list, num_process=6)
    plot_multiple_ex6(nsim,log_path)
    
    
if __name__ == '__main__':
    exercise6()
    plt.show()
