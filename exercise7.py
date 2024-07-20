

from simulation_parameters import SimulationParameters
from util.run_closed_loop import run_multiple, run_single
import numpy as np
import farms_pylog as pylog
from plot_results import plot_CPG,plot_MC,plot_SS, plot_trajectory
import matplotlib.pyplot as plt
import os


def exercise7(ind=0, w_stretch=5): #replace the index to store for different values of w_stretch

    pylog.info("Ex 7")
    pylog.info("Implement exercise 7")
    os.makedirs('./logs/exercise7', exist_ok=True)
    log_path = './logs/exercise7/w_stretch'+str(ind)+'/'
    os.makedirs(log_path, exist_ok=True)



    nsim = 4

    pylog.info(
        "Running multiple simulations in parallel from a list of SimulationParameters")
    pars_list = [
        SimulationParameters(
        simulation_i=i,
        n_iterations=3001,
        log_path=log_path,
        compute_metrics=3,
        w_stretch=w_stretch,
        Idiff=Idiff,
        return_network=True,
        headless=True
        )
        for i, Idiff in enumerate(np.linspace(0, 4, nsim))
        #for i, I in enumerate(np.linspace(0, 30, nsim)) #---to get the plots with varying I
    

    ]
    controller= run_multiple(pars_list, num_process=8) 


if __name__ == '__main__':
    exercise7()
    
 


