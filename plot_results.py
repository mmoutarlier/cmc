"""Plot results"""

import farms_pylog as pylog
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from util.rw import load_object
from plotting_common import plot_2d, save_figures, plot_left_right, plot_trajectory, plot_time_histories, plot_time_histories_multiple_windows
import numpy as np
from metrics import compute_speed_PCA
import matplotlib
import os
import argparse
from matplotlib.patches import Circle
from scipy.optimize import curve_fit
matplotlib.rc('font', **{"size": 15})


def plot_exercise_multiple(n_simulations, logdir):
    """
    Example showing how to load a simulation file and use the plot2d function
    """
    pylog.info(
        "Example showing how to load the simulation file and use the plot2d function")
    fspeeds = np.zeros([n_simulations, 3])
    for i in range(n_simulations):
        # load controller
        controller = load_object(logdir+"controller"+str(i))
        fspeeds[i] = [
            controller.pars.amp,
            controller.pars.wavefrequency,
            np.mean(controller.metrics["fspeed_cycle"])
        ]

    plt.figure('exercise_multiple', figsize=[10, 10])
    plot_2d(
        fspeeds,
        ['amp', 'wavefrequency', 'Forward Speed [m/s]'],
        cmap='nipy_spectral'
    )

def plot_MC(controller):
    #plot muscle activities
    plt.figure('muscle_activities_single')
    plot_left_right(
        controller.times,
        controller.state,
        controller.muscle_l,
        controller.muscle_r,
        cm="green",
        offset=0.1)

def plot_CPG(controller):
    #plot CPG activities
    plt.figure('neuron_CPG_activities')
    plot_left_right(
        controller.times,
        controller.state,
        controller.neuron_l,
        controller.neuron_r,
        cm="green",
        offset=0.1)

def plot_SS(controller):
    #plot SS activities
    plt.figure('stretch_sensitive_neuron_activities')
    plot_left_right(
        controller.times,
        controller.state,
        controller.sensor_l,
        controller.sensor_r,
        cm="green",
        offset=0.1)
    



def plot_multiple_ex5(n_simulation, logdir):

    I_diff = np.zeros(n_simulation)
    curvature = np.zeros(n_simulation)
    lspeed = np.zeros(n_simulation)
    for i in range(n_simulation):
        controller = load_object(logdir+"controller"+str(i))
        I_diff[i] = controller.pars.Idiff
        curvature[i] = controller.metrics["curvature"]
        lspeed[i] = controller.metrics["lspeed_cycle"]
    
    plt.figure('curvature vs I_diff')
    plt.plot(I_diff,curvature)
    plt.xlabel('I_diff')
    plt.ylabel('curvature')

    plt.figure('lateral speed vs I_diff')
    plt.plot(I_diff,lspeed)
    plt.xlabel('I_diff')
    plt.ylabel('lspeed')

def plot_multiple_ex6(n_simulation, logdir):

    W_stretch = np.zeros(n_simulation)
    frequency = np.zeros(n_simulation)
    wavefrequency = np.zeros(n_simulation)
    fspeed = np.zeros(n_simulation)
    fspeed_pca = np.zeros(n_simulation)
    for i in range(n_simulation):
        controller = load_object(logdir+"controller"+str(i))
        W_stretch[i] = controller.pars.w_stretch
        frequency[i] = controller.metrics["frequency"]
        wavefrequency[i] = controller.metrics["wavefrequency"]
        fspeed[i] =  controller.metrics["fspeed_cycle"] #controller.metrics["fspeed_cycle"]
        fspeed_pca[i] = controller.metrics["fspeed_PCA"]
    
    plt.figure('freq_gss')
    plt.plot(W_stretch,frequency)
    plt.xlabel('W_stretch')
    plt.ylabel('frequency')

    plt.figure('wavefreq_gss')
    plt.plot(W_stretch,wavefrequency)
    plt.xlabel('W_stretch')
    plt.ylabel('wavefrequency')

    plt.figure('fspeed_gss')
    plt.plot(W_stretch,fspeed)
    plt.xlabel('W_stretch')
    plt.ylabel('fspeed_cycle')

    plt.figure('fspeed_pca_gss')
    plt.plot(W_stretch,fspeed_pca)
    plt.xlabel('W_stretch')
    plt.ylabel('fspeed_PCA')

def circle_func(x, xc, yc, r):
    """Circle function"""
    return np.sqrt((x[:, 0] - xc)**2 + (x[:, 1] - yc)**2) - r

def plot_mean_trajectory(controller, color=None, sim_fraction=1):

    mean_positions = np.mean(np.array(controller.links_positions),axis=1)
    n_steps = mean_positions.shape[0]
    n_steps_considered = round(n_steps * sim_fraction)

    mean_positions = mean_positions[-n_steps_considered:, :2]

    """Plot positions"""
    plt.figure("trajectory")
    plt.plot(mean_positions[:-1, 0],mean_positions[:-1, 1], label='Center of mass trajectory', color=color)
    
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.axis('equal')
    plt.grid(True)

    
    # Fit a circle to the trajectory
    popt, _ = curve_fit(circle_func, mean_positions, np.zeros(mean_positions.shape[0]))
    # Plot fitted circle
    xc, yc, r = popt
    theta = np.linspace(0, 2 * np.pi, 100)
    plt.plot(xc + r * np.cos(theta), yc + r * np.sin(theta), color='red', linestyle='--', label='Fitted circle')
    print('Radius of fitted circle',r)
    print("1/curvature=",1/controller.metrics["curvature"])
    plt.legend()  # Add the legend to the plot
    plt.show()

def plot_exercise_7_1(n_simulations, logdir):
    """
    Example showing how to load a simulation file and use the plot2d function
    """
    pylog.info(
        "Example showing how to load the simulation file and use the plot2d function")
    freq = np.zeros([n_simulations, 3])
    for i in range(n_simulations):
        # load controller
        controller = load_object(logdir+"controller"+str(i))
        freq[i] = [
            controller.pars.I,
            controller.pars.b,
            np.mean(controller.metrics["frequency"])
        ]

    plt.figure('exercise_7_1', figsize=[10, 10])
    plot_2d(
        freq,
        ['I', 'b', 'frequency'],
        cmap='nipy_spectral'
    )

def plot_exercise_7_2(n_simulations, logdir):
    """
    Example showing how to load a simulation file and use the plot2d function
    """
    pylog.info(
        "Example showing how to load the simulation file and use the plot2d function")
    wavefreq = np.zeros([n_simulations, 3])
    for i in range(n_simulations):
        # load controller
        controller = load_object(logdir+"controller"+str(i))
        wavefreq[i] = [
            controller.pars.I,
            controller.pars.b,
            np.mean(controller.metrics["wavefrequency"])
        ]
        

    plt.figure('exercise_7_2', figsize=[10, 10])
    plot_2d(
        wavefreq,
        ['I', 'b', 'wavefrequency'],
        cmap='nipy_spectral'
    )

def plot_exercise_7_3(n_simulations, logdir):
    """
    Example showing how to load a simulation file and use the plot2d function
    """
    pylog.info(
        "Example showing how to load the simulation file and use the plot2d function")
    lspeed = np.zeros([n_simulations, 3])
    for i in range(n_simulations):
        # load controller
        controller = load_object(logdir+"controller"+str(i))
        lspeed[i] = [
            controller.pars.I,
            controller.pars.b,
            np.mean(controller.metrics["lspeed_cycle"])
        ]
        

    plt.figure('exercise_7_3', figsize=[10, 10])
    plot_2d(
        lspeed,
        ['I', 'b', 'lspeed'],
        cmap='nipy_spectral'
    )

def plot_exercise_7_4(n_simulations,logdir):
    I_values = np.zeros(n_simulations)
    wavefreq = np.zeros(n_simulations)
    freq = np.zeros(n_simulations)
    ptcc_values = np.zeros(n_simulations)
    
    for i in range(n_simulations):
        file_path = os.path.join(logdir, "controller" + str(i))
        controller = load_object(file_path)
        I_values[i] = controller.pars.I
        wavefreq[i] = controller.metrics["wavefrequency"]
        freq[i] = controller.metrics["frequency"]
        ptcc_values[i] = controller.metrics["ptcc"]
    
    plt.figure('wavefrequency vs I')
    plt.plot(I_values, wavefreq, marker='o', linestyle='-', linewidth=2)
    plt.xlabel('I')
    plt.ylabel('wavefrequency')
    plt.title('Wavefrequency vs Id')
    plt.grid(True)
    plt.savefig(os.path.join(logdir, 'wavefrequency_vs_I.png'))
    
    plt.figure('frequency vs I')
    plt.plot(I_values, freq, marker='o', linestyle='-', linewidth=2)
    plt.xlabel('I')
    plt.ylabel('frequency')
    plt.title('Frequency vs I')
    plt.grid(True)
    plt.savefig(os.path.join(logdir, 'frequency_vs_Idiff.png'))
    
    plt.figure('ptcc values vs I')
    plt.plot(I_values, ptcc_values, marker='o', linestyle='-', linewidth=2)
    plt.axhline(y=1.5, color='r', linestyle='--', label='Threshold')
    plt.xlabel('Idiff')
    plt.ylabel('ptcc')
    plt.title('PTCC Values vs Idiff')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(logdir, 'ptcc_values_vs_Idiff.png'))
    
    # Show all plots
    plt.show()

def plot_exercise_7_5(n_simulation, logdir):
    
    ptcc = np.zeros([n_simulation, 3])
    feasible_Is = []
    for i in range(n_simulation):
                # load controller
                
        controller = load_object(logdir+"controller"+str(i))
        ptcc[i] = [
        controller.pars.Idiff,
        controller.metrics["wavefrequency"],
        np.mean(controller.metrics["ptcc"])
        ]
        if np.mean(controller.metrics["ptcc"]) > 1.5:
            feasible_Is.append(controller.pars.Idiff)


    plt.figure('ptcc with wavefrequency and Idiff', figsize=[10, 10])
    plot_2d(
    ptcc,
    ['Idiff', 'wavefrequency', 'ptcc'],
    cmap='nipy_spectral'
    )
            
    ptcc = np.zeros([n_simulation, 3])
    for i in range(n_simulation):
        # load controller
        controller = load_object(logdir+"controller"+str(i))
        ptcc[i] = [
        controller.pars.Idiff,
        controller.metrics["frequency"],
        np.mean(controller.metrics["ptcc"])
        ]

    plt.figure('ptcc with frequency and I', figsize=[10, 10])
    plot_2d(
        ptcc,
        ['Idiff', 'frequency', 'ptcc'],
        cmap='nipy_spectral'
        )



def plot_exercise_8(n_simulations, logdir):
    """
    Example showing how to load a simulation file and use the plot2d function
    """
    pylog.info(
        "Example showing how to load the simulation file and use the plot2d function")
    fspeeds = np.zeros([n_simulations, 3])
    for i in range(n_simulations):
        # load controller
        controller = load_object(logdir+"controller"+str(i))
        fspeeds[i] = [
            controller.pars.noise_sigma,
            controller.pars.w_stretch,
            np.mean(controller.metrics["fspeed_PCA"])
        ]
        print(fspeeds[i])


    plt.figure('exercice_8_2', figsize=[10, 10])
    plot_2d(
        fspeeds,
        ['noise', 'w_strech', 'Forward Speed PCA [m/s]'],
        cmap='nipy_spectral'
    )

    pylog.info(
        "Example showing how to load the simulation file and use the plot2d function")
    fspeeds = np.zeros([n_simulations, 3])
    for i in range(n_simulations):
        # load controller
        controller = load_object(logdir+"controller"+str(i))
        fspeeds[i] = [
            controller.pars.noise_sigma,
            controller.pars.w_stretch,
            np.mean(controller.metrics["lspeed_PCA"])
        ]
        print(fspeeds[i])


    plt.figure('exercice_8', figsize=[10, 10])
    plot_2d(
        fspeeds,
        ['noise', 'w_strech', 'Lateral Speed PCA [m/s]'],
        cmap='nipy_spectral'
    )

'''def main(plot=True):
    """Main"""

    # pylog.info("Here is an example to show how you can load a single simulation and which data you can load")
    # controller = load_object("logs/example_single/controller0")

    # # neural data
    # state   = controller.state
    # metrics = controller.metrics

    # # mechanical data
    # links_positions       = controller.links_positions # the link positions
    # links_velocities      = controller.links_velocities # the link velocities
    # joints_active_torques = controller.joints_active_torques # the joint active torques
    # joints_velocities     = controller.joints_velocities # the joint velocities
    # joints_positions      = controller.joints_positions # the joint positions


if __name__ == '__main__':
    main(plot=True)'''



def main(exercise_nb, plot=True):
    """Main"""
    
    controller_nb = 0
    


    if exercise_nb == 7:
        controller = load_object("logs/exercise{}/w_stretch{}/controller{}".format(exercise_nb,controller_nb, controller_nb*4)) # replace with w_stretch0 or w_stretch1 or w_stretch2 or w_stretch3 or w_stretch4
    else:
        controller = load_object("logs/exercise{}/controller{}".format(exercise_nb, controller_nb))

    # neural data
    state = controller.state
    metrics = controller.metrics

    times = controller.times
    left_idx_muscle = controller.muscle_l
    right_idx_muscle = controller.muscle_r

    left_idx_neuron = controller.neuron_l
    right_idx_neuron = controller.neuron_r


    

    if plot:

        if exercise_nb == 3:
            #plot muscle activities
            plt.figure('muscle_activities_single')
            plot_left_right(
                times,
                state,
                left_idx_muscle,
                right_idx_muscle,
                cm="green",
                offset=0.1)

            #plot CPG activities
            plt.figure('neuron_CPG_acitivities')
            plot_left_right(
                times,
                state,
                left_idx_neuron,
                right_idx_neuron,
                cm="green",
                offset=0.1)
        
        if exercise_nb == 4:
            n_simulations = 10
            I_values = np.zeros(n_simulations)
            wavefreq = np.zeros(n_simulations)
            freq = np.zeros(n_simulations)
            ptcc_values = np.zeros(n_simulations)
            for i in range(n_simulations):
                controller = load_object("logs/exercise{}/controller{}".format(exercise_nb, i*n_simulations))
                I_values[i] = controller.pars.I
                wavefreq[i] = controller.metrics["wavefrequency"]
                freq[i] = controller.metrics["frequency"]
                ptcc_values[i] = controller.metrics["ptcc"]

            
            plt.figure('wavefrequency vs I')
            plt.plot(I_values,wavefreq)
            plt.xlabel('I')
            plt.ylabel('wavefrequency')

            plt.figure('frequency vs I')
            plt.plot(I_values,freq)
            plt.xlabel('I')
            plt.ylabel('frequency')

            plt.figure('ptcc values vs I')
            plt.plot(I_values,ptcc_values)
            plt.axhline(y=1.5, color='r', linestyle='--') 
            plt.xlabel('I')
            plt.ylabel('ptcc')

            ptcc = np.zeros([n_simulations, 3])
            feasible_Is = []
            for i in range(n_simulations):
                # load controller
                controller = load_object("logs/exercise{}/controller{}".format(exercise_nb, i*n_simulations))
                ptcc[i] = [
                    controller.pars.I,
                    controller.metrics["wavefrequency"],
                    np.mean(controller.metrics["ptcc"])
                ]
                if np.mean(controller.metrics["ptcc"]) > 1.5:
                    feasible_Is.append(controller.pars.I)


            plt.figure('ptcc with wavefrequency and I', figsize=[10, 10])
            plot_2d(
                ptcc,
                ['I', 'wavefrequency', 'ptcc'],
                cmap='nipy_spectral'
            )
            
            ptcc = np.zeros([n_simulations, 3])
            for i in range(n_simulations):
                # load controller
                controller = load_object("logs/exercise{}/controller{}".format(exercise_nb, i*n_simulations))
                ptcc[i] = [
                    controller.pars.I,
                    controller.metrics["frequency"],
                    np.mean(controller.metrics["ptcc"])
                ]

            plt.figure('ptcc with frequency and I', figsize=[10, 10])
            plot_2d(
                ptcc,
                ['I', 'frequency', 'ptcc'],
                cmap='nipy_spectral'
            )


        if exercise_nb ==7:
            #plot_exercise_7_1(4, "logs/exercise7/w_stretch0/") #choose depending the plot wanted !
            #plot_exercise_7_2(4, "logs/exercise7/w_stretch0/")
            #plot_exercise_7_3(4, "logs/exercise7/w_stretch0/")
            #plot_exercise_7_4(4, "logs/exercise7/w_stretch0/")
            plot_exercise_7_5(4, "logs/exercise7/w_stretch0/")
    

        if exercise_nb == 8:
            plot_exercise_8(16, "logs/exercise8/")

        #if want to save figures:
        save_figures()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot results from controller files.')
    parser.add_argument('exercise_number', type=int, help='Exercise number')
    args = parser.parse_args()
    main(args.exercise_number, plot=True)
    plt.show()