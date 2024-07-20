"""Network controller"""

import numpy as np
from scipy.interpolate import CubicSpline
import scipy.stats as ss
import farms_pylog as pylog


class FiringRateController_close_loop:
    """zebrafish controller"""

    def __init__(
            self,
            pars
    ):
        super().__init__()

        self.n_iterations = pars.n_iterations
        self.n_neurons = pars.n_neurons
        self.n_muscle_cells = pars.n_muscle_cells
        self.timestep = pars.timestep
        self.times = np.linspace(
            0,
            self.n_iterations *
            self.timestep,
            self.n_iterations)
        self.pars = pars

        self.n_eq = self.n_neurons*4 + self.n_muscle_cells*2 + self.n_neurons * \
            2  # number of equations: number of CPG eq+muscle cells eq+sensors eq
        
        # Muscles activations indexes
        self.muscle_l = 4*self.n_neurons + 2 * \
            np.arange(0, self.n_muscle_cells)  # muscle cells left indexes
        self.muscle_r = self.muscle_l+1  # muscle cells right indexes
        self.all_muscles = 4*self.n_neurons + \
            np.arange(0, 2*self.n_muscle_cells)  # all muscle cells indexes
        
        
        # vector of indexes for the CPG activity variables - modify this
        # according to your implementation

        # Firing rate indexes
        self.neuron_l =  4 * np.arange(0, self.n_neurons)
        self.neuron_r = self.neuron_l + 2
        self.all_v = np.concatenate([self.neuron_l,self.neuron_r])
        #self.all_v = range(self.n_neurons*2)
        
        # Firing rate adaptation indexes
        self.left_a = self.neuron_l + 1
        self.right_a = self.neuron_l + 3
        self.all_a = np.concatenate([self.left_a,self.right_a])

        # Stretch sensory population indexes
        self.sensor_l = 4*self.n_neurons + 2*self.n_muscle_cells + 2*np.arange(0,self.n_neurons)
        self.sensor_r = self.sensor_l + 1
        self.all_s = 4*self.n_neurons + 2*self.n_muscle_cells + np.arange(0,2*self.n_neurons)

        self.state = np.zeros([self.n_iterations, self.n_eq])  # equation state
        self.dstate = np.zeros([self.n_eq])  # derivative state
        self.state[0] = np.random.rand(self.n_eq)  # set random initial state

        self.poses = np.array([
            0.007000000216066837,
            0.00800000037997961,
            0.008999999612569809,
            0.009999999776482582,
            0.010999999940395355,
            0.012000000104308128,
            0.013000000268220901,
            0.014000000432133675,
            0.014999999664723873,
            0.01600000075995922,
        ])  # active joint distances along the body (pos=0 is the tip of the head)
        self.poses_ext = np.linspace(
            self.poses[0], self.poses[-1], self.n_neurons)  # position of the sensors

        # initialize ode solver
        self.f = self.ode_rhs

        # stepper function selection
        if self.pars.method == "euler":
            self.step = self.step_euler
        elif self.pars.method == "noise":
            self.sigma = self.pars.noise_sigma
            self.step = self.step_euler_maruyama
            # vector of noise for the CPG voltage equations (2*n_neurons)
            self.noise_vec = np.zeros(self.n_neurons*2)

        # zero vector activations to make first and last joints passive
        # pre-computed zero activity for the first 4 joints
        self.zeros8 = np.zeros(8)
        # pre-computed zero activity for the tail joint
        self.zeros2 = np.zeros(2)


    def get_ou_noise_process_dw(self,timestep, x_prev, sigma):
        """
        Implement the integration of the Ornstein-Uhlenbeck processes using the Euler-Maruyama method.
        dx_t = -0.5*x_t*dt + sigma*dW_t 
        Parameters
        ----------
        timestep: float
        Timestep size.
        x_prev: np.array
            Previous state of the OU process.
        sigma: float
            Noise level, standard deviation of the Wiener process increment.
        Returns
        -------
        np.array
            Next state of the OU process.
        """
       # Generate Wiener increments
        dW = np.random.normal(loc=0.0, scale=np.sqrt(timestep), size=x_prev.shape)

        # Calculate the increment in the Ornstein-Uhlenbeck process
        dx_process = -0.1 * x_prev * timestep + sigma * np.sqrt(timestep) * dW

        # Update the process value
        x_new = dx_process

        return x_new
    

    def step_euler(self, iteration, time, timestep, pos=None):
        """Euler step"""
        self.state[iteration+1, :] = self.state[iteration, :] + \
            timestep*self.f(time, self.state[iteration], pos=pos)
        return np.concatenate([
            self.zeros8,  # the first 4 passive joints
            self.motor_output(iteration),  # the active joints
            self.zeros2  # the last (tail) passive joint
        ])

    def step_euler_maruyama(self, iteration, time, timestep, pos=None):
        """Euler Maruyama step"""
        self.state[iteration+1, :] = self.state[iteration, :] + \
            timestep*self.f(time, self.state[iteration], pos=pos)
        self.noise_vec = self.get_ou_noise_process_dw(
            timestep, self.noise_vec, self.pars.noise_sigma)
        self.state[iteration+1, self.all_v] += self.noise_vec
        self.state[iteration+1,
                   self.all_muscles] = np.maximum(self.state[iteration+1,
                                                             self.all_muscles],
                                                  0)  # prevent from negative muscle activations
        return np.concatenate([
            self.zeros8,  # the first 4 passive joints
            self.motor_output(iteration),  # the active joints
            self.zeros2  # the last (tail) passive joint
        ])

    def motor_output(self, iteration):
        """
        Here you have to final muscle activations for the 10 active joints.
        It should return an array of 2*n_muscle_cells=20 elements,
        even indexes (0,2,4,...) = left muscle activations
        odd indexes (1,3,5,...) = right muscle activations
        """

        return self.pars.act_strength*self.state[iteration, self.all_muscles]
    
        '''return np.zeros(
            2 *
            self.n_muscle_cells)  # here you have to final active muscle equations for the 10 joints'''

    def ode_rhs(self,  _time, state, pos=None):
        """Network_ODE
        You should implement here the right hand side of the system of equations
        Parameters
        ----------
        _time: <float>
            Time
        state: <np.array>
            ODE states at time _time
        Returns
        -------
        dstate: <np.array>
            Returns derivative of state
        """
        #compute W_in matrix:
        W_in = np.zeros((self.n_neurons, self.n_neurons))
        for i in range(self.n_neurons):
            for j in range(self.n_neurons):
                if (i <= j) and ((j-i) <= self.pars.n_asc):
                    W_in[i][j] = 1/(j-i+1)
                elif (i > j) and ((i-j) <= self.pars.n_desc):
                    W_in[i][j] = 1/(i-j+1)
                else:
                    W_in[i][j] = 0

        #compute W_ss matrix:
        W_ss = np.zeros((self.n_neurons, self.n_neurons))
        for i in range(self.n_neurons):
            for j in range(self.n_neurons):
                if (i <= j) and ((j-i) <= self.pars.n_asc_str):
                    W_ss[i][j] = 1/(j-i+1)
                elif (i > j) and ((i-j) <= self.pars.n_desc_str):
                    W_ss[i][j] = 1/(i-j+1)
                else:
                    W_ss[i][j] = 0

        #compute W_mc matrix:
        W_mc = np.zeros((self.n_muscle_cells, self.n_neurons))
        n_mc = self.pars.n_neurons/self.pars.n_muscle_cells
        for i in range(self.n_muscle_cells):
            for j in range(self.n_neurons):
                if (n_mc*i <= j) and (j <= n_mc*(i+1)-1):
                    W_mc[i][j] = 1
                else:
                    W_mc[i][j] = 0
        

       # Compute the angle positions at 50 joint locations
        cs = CubicSpline(self.poses, pos)
        theta = cs(self.poses_ext)
        
        self.dstate[self.neuron_l] = (-state[self.neuron_l] + np.sqrt(np.maximum(self.pars.I + self.pars.Idiff - self.pars.b*state[self.left_a] - np.dot(self.pars.w_inh*W_in, state[self.neuron_r]) - np.dot(self.pars.w_stretch*W_ss, state[self.sensor_r]), 0))) * (1/self.pars.tau)
        self.dstate[self.left_a] = (-state[self.left_a] + self.pars.gamma * state[self.neuron_l])*(1/self.pars.taua)
        self.dstate[self.neuron_r] = (-state[self.neuron_r] + np.sqrt(np.maximum(self.pars.I - self.pars.Idiff - self.pars.b*state[self.right_a] - np.dot(self.pars.w_inh*W_in,state[self.neuron_l]) - np.dot(self.pars.w_stretch*W_ss, state[self.sensor_l]), 0))) * (1/self.pars.tau)
        self.dstate[self.right_a] = (-state[self.right_a] + self.pars.gamma * state[self.neuron_r])*(1/self.pars.taua)
        self.dstate[self.muscle_l] = (np.dot(self.pars.w_V2a2muscle*W_mc, state[self.neuron_l])*(1-state[self.muscle_l])/self.pars.taum_a) - state[self.muscle_l]/self.pars.taum_d
        self.dstate[self.muscle_r] = (np.dot(self.pars.w_V2a2muscle*W_mc, state[self.neuron_r])*(1-state[self.muscle_r])/self.pars.taum_a) - state[self.muscle_r]/self.pars.taum_d
        self.dstate[self.sensor_l] = (np.sqrt(np.maximum(theta,0))*(1-state[self.sensor_l]) - state[self.sensor_l])*(1/self.pars.tau_str)
        self.dstate[self.sensor_r] = (np.sqrt(np.maximum(-theta,0))*(1-state[self.sensor_r]) - state[self.sensor_r])*(1/self.pars.tau_str)
        return self.dstate

