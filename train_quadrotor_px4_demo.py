import lyapunov
import feedback_system
import train_lyapunov_barrier
import utils as utils
import quadrotor
import r_options
import pandas as pd

import torch
import numpy as np
import argparse
import os
import sys
import gurobipy

# roll pitch yaw u of equilibrium


def train_forward_model(forward_model, rpyu_equilibrium, model_dataset,
                        num_epochs=100, batch_size=20, lr=0.005):
    # The forward model maps (roll[n], pitch[n], yaw[n],
    # roll_sp[n], pitch_sp[n], yaw_sp[n], thrust_sp[n]) to
    # (dx[n+1] - dx[n], dy[n+1] - dy[n], dz[n+1] - dz[n], roll[n+1] - roll[n],
    # pitch[n+1] - pitch[n], yaw[n+1] - yaw[n])

    # The forward model maps (rpy[n], angular_vel[n], u[n]) to
    # (rpy[n+1], posdot[n+1] - posdot[n], angular_vel[n+1])
    network_input_data, network_output_data = model_dataset[:]
    v_dataset = torch.utils.data.TensorDataset(network_input_data,
                                               network_output_data)  # dataset for training (come visto nel quadrator2d)

    def compute_next_v(model, rpyu):
        # Questa funzione calcola la differenza tra l'output del modello quando si passa rpyu e l'output del modello quando si passa rpyu_equilibrium. In sostanza, sta calcolando la differenza tra l'uscita del modello in due stati diversi.
        return model(rpyu) - model(rpyu_equilibrium)

    utils.train_approximator(v_dataset,
                             forward_model,
                             compute_next_v,
                             batch_size=batch_size,
                             num_epochs=num_epochs,
                             lr=lr)  # vera e propria funzione di training


def train_lqr_value_approximator(state_value_dataset, lyapunov_relu, V_lambda, R, x_equilibrium, num_epochs=100, batch_size=20, lr=0.001):
    """
    We train both lyapunov_relu and R such that ϕ(x) − ϕ(x*) + λ|R(x−x*)|₁
    approximates the lqr cost-to-go.
    """

    R.requires_grad_(True)

    def compute_v(model, x):
        return model(x) - model(x_equilibrium) + V_lambda * torch.norm(
            R @ (x - x_equilibrium).T, p=1, dim=0).reshape((-1, 1))

    utils.train_approximator(state_value_dataset,
                             lyapunov_relu,
                             compute_v,
                             batch_size=batch_size,
                             num_epochs=num_epochs,
                             lr=lr,
                             additional_variable=[R])
    R.requires_grad_(False)


def train_controller_approximator(control_dataset, controller_relu, state_eq, control_equilibrium, lr, num_epochs=100, batch_size=20):

    def compute_control(model, dataset):
        return model(dataset) - model(state_eq) + control_equilibrium

    utils.train_approximator(control_dataset,
                             controller_relu,
                             compute_control,
                             batch_size=batch_size,
                             num_epochs=num_epochs,
                             lr=lr)


def load_data(path, n_in):
    Dati = pd.read_csv(path, sep=',', header=None)
    Dati_val = np.array(Dati.values)
    dati_input = Dati_val[:, :n_in]
    dati_output = Dati_val[:, n_in:]

    return dati_input, dati_output


if __name__ == "__main__":

    data_in, data_out = load_data('Dati_sistema_errore.csv', 7)
    controller_in, controller_out = load_data(
        'Dati_sistema_controllore_fixed.csv', 9)
    cost_in, cost_out = load_data('Dati_sistema_lyapunov.csv', 9)

    model_dataset = torch.utils.data.TensorDataset(
        torch.tensor(data_in), torch.tensor(data_out))
    controller_dataset = torch.utils.data.TensorDataset(
        torch.tensor(controller_in), torch.tensor(controller_out))
    cost_dataset = torch.utils.data.TensorDataset(
        torch.tensor(cost_in), torch.tensor(cost_out))

    load_models = False
    train_on_samples = True

    dtype = torch.float64
    dt = 0.01
    hover_thrust = -9.81
    rpyu_equilibrium = torch.tensor([0., 0., 0., 0., 0., 0., hover_thrust],
                                    dtype=dtype)
    V_lambda = 0.5
    x_lo = torch.tensor(
        [-.1, -.1, -1.2, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1],
        dtype=dtype)  # limiti inf

    x_up = torch.tensor(
        [.1, .1, -0.8, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], dtype=dtype)

    u_lo = torch.tensor([-2., -2., -2., 2.], dtype=dtype)
    u_up = torch.tensor([2., 2., 2., 18.], dtype=dtype)

    x_eq = torch.tensor(
        [0., 0., -1., 0., 0., 0., 0., 0., 0.],
        dtype=dtype)

    u_eq = torch.tensor([0, 0, 0, hover_thrust], dtype=dtype)

    forward_model = utils.setup_relu((7, 14, 14, 6),
                                     params=None,
                                     bias=True,
                                     negative_slope=0.01,
                                     dtype=dtype)

    R = torch.cat((.1 * torch.eye(9, dtype=dtype), .123 * torch.ones(
        (6, 9), dtype=dtype)),
        dim=0)

    lyapunov_relu = utils.setup_relu((9, 12, 6, 1),  # 1 output (cost funct)
                                     params=None,
                                     negative_slope=0.1,
                                     bias=True,
                                     dtype=dtype)
    controller_relu = utils.setup_relu((9, 7, 4),  # 4 output (thrust)
                                       params=None,
                                       negative_slope=0.01,
                                       bias=True,
                                       dtype=dtype)

    if load_models:

        R = torch.load("R_26_12_2023.pt")
        forward_model = torch.load("px4_forwardModel_trained.pt")
        lyapunov_relu = torch.load("px4_lyapunov_func_trained.pt")
        controller_relu = torch.load("px4_controller_trained.pt")
    else:
        train_forward_model(forward_model,
                            rpyu_equilibrium,
                            model_dataset,
                            num_epochs=100)
        torch.save(forward_model, 'px4_forwardModel_trained.pt')

        train_controller_approximator(
            controller_dataset, controller_relu, x_eq, u_eq, lr=0.001)
        torch.save(controller_relu, 'px4_controller_trained.pt')

        train_lqr_value_approximator(
            cost_dataset, lyapunov_relu, V_lambda, R, x_eq)
        torch.save(lyapunov_relu, 'px4_lyapunov_func_trained.pt')
        torch.save(R, "R_26_12_2023.pt")

        sys.exit(0)

    forward_system = quadrotor.QuadrotorWithPixhawkReLUSystem(
        dtype, x_lo, x_up, u_lo, u_up, forward_model, hover_thrust, dt)
    forward_system.x_equilibrium = x_eq

    closed_loop_system = feedback_system.FeedbackSystem(
        forward_system, controller_relu, forward_system.x_equilibrium,
        forward_system.u_equilibrium,
        u_lo.detach().numpy(),
        u_up.detach().numpy())

    lyap = lyapunov.LyapunovDiscreteTimeHybridSystem(closed_loop_system,
                                                     lyapunov_relu)

    R_options = r_options.FixedROptions(R)
    dut = train_lyapunov_barrier.Trainer()

    dut.add_lyapunov(
        lyap, V_lambda, closed_loop_system.x_equilibrium, R_options)
    dut.max_iterations = 1000

    # dut.add_derivative_adversarial_state = True
    dut.lyapunov_positivity_mip_term_threshold = None
    dut.lyapunov_derivative_mip_term_threshold = None
    dut.lyapunov_derivative_mip_params = {
        # gurobipy.GRB.Attr.MIPGap: 1.,
        gurobipy.GRB.Param.OutputFlag: False,
        # gurobipy.GRB.Param.TimeLimit: 100,
        # gurobipy.GRB.Param.MIPFocus: 1
    }
    # dut.lyapunov_positivity_mip_warmstart = True
    # dut.lyapunov_derivative_mip_warmstart = True
    dut.lyapunov_positivity_mip_pool_solutions = 1
    dut.lyapunov_derivative_mip_pool_solutions = 1
    dut.lyapunov_derivative_convergence_tol = 1E-3
    dut.lyapunov_positivity_convergence_tol = 1E-3
    dut.lyapunov_positivity_epsilon = 0.1
    dut.lyapunov_derivative_epsilon = 0.001
    dut.lyapunov_derivative_eps_type = lyapunov.ConvergenceEps.ExpLower
    # dut.save_network_path = 'models'
    state_samples_all = utils.uniform_sample_in_box(x_lo, x_up, 200000)
    dut.output_flag = True
    if train_on_samples:
        print("Training on samples start!!")
        dut.train_lyapunov_on_samples(state_samples_all,
                                      num_epochs=30,
                                      batch_size=200)
    dut.enable_wandb = False
    print("Training on samples done!!")
    dut.train(torch.empty((0, 9), dtype=dtype))
    pass
