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
import gurobipy
import wandb
import sys
import csv
import hTorch.htorch.quaternion as quaternion
import hTorch.htorch.layers as layers

numQOut = 1
gain = 1
qReal = 0.05*gain
controllerLayers = (3, 2, numQOut)
name = f"Weights/Quaternion/Out{numQOut}/"
for i in controllerLayers:
    name = f"{name}{i}"
name = f"{name}/Gain{gain}/QReal{qReal}"
print(name)


dataN = {
    "weights_paths": f"{name}",
    "num_layers": len(controllerLayers) - 1,
    "slope": 0.01,
    "isQuaternion": True,
    "realPart": qReal,
    "num_output_neurons": controllerLayers[-1],
    "gain": gain
}


def train_forward_model(forward_model, rpyu_equilibrium, model_dataset,
                        num_epochs=100, batch_size=20, lr=0.005, wandb_dict=None):
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
        return model(rpyu) - model(rpyu_equilibrium.clone())

    utils.train_approximator(v_dataset,
                             forward_model,
                             compute_next_v,
                             batch_size=batch_size,
                             num_epochs=num_epochs,
                             lr=lr,
                             wandb_dict=wandb_dict)  # vera e propria funzione di training


def train_lqr_value_approximator(state_value_dataset, lyapunov_relu, V_lambda, R, x_equilibrium, num_epochs=100, batch_size=20, lr=0.001, wandb_dict=None):
    """
    We train both lyapunov_relu and R such that ϕ(x) − ϕ(x*) + λ|R(x−x*)|₁
    approximates the lqr cost-to-go.
    """

    R.requires_grad_(True)
    R = R.detach()

    def compute_v(model, x):
        return model(x) - model(x_equilibrium) + (V_lambda * torch.norm(
            R @ (x - x_equilibrium).T, p=1, dim=0).reshape((-1, 1)))

    utils.train_approximator(state_value_dataset,
                             lyapunov_relu,
                             compute_v,
                             batch_size=batch_size,
                             num_epochs=num_epochs,
                             lr=lr,
                             additional_variable=[R],
                             wandb_dict=wandb_dict)
    R.requires_grad_(False)


def train_controller_approximator(control_dataset, controller_relu, state_eq, control_equilibrium, lr, num_epochs=100, batch_size=20, wandb_dict=None, modelSave=None):

    def compute_control(model, dataset):
        return model(dataset) - model(state_eq) + control_equilibrium
    
    utils.train_approximator(control_dataset,
                             controller_relu,
                             compute_control,
                             batch_size=batch_size,
                             num_epochs=num_epochs,
                             lr=lr,
                             wandb_dict=wandb_dict, save_csv=True, modelSave=modelSave)


def load_data(path, n_in):
    Dati = pd.read_csv(path, sep=',', header=None)
    Dati_val = np.array(Dati.values)
    dati_input = gain * Dati_val[:, :n_in]
    dati_output = gain * Dati_val[:, n_in:]

    return dati_input, dati_output


def loadDataAsTensors(path, n_in):
    data_in, data_out = load_data(path, n_in)
    return torch.utils.data.TensorDataset(
        torch.tensor(data_in), torch.tensor(data_out))


def loadDataForward(path):
    Dati = pd.read_csv(path, sep=',', header=None)
    Dati_val = np.array(Dati.values)
    input = torch.tensor(Dati_val)
    output = torch.tensor(Dati_val)

    input_quaternion = []
    for i in range(input.shape[0]):
        q2 = quaternion.QuaternionTensor(
            torch.cat((torch.tensor([q]), input[i, 3:6]), dim=0))
        qa = quaternion.QuaternionTensor(
            torch.cat((torch.tensor([q]), output[i]), dim=0))
        input_quaternion.append([q2, qa])

    output_quaternion = []
    for i in range(output.shape[0]):
        q2 = quaternion.QuaternionTensor(
            torch.cat((torch.tensor([q]), input[i, 3:6]), dim=0))
        q3 = quaternion.QuaternionTensor(
            torch.cat((torch.tensor([q]), input[i, 6:9]), dim=0))
        output_quaternion.append([q3, q2])

    return torch.utils.data.TensorDataset(
        input_quaternion, output_quaternion)


def loadDatacontroller(path):
    Dati = pd.read_csv(path, sep=',', header=None)
    Dati_val = np.array(Dati.values)
    input = Dati_val[:, :9]
    output = Dati_val[:, 9:]

    input_quaternion = []
    for i in range(input.shape[0]):
        q = stateToQuaternion(input[i])
        input_quaternion.append(q)

    output_quaternion = []
    for i in range(output.shape[0]):
        q = actionToQuaternion(output[i])
        output_quaternion.append(q)
    return torch.utils.data.TensorDataset(
        torch.tensor(np.array(input_quaternion)), torch.tensor(np.array(output_quaternion)))


def loadDataLyapunov(path):
    Dati = pd.read_csv(path, sep=',', header=None)
    Dati_val = np.array(Dati.values)
    input = torch.tensor(Dati_val)
    output = torch.tensor(Dati_val)

    input_quaternion = []
    for i in range(input.shape[0]):
        q1 = quaternion.QuaternionTensor(
            torch.cat((torch.tensor([q]), input[i, 0:3]), dim=0))
        q2 = quaternion.QuaternionTensor(
            torch.cat((torch.tensor([q]), input[i, 3:6]), dim=0))
        q3 = quaternion.QuaternionTensor(
            torch.cat((torch.tensor([q]), input[i, 6:9]), dim=0))
        input_quaternion.append([q1, q2, q3])

    output_quaternion = []
    for i in range(output.shape[0]):
        output_quaternion.append(-1)

    return torch.utils.data.TensorDataset(
        input_quaternion, output_quaternion)

# State is a np.array of 9 elements


def stateToQuaternion(state):
    q = np.concatenate((np.array([qReal]), state[0:3], np.array(
        [qReal]), state[3:6], np.array([qReal]), state[6:9]))
    return q


# Action is a np.array of 4 elements
def actionToQuaternion(action):
    if numQOut == 2:
        q = np.concatenate((np.array([qReal]), action, np.array(
            [0]), np.array([0]), np.array([0])))
    else:
        q = action
    return q


if __name__ == "__main__":
    # Import datasets
    # model_dataset = loadDataForward("Dataset/Data.csv")
    # cost_dataset = loadDataLyapunov("Dataset/Data.csv")
    controller_dataset = loadDatacontroller("Dataset/Data.csv")

    train_on_samples = True
    dtype = torch.float64
    dt = 0.01
    V_lambda = 0.5

    ########################################
    # Control Parameters
    hover_thrust = 0.605

    x_lo = torch.tensor(
        [-.1, -.1, -1.2, -0.1, -0.1, -2, -0.1, -0.1, -0.1],
        dtype=dtype)
    x_up = torch.tensor(
        [.1, .1, -0.8, 0.1, 0.1, 2, 0.1, 0.1, 0.1], dtype=dtype)
    u_lo = torch.tensor([-2., -2., -2., 2.], dtype=dtype)
    u_up = torch.tensor([2., 2., 2., 18.], dtype=dtype)

    x_eq = torch.tensor(
        [0., 0., -1., 0., 0., 1.5708, 0., 0., 0.],
        dtype=dtype)
    u_eq = torch.tensor([0, 0, 0, hover_thrust], dtype=dtype)
    rpyu_equilibrium = torch.tensor([x_eq[3], x_eq[4], x_eq[5], u_eq[0], u_eq[1], u_eq[2], u_eq[3]],
                                    dtype=dtype)

    # q2 = torch.cat((q, rpyu_equilibrium[0:3]), dim=0)
    # qa = rpyu_equilibrium[3:]
    # rpyu_equilibrium = torch.cat((q2, qa), dim=0)

    x_lo = torch.tensor(stateToQuaternion(np.array(x_lo)))
    x_up = torch.tensor(stateToQuaternion(np.array(x_up)))
    x_eq = torch.tensor(stateToQuaternion(np.array(x_eq)))

    u_lo = torch.tensor(actionToQuaternion(np.array(u_lo)))
    u_up = torch.tensor(actionToQuaternion(np.array(u_up)))
    u_eq = torch.tensor(actionToQuaternion(np.array(u_eq)))
    #######################################

    # Define the models
    controller_relu = torch.nn.Sequential(
        layers.QLinear(3, 2, bias=True).type(torch.float64),
        torch.nn.LeakyReLU(0.01),
        layers.QLinear(2, 1, bias=True).type(torch.float64)
    )
    # controller_relu = utils.setup_relu_quaternion(controllerLayers,
    #                                               params=None,
    #                                               negative_slope=0.01,
    #                                               bias=True,
    #                                               dtype=dtype)

    forward_model = utils.setup_relu_quaternion((2, 4, 4, 2),
                                                params=None,
                                                bias=True,
                                                negative_slope=0.01,
                                                dtype=dtype)

    R = torch.cat((.1 * torch.eye(9, dtype=dtype), .123 * torch.ones(
        (6, 9), dtype=dtype)),
        dim=0)

    lyapunov_relu = utils.setup_relu_quaternion((3, 3, 2, 1),
                                                params=None,
                                                negative_slope=0.1,
                                                bias=True,
                                                dtype=dtype)

    # Wandb Initialization
    # wandb.login(key="e3f943e00cb1fa8a14fd0ea76ed9ee6d50f86f5b")
    wandb_dict = {"entity": "emacannizzo",
                  "project": "Quadrotor Test",
                  "run_name": "Training"}
    wandb_dictPre = {"entity": "emacannizzo",
                     "project": "Quadrotor Test",
                     "run_name": "Training"}

    load_models = False
    load_pretrained = False
    last_epoch_saved = -1

    if load_models and wandb_dict is not None:
        wandb_dictPre["run_name"] = "PreTrainingForward"
        utils.wandbDownload(wandb_dictPre, "model.pt")
        forward_model = torch.load(utils.wandbGetLocalPath(
            wandb_dictPre) + "/model.pt")

        if load_pretrained:
            wandb_dictPre["run_name"] = "PreTrainingController"
            utils.wandbDownload(wandb_dictPre, "model.pt")
            controller_relu = torch.load(utils.wandbGetLocalPath(
                wandb_dictPre) + "/model.pt")

            wandb_dictPre["run_name"] = "PreTrainingLyapunov"
            utils.wandbDownload(wandb_dictPre, "model.pt")
            lyapunov_relu = torch.load(utils.wandbGetLocalPath(
                wandb_dictPre) + "/model.pt")
            utils.wandbDownload(wandb_dictPre, "R.pt")
            R = torch.load(utils.wandbGetLocalPath(
                wandb_dictPre) + "/R.pt")
            R = R[0]
        else:
            utils.wandbDownload(
                wandb_dict, f"{last_epoch_saved}/controller.pt")
            controller_relu = torch.load(utils.wandbGetLocalPath(
                wandb_dict) + f"/{last_epoch_saved}/controller.pt")

            utils.wandbDownload(wandb_dict, f"{last_epoch_saved}/lyapunov.pt")
            lyapunov_relu = torch.load(utils.wandbGetLocalPath(
                wandb_dict) + f"/{last_epoch_saved}/lyapunov.pt")

            utils.wandbDownload(wandb_dict, f"{last_epoch_saved}/R.pt")
            R = torch.load(utils.wandbGetLocalPath(wandb_dict) +
                           f"/{last_epoch_saved}/R.pt")

    else:
        # wandb_dictPre["run_name"] = "PreTrainingForward"
        # train_forward_model(forward_model,
        #                     rpyu_equilibrium,
        #                     model_dataset,
        #                     num_epochs=20, batch_size=200, wandb_dict=None)

        # wandb_dictPre["run_name"] = "PreTrainingLyapunov"
        # train_lqr_value_approximator(
        #     cost_dataset, lyapunov_relu, V_lambda, R, x_eq, num_epochs=20, batch_size=200, wandb_dict=None)

        wandb_dictPre["run_name"] = "PreTrainingController"
        train_controller_approximator(
            controller_dataset, controller_relu, x_eq, u_eq, lr=0.001, num_epochs=1200, batch_size=500, wandb_dict=None, modelSave=dataN)

    sys.exit()
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
    dut.output_flag = True

    # utils.wandbDownload(wandb_dict, "state_samples_all.csv")
    # state_samples_all = utils.csvToTensor(utils.wandbGetLocalPath(wandb_dict) + "/state_samples_all.csv", dtype=dtype)
    state_samples_all = utils.uniform_sample_in_box(x_lo, x_up, 50)
    dut.train_lyapunov_on_samples(state_samples_all,
                                  num_epochs=1000,
                                  batch_size=200, wandb_dict=wandb_dict,
                                  last_epoch_saved=last_epoch_saved)

    # if train_on_samples:
    #     print("Training on samples start!!")
    #     dut.train_lyapunov_on_samples(state_samples_all,
    #                                   num_epochs=30,
    #                                   batch_size=200)
    # dut.enable_wandb = False
    # print("Training on samples done!!")
    # dut.train(torch.empty((0, 9), dtype=dtype))

    pass
