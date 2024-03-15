import math
import pandas as pd
from array import *
from particles import dict_particles
import csv
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import seaborn as sns

frame_particles = pd.DataFrame(dict_particles).T


Energy_exp = []
counts = []
angles = [17.4]
with open(
    "/Users/semenraydun/Desktop/NIRS/RUN019/RUN019_6He_bin/RUN019_dE2(6He)_17.4.txt",
    "r",
) as file:
    for line in file:
        data = line.split()
        Energy_exp.append(float(data[0]))
        counts.append(float(data[1]))

# def get_input(frame_particles: pd.DataFrame):
#     """Input from the user"""
#     A = None
#     B = None
#     C = None
#     D = None
#     E_Aparticle = None
#     angles = None

#     while A not in frame_particles.index:
#         A = input("Beam particle: ")
#         if A not in frame_particles.index:
#             print("Error: Invalid beam particle!")

#     while B not in frame_particles.index:
#         B = input("Target: ")
#         if B not in frame_particles.index:
#             print("Error: Invalid target!")

#     while C not in frame_particles.index:
#         C = input("Fragment: ")
#         if C not in frame_particles.index:
#             print("Error: Invalid beam particle!")

#     while D not in frame_particles.index:
#         D = input("Residual: ")
#         if D not in frame_particles.index:
#             print("Error: Invalid beam particle!")

#     # while E_Aparticle is None:
#     #     try:
#     #         E_Aparticle = float(input("Incident particle energy: "))
#     #     except ValueError:
#     #         print("Error: Incident particle energy must be a valid float!")

#     # while angles is None:
#     #     try:
#     #         angles = [
#     #             float(angle)
#     #             for angle in input("Enter a list of angles (space-separated): ").split()
#     #         ]
#     #     except ValueError:
#     #         print("Error: Angles must be valid floats!")

#     return A, B, C, D


def calculate_Q_react(A: str, B: str, C: str, D: str, frame_particles: pd.DataFrame):
    """Calculations of the reactions Q-value"""
    input_channel_mass = (
        (float(frame_particles.loc[A, "Z"]) + float(frame_particles.loc[B, "Z"]))
        * 938.27199838
        + (float(frame_particles.loc[A, "N"]) + float(frame_particles.loc[B, "N"]))
        * 939.56533038
        + (
            float(frame_particles.loc[A, "E_binding"])
            + float(frame_particles.loc[B, "E_binding"])
        )
    )
    output_channel_mass = (
        (float(frame_particles.loc[C, "Z"]) + float(frame_particles.loc[D, "Z"]))
        * 938.27199838
        + (float(frame_particles.loc[C, "N"]) + float(frame_particles.loc[D, "N"]))
        * 939.56533038
        + (
            float(frame_particles.loc[C, "E_binding"])
            + float(frame_particles.loc[D, "E_binding"])
        )
    )

    Q_react = float(input_channel_mass - output_channel_mass)
    return Q_react


def calculate_E_t(
    A: str,
    B: str,
    C: str,
    D: str,
    E_Aparticle: float,
    theta_lab: float,
    frame_particles: pd.DataFrame,
    Q_react,
):
    """Calculates the energy of the registered particle"""
    E_t = E_Aparticle + Q_react

    y = (
        float(frame_particles.loc[A, "Mass_amu"]) * 931.5
        + float(frame_particles.loc[B, "Mass_amu"]) * 931.5
    )

    betac = math.sqrt(
        E_Aparticle
        * (E_Aparticle + 2 * float(frame_particles.loc[A, "Mass_amu"]) * 931.5)
    ) / (y + E_Aparticle)

    ecmi = math.sqrt(
        y * y + (2 * E_Aparticle * float(frame_particles.loc[B, "Mass_amu"]) * 931.5)
    )

    ecmf = (
        ecmi
        + Q_react
        - y
        + float(frame_particles.loc[C, "Mass_amu"]) * 931.5
        + float(frame_particles.loc[D, "Mass_amu"]) * 931.5
    )

    e3cm = (
        ecmf * ecmf
        + (
            float(frame_particles.loc[C, "Mass_amu"]) * 931.5
            + float(frame_particles.loc[D, "Mass_amu"]) * 931.5
        )
        * (
            float(frame_particles.loc[C, "Mass_amu"]) * 931.5
            - float(frame_particles.loc[D, "Mass_amu"]) * 931.5
        )
    ) / (2 * ecmf)

    y_new = ((e3cm / (float(frame_particles.loc[C, "Mass_amu"]) * 931.5)) ** 2) * (
        1 - betac * betac
    )

    cosagl = math.cos(theta_lab * math.pi / 180)

    BB = (0 - betac) * cosagl

    AA = y_new + (BB * BB)

    CC = 1 - y_new

    D2 = BB * BB - AA * CC

    if D2 >= 0:
        b3L1 = ((0 - BB) + math.sqrt(D2)) / AA
    else:
        b3L1 = -100

    if b3L1 > 0.0000001:
        E_light_particle_out = (float(frame_particles.loc[C, "Mass_amu"]) * 931.5) * (
            (1 / math.sqrt(1 - b3L1 * b3L1)) - 1
        )
        E_light_particle_out_u = E_light_particle_out / float(
            frame_particles.loc[C, "Mass_amu"]
        )
    else:
        E_light_particle_out = None
        E_light_particle_out_u = None

    return E_light_particle_out


def approximate(E_calib, E_11C):
    X_fact = np.array(E_calib)
    Y_fact = np.array(E_11C)
    linear_func = lambda x, b0, b1: b0 + b1 * x
    quadratic_func = lambda x, b0, b1, b2: b0 + b1 * x + b2 * x**2
    qubic_func = lambda x, b0, b1, b2, b3: b0 + b1 * x + b2 * x**2 + b3 * x**3

    models_dict = {
        "linear": linear_func,
        "quadratic": quadratic_func,
        "qubic": qubic_func,
    }

    model_list = ["linear", "quadratic", "qubic"]
    methods_dict = {"linear": "lm", "quadratic": "lm", "qubic": "lm"}
    p0_dict = {"linear": [0, 0], "quadratic": [0, 0, 0], "qubic": [0, 0, 0, 0]}
    formulas_dict = {
        "linear": "y = b0 + b1*x",
        "quadratic": "y = b0 + b1*x + b2*x^2",
        "qubic": "y = b0 + b1*x + b2*x^2 + b3*x^3",
    }

    # return all optional outputs
    full_output = False

    # variables to save the calculation results
    calculation_results_df = pd.DataFrame(
        columns=[
            "func",
            "p0",
            "popt",
            "cov_x",
            "SSE",
            "MSE",
            "pcov",
            "perr",
            "Y_calc",
            "error_metrics",
        ]
    )
    error_metrics_results_df = pd.DataFrame(
        columns=["MSE", "RMSE", "MAE", "MSPE", "MAPE", "RMSLE", "R2"]
    )

    # calculations
    for func_name in model_list:
        print(f"{func_name.upper()} MODEL: {formulas_dict[func_name]}")
        func = models_dict[func_name]
        calculation_results_df.loc[func_name, "func"] = func
        p0 = p0_dict[func_name]
        print(f"p0 = {p0}")
        calculation_results_df.loc[func_name, "p0"] = p0
        if full_output:
            (popt, pcov, infodict, mesg, ier) = curve_fit(
                func,
                X_fact,
                Y_fact,
                p0=p0,
                method=methods_dict[func_name],
                full_output=full_output,
            )
            integer_flag = (
                f"ier = {ier}, the solution was found"
                if ier <= 4
                else f"ier = {ier}, the solution was not found"
            )
            print(integer_flag, "\n", mesg)
            calculation_results_df.loc[func_name, "popt"] = popt
            print(f"parameters = {popt}")
            calculation_results_df.loc[func_name, "pcov"] = pcov
            print(f"pcov =\n {pcov}")
            perr = np.sqrt(np.diag(pcov))
            calculation_results_df.loc[func_name, "perr"] = perr
            print(f"perr = {perr}\n")
            print(f"popt= {popt[1]}\n")
        else:
            (popt, pcov) = curve_fit(
                func,
                X_fact,
                Y_fact,
                p0=p0,
                method=methods_dict[func_name],
                full_output=full_output,
            )
            calculation_results_df.loc[func_name, "popt"] = popt
            print(f"parameters = {popt}")
            calculation_results_df.loc[func_name, "pcov"] = pcov
            print(f"pcov =\n {pcov}")
            perr = np.sqrt(np.diag(pcov))
            calculation_results_df.loc[func_name, "perr"] = perr
            print(f"perr = {perr}\n")
        Y_calc = func(X_fact, *popt)
        calculation_results_df.loc[func_name, "Y_calc"] = Y_calc
        print(Y_calc)

    # Построение графиков
    plt.figure()
    color_dict = {"linear": "blue", "quadratic": "green", "qubic": "slateblue"}

    linewidth_dict = {"linear": 2, "quadratic": 2, "qubic": 2}

    plt.rcParams["axes.titlesize"] = 14
    plt.rcParams["legend.fontsize"] = 12
    plt.rcParams["xtick.labelsize"] = 12
    plt.rcParams["ytick.labelsize"] = 12

    sns.scatterplot(x=X_fact, y=Y_fact, label="data", s=75, color="red")

    for func_name in model_list:
        label = func_name
        func = calculation_results_df.loc[func_name, "func"]
        popt = calculation_results_df.loc[func_name, "popt"]
        sns.lineplot(
            x=X_fact,
            y=func(X_fact, *popt),
            color=color_dict[func_name],
            linewidth=linewidth_dict[func_name],
            legend=True,
            label=label,
        )

    plt.show()


def calibration(E_He, counts):

    plt.figure()
    plt.vlines(
        x=E_He[0], ymin=0, ymax=50, colors=["red"], linestyle="dashed", linewidth=1
    )
    plt.vlines(
        x=E_He[1], ymin=0, ymax=50, colors=["blue"], linestyle="dashed", linewidth=1
    )
    plt.vlines(
        x=E_He[2], ymin=0, ymax=50, colors=["green"], linestyle="dashed", linewidth=1
    )
    plt.vlines(
        x=E_He[3], ymin=0, ymax=50, colors=["black"], linestyle="dashed", linewidth=1
    )
    plt.legend(["11C", "13N", "17F", "12C"])
    plt.axhline(y=12.1, color="grey", linewidth=2)
    plt.plot(Energy_exp, counts, color="black")
    plt.xlabel("E, MeV")
    plt.ylabel("counts")

    print("Ground state calibration")

    max = counts[len(counts) - 50]
    for i in range(len(counts) - 50, len(counts)):
        if counts[i] > max:
            max = counts[i]
            index = i

    E_He_change = [e[:] for e in E_He]
    delta = abs(Energy_exp[index] - E_He[0][0])
    for i in range(0, 4):
        k = 0
        for E in E_He_change[i]:
            E_He_change[i][k] = E - delta
            k += 1
    print("Сдвиг g.s на ", delta)

    # График калибровки по g.s

    plt.figure()
    plt.vlines(
        x=E_He_change[0],
        ymin=0,
        ymax=50,
        colors=["red"],
        linestyle="dashed",
        linewidth=1,
    )
    plt.vlines(
        x=E_He_change[1],
        ymin=0,
        ymax=50,
        colors=["blue"],
        linestyle="dashed",
        linewidth=1,
    )
    plt.vlines(
        x=E_He_change[2],
        ymin=0,
        ymax=50,
        colors=["green"],
        linestyle="dashed",
        linewidth=1,
    )
    plt.vlines(
        x=E_He_change[3],
        ymin=0,
        ymax=50,
        colors=["black"],
        linestyle="dashed",
        linewidth=1,
    )
    plt.legend(["11C", "13N", "17F", "12C"])
    plt.bar(Energy_exp, counts, align="edge", linewidth=1.0, edgecolor="k", width=0.1)
    plt.show(block=False)

    # Ручная калибровка
    print("Введите калибровку вручную")
    lvl = frame_particles.loc["11C", "Energy_lvl"]
    Calib = []
    for i in range(len(E_He_change[0])):
        print("Уровень: ", lvl[str(i)]["E_x"])
        Calib.append(float(input()))

    approximate(Calib, E_He_change[0])


def Kinematics(A, B, C, D, angles):
    E_Aparticle = 58
    lvl = frame_particles.loc[D, "Energy_lvl"]
    i = 0
    E_x = []
    while i < len(lvl):
        E_x.append(lvl[str(i)]["E_x"])
        i += 1
    Q_react = []
    for i in range(len(E_x)):
        Q_react.append(calculate_Q_react(A, B, C, D, frame_particles) - E_x[i])
    E_particle = []
    for i in range(len(Q_react)):
        E_particle.append(
            calculate_E_t(A, B, C, D, E_Aparticle, angles, frame_particles, Q_react[i])
        )
    return E_particle


def main():
    E_He = []
    with open("/Users/semenraydun/Desktop/NIRS/reactions.txt", "r") as file:
        for line in file:
            data = line.split()
            A = data[0]
            B = data[1]
            C = data[2]
            D = data[3]
            E_He.append(Kinematics(A, B, C, D, angles[0]))
    calibration(E_He, counts)


main()
