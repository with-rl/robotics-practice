# Copyright 2024 @with-RL
# Reference from
#    - https://pab47.github.io/legs.html
#    - https://github.com/kimsooyoung/robotics_python
#
# Licensed under the Apache License, Version 2.0 (the "License");
#     http://www.apache.org/licenses/LICENSE-2.0

import sympy as sy


def main():
    theta1 = sy.symbols("theta1", real=True)
    q = sy.Matrix([theta1])
    theta1_d = sy.symbols("theta1_d", real=True)
    q_d = sy.Matrix([theta1_d])
    theta1_dd = sy.symbols("theta1_dd", real=True)
    q_dd = sy.Matrix([theta1_dd])

    c1 = sy.symbols("c1", real=True)
    l1 = sy.symbols("l1", real=True)
    m1 = sy.symbols("m1", real=True)
    I1 = sy.symbols("I1", real=True)

    g = sy.symbols("g", real=True)

    #
    # 1. Homogeneous Matrix
    #
    H_01 = sy.Matrix(
        [
            [sy.cos(theta1), -sy.sin(theta1), 0],
            [sy.sin(theta1), sy.cos(theta1), 0],
            [0, 0, 1],
        ]
    )
    G1 = H_01 * sy.Matrix([c1, 0, 1])
    G1_xy = sy.Matrix([G1[0], G1[1]])
    print("*" * 20, "Homogeneous Matrix", "*" * 20)
    print(f"H_01: {H_01}")
    print(f"G1_xy: {G1_xy}")
    print()

    #
    # Lagrangian
    #
    v_G1 = G1_xy.jacobian(q) * q_d
    T = sy.simplify(0.5 * m1 * v_G1.dot(v_G1) + 0.5 * I1 * theta1_d**2)
    V = sy.simplify(m1 * g * G1[1])
    L = sy.simplify(T - V)
    print("*" * 20, "Lagrangian", "*" * 20)
    print(f"v_G1: {v_G1}")
    print(f"T: {T}")
    print(f"V: {V}")
    print(f"L: {L}")
    print()

    #
    # Euler lagrange
    #
    dL_dq_d = []
    dt_dL_dq_d = []
    dL_dq = []
    EOM = []

    for i in range(len(q)):
        dL_dq_d.append(sy.diff(L, q_d[i]))

        temp = 0
        for j in range(len(q)):
            temp += (
                sy.diff(dL_dq_d[i], q[j]) * q_d[j]
                + sy.diff(dL_dq_d[i], q_d[j]) * q_dd[j]
            )
        dt_dL_dq_d.append(temp)
        dL_dq.append(sy.diff(L, q[i]))
        EOM.append(dt_dL_dq_d[i] - dL_dq[i])
    EOM = sy.simplify(sy.Matrix(EOM))

    print("*" * 20, "Euler lagrange", "*" * 20)
    for i in range(len(EOM)):
        print(f"EOM[{i}]: {EOM[i]}")
    print()

    #
    # EOM M, C, G format
    #
    print("*" * 20, "M, C, G format", "*" * 20)
    M = EOM.jacobian(q_dd)
    b = EOM.subs([(theta1_dd, 0)])
    G = b.subs([(theta1_d, 0)])
    C = b - G

    for i in range(len(M)):
        print(f"M[{i}]: {M[i]}")
    for i in range(len(C)):
        print(f"C[{i}]: {C[i]}")
    for i in range(len(G)):
        print(f"G[{i}]: {G[i]}")
    print()


if __name__ == "__main__":
    main()
