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

    #
    # Trajectory
    #
    t = sy.symbols("t", real=True)
    a10, a11, a12, a13 = sy.symbols("a10 a11 a12 a13", real=True)
    a20, a21, a22, a23 = sy.symbols("a20 a21 a22 a23", real=True)
    a30, a31, a32, a33 = sy.symbols("a30 a31 a32 a33", real=True)

    pi = sy.pi

    f1 = a10 + a11 * t + a12 * t**2 + a13 * t**3
    f2 = a20 + a21 * t + a22 * t**2 + a23 * t**3
    f3 = a30 + a31 * t + a32 * t**2 + a33 * t**3
    f1_d, f2_d, f3_d = sy.diff(f1, t), sy.diff(f2, t), sy.diff(f3, t)
    f1_dd, f2_dd, f3_dd = sy.diff(f1_d, t), sy.diff(f2_d, t), sy.diff(f3_d, t)

    t1, t2, t3, t4 = 0.0, 2.0, 4.0, 6.0
    theta1, theta2, theta3, theta4 = -pi / 2, 0, pi / 2, 0

    equ = sy.Matrix(
        [
            f1.subs(t, t1) - theta1,
            f1.subs(t, t2) - theta2,
            f2.subs(t, t2) - theta2,
            f2.subs(t, t3) - theta3,
            f3.subs(t, t3) - theta3,
            f3.subs(t, t4) - theta4,
            f1_d.subs(t, t1) - 0,
            f2_d.subs(t, t2) - f1_d.subs(t, t2),
            f3_d.subs(t, t3) - f2_d.subs(t, t3),
            f3_d.subs(t, t4) - 0,
            f2_dd.subs(t, t2) - f1_dd.subs(t, t2),
            f3_dd.subs(t, t3) - f2_dd.subs(t, t3),
        ]
    )
    q = [a10, a11, a12, a13, a20, a21, a22, a23, a30, a31, a32, a33]

    A = equ.jacobian(q)
    b = -equ.subs(
        [
            (a10, 0),
            (a11, 0),
            (a12, 0),
            (a13, 0),
            (a20, 0),
            (a21, 0),
            (a22, 0),
            (a23, 0),
            (a30, 0),
            (a31, 0),
            (a32, 0),
            (a33, 0),
        ]
    )
    x = A.inv() * b
    print("*" * 20, "Traj optimization", "*" * 20)
    print(f"A = {A}")
    print(f"b = {b}")
    print(f"a10 = {x[0]}")
    print(f"a11 = {x[1]}")
    print(f"a12 = {x[2]}")
    print(f"a13 = {x[3]}")
    print(f"a20 = {x[4]}")
    print(f"a21 = {x[5]}")
    print(f"a22 = {x[6]}")
    print(f"a23 = {x[7]}")
    print(f"a30 = {x[8]}")
    print(f"a31 = {x[9]}")
    print(f"a32 = {x[10]}")
    print(f"a33 = {x[11]}")
    print()


if __name__ == "__main__":
    main()
