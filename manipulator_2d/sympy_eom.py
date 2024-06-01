# Copyright 2024 @with-RL
# Reference from
#    - https://pab47.github.io/legs.html
#    - https://github.com/kimsooyoung/robotics_python
#
# Licensed under the Apache License, Version 2.0 (the "License");
#     http://www.apache.org/licenses/LICENSE-2.0

import sympy as sy


def main():
    theta1, theta2 = sy.symbols("theta1 theta2", real=True)
    l1, l2 = sy.symbols("l1 l2", real=True)

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
    H_12 = sy.Matrix(
        [
            [sy.cos(theta2), -sy.sin(theta2), l1],
            [sy.sin(theta2), sy.cos(theta2), 0],
            [0, 0, 1],
        ]
    )
    H_02 = sy.simplify(H_01 * H_12)
    print("*" * 20, "Homogeneous Matrix", "*" * 20)
    print(f"H_01: {H_01}")
    print(f"H_12: {H_12}")
    print(f"H_02: {H_02}")
    print()

    #
    # 2. Jacobian
    #
    q = [theta1, theta2]

    O_1 = H_01 * sy.Matrix([l1, 0, 1])
    O_1 = sy.Matrix([O_1[0], O_1[1]])
    J_1 = O_1.jacobian(q)

    E_0 = H_02 * sy.Matrix([l2, 0, 1])
    E_0 = sy.Matrix([E_0[0], E_0[1]])
    J_2 = E_0.jacobian(q)

    print("*" * 20, "Jacobian", "*" * 20)
    print(f"O_1: {O_1}")
    print(f"J_1: {J_1}")
    print(f"E_0: {E_0}")
    print(f"J_2: {J_2}")
    print()


if __name__ == "__main__":
    main()
