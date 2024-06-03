# Copyright 2024 @with-RL
# Reference from
#    - https://pab47.github.io/legs.html
#    - https://github.com/kimsooyoung/robotics_python
#
# Licensed under the Apache License, Version 2.0 (the "License");
#     http://www.apache.org/licenses/LICENSE-2.0

import sympy as sy


def main():
    x, y = sy.symbols("x y", real=True)
    x_d, y_d = sy.symbols("x_d y_d", real=True)
    x_dd, y_dd = sy.symbols("x_dd y_dd", real=True)
    m, c, g = sy.symbols("m c g", real=True)

    #
    # Lagrangian
    #
    T = m * (x_d**2 + y_d**2) / 2
    V = m * g * y
    L = T - V
    print("*" * 20, "Lagrangian", "*" * 20)
    print(f"T: {T}")
    print(f"V: {V}")
    print(f"L: {L}")
    print()

    #
    # Euler lagrange
    #
    v = sy.sqrt(x_d**2 + y_d**2)
    Fx = -c * x_d * v
    Fy = -c * y_d * v

    dL_dx_d = sy.diff(L, x_d)
    dt_dL_dx_d = (
        sy.diff(dL_dx_d, x) * x_d
        + sy.diff(dL_dx_d, x_d) * x_dd
        + sy.diff(dL_dx_d, y) * y_d
        + sy.diff(dL_dx_d, y_d) * y_dd
    )
    dL_dx = sy.diff(L, x)
    EOM_x = dt_dL_dx_d - dL_dx - Fx
    EOM_x = sy.solve(EOM_x, x_dd)

    dL_dy_d = sy.diff(L, y_d)
    dt_dL_dy_d = (
        sy.diff(dL_dy_d, x) * x_d
        + sy.diff(dL_dy_d, x_d) * x_dd
        + sy.diff(dL_dy_d, y) * y_d
        + sy.diff(dL_dy_d, y_d) * y_dd
    )
    dL_dy = sy.diff(L, y)
    EOM_y = dt_dL_dy_d - dL_dy - Fy
    EOM_y = sy.solve(EOM_y, y_dd)

    print("*" * 20, "Euler lagrange", "*" * 20)
    print(f"Fx: {Fx}")
    print(f"Fy: {Fy}")
    print(f"EOM_x: {EOM_x}")
    print(f"EOM_y: {EOM_y}")


if __name__ == "__main__":
    main()
