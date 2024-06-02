# Copyright 2024 @with-RL
# Reference from
#    - https://pab47.github.io/legs.html
#    - https://github.com/kimsooyoung/robotics_python
#
# Licensed under the Apache License, Version 2.0 (the "License");
#     http://www.apache.org/licenses/LICENSE-2.0

import sympy as sy


def main():
    omega_R, omega_L = sy.symbols("omega_R omega_L", real=True)
    r, b = sy.symbols("r b", real=True)
    vel_c, omega_c = sy.symbols("vel_c omega_c", real=True)

    A = sy.Matrix([[r / 2, r / 2], [r / (2 * b), -r / (2 * b)]])
    x = sy.Matrix([omega_R, omega_L])
    b = sy.Matrix([vel_c, omega_c])

    print(A)
    print(A * x)
    print(b)


if __name__ == "__main__":
    main()
