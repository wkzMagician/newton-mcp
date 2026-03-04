# Computer Animation - Exercise

## Overview

This repository contains the code for the computer animation exercise. The code uses [Newton@2a6df66](https://github.com/newton-physics/newton/tree/2a6df66595c05d655876a972df06ef810816a7bc) as a backend for the physics simulation.

## Installation

1. Install **Git** and [**uv**](https://docs.astral.sh/uv/getting-started/installation/) on your system.
   On Windows, **uv** can be simply installed by running the following command in PowerShell:
   ```powershell
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```
2. Clone this respository:
   ```bash
   git clone http://dalab.se.sjtu.edu.cn/gitlab/courses/ca-framework-2026.git
   ```
3. Install dependencies:
   ```bash
   uv sync --extra examples
   ```
4. Use any IDE (PyCharm is recommended) of your choice to open the project and run the script in `examples/ca_exercises/` directory. PyCharm will automatically recognize the virtual environments, if not, you can activate it using the command below:
   ```bash
   .\.venv\Scripts\activate
   ```

## Handin

**You only need to modify the `TODO` sections in the code.** You can run the code to test your implementation, but please do not modify the code structure or add any new files.

After you finish one exercise, **You only need to submit the modified `.py` files in `newton/_src/solvers/ca_exercises/`.** Please compress these files into a zip file and submit it to Canvas. The file name should be in the format of `ca_exercise<n>_<student_id>.zip`.

Notes: each exercise may have its own additional requirements, please pay attention.
