import copy
from pathlib import Path

import cvxpy as cvx
import numpy as np
import pandas as pd

np.random.seed(0)


class ResourceAllocation():
    def __init__(
        self,
        agent_num: int,
        delay_max: int,
        is_bandit: bool,
        smoothing_parameter_f: float = 10**-1,
        smoothing_parameter_g: float = 10**-1,
        maximum_random_num_bandit: tuple[int] = 100,
        constraint_set_upper: tuple[int] = (300, 350),
        constraint_set_lower: tuple[int] = (50, 80),
        constraint_dimensions: int = 2,
        average_demand: int = 290,
        initial_x_range: tuple[int] = (200, 300),
        a_range: tuple[int] = (0.025, 0.03),
        b_range: tuple[int] = (15, 20),
        c_range: tuple[int] = (25, 30),
        objective_weight: int = 1,
        constraint_weight: int = 1
    ) -> None:
        self.agent_num = agent_num
        self.delay_max = delay_max

        self.is_bandit = is_bandit
        self.smoothing_parameter_f = smoothing_parameter_f
        self.smoothing_parameter_g = smoothing_parameter_g
        self.maximum_random_num_bandit = maximum_random_num_bandit

        self.constraint_dimensions = constraint_dimensions

        self.base_demand = average_demand * agent_num
        self.calculate_current_demand(iteration=0)

        self.objective_weight = objective_weight
        self.constraint_weight = constraint_weight

        self.k = 0

        self.expand_agent_num = agent_num * (delay_max + 1)

        self.a_range = a_range
        self.b_range = b_range
        self.c_range = c_range

        # 各変数の初期化
        self._init_variables(initial_x_range)
        self.pre_g = self.calculate_g(self.x)

        self.x_upper = np.random.randint(*constraint_set_upper, agent_num)
        self.x_lower = np.random.randint(*constraint_set_lower, agent_num)

        self.regret = 0
        self.constraint_violation = [0 for _ in range(constraint_dimensions)]

        # For storing some variables.
        self.x_list = []
        self.optimal_x_list = []
        self.regret_list = []
        self.constraint_violation_list = []

        # print(f'{self.w=}')
        # print(f'{self.hat_lambda=}')
        # print(f'{self.hat_y=}')

        # print(f'{self.s=}')
        # print(f'{self.x=}')

        # print(f'{self.lambda_=}')
        # print(f'{self.y=}')

    def __call__(self, iteration: int, weight_matrix: np.ndarray) -> tuple[np.ndarray]:
        self.calculate_coefficient_objective()
        self.calculate_current_demand(iteration)

        # print(f'{self.a=}')
        # print(f'{self.b=}')
        # print(f'{self.c=}')

        self.update(iteration, weight_matrix)

        optimal_cost, optimal_x = self.calculate_optimal_value()
        local_cost = self.calculate_f(self.x)

        self.regret += np.abs(np.sum(local_cost) - optimal_cost)

        self.constraint_violation += np.sum(self.g, axis=0)
        constraint_violation_tmp = np.linalg.norm(self.constraint_violation[self.constraint_violation >= 0])
        if iteration != 0:
            self.regret_list.append(self.regret / iteration)
            self.constraint_violation_list.append(constraint_violation_tmp / iteration)
        else:
            self.regret_list.append(self.regret)
            self.constraint_violation_list.append(constraint_violation_tmp)

        self.pre_g = copy.deepcopy(self.g)

        self.x_list.append(self.x)
        self.optimal_x_list.append(optimal_x)

        return self.regret_list, self.constraint_violation_list

    def calculate_current_demand(self, iteration):
        self.current_demand = self.base_demand + 10 * np.sin(iteration * np.pi / 64) + np.random.normal(0, 0.1)

    def calculate_coefficient_objective(self):
        self.a = np.random.uniform(*self.a_range, self.agent_num)
        self.b = np.random.uniform(*self.b_range, self.agent_num)
        self.c = np.random.uniform(*self.c_range, self.agent_num)

    def _init_variables(self, initial_x_range):
        self.w = np.ones(self.expand_agent_num)
        self.x = np.random.randint(*initial_x_range, self.agent_num)
        self.lambda_ = np.zeros((self.expand_agent_num, self.constraint_dimensions))

        self.g_tmp = np.zeros((self.expand_agent_num, self.constraint_dimensions))
        self.gradient_g_tmp = np.zeros((self.expand_agent_num, self.constraint_dimensions))
        self.y = self.calculate_g(self.x)

    def calculate_f(self, x):
        return self.objective_weight * (self.a * x ** 2 + self.b * x + self.c)

    def calculate_gradient_f(self, x: np.ndarray):
        if self.is_bandit:
            # - self.maximum_random_num_bandit以上self.maximum_random_num_bandit以下の範囲の一様分布から乱数を生成
            random_number = 2 * self.maximum_random_num_bandit * np.random.rand(self.agent_num) - self.maximum_random_num_bandit
            diff = random_number * self.smoothing_parameter_f
            gradient_f = random_number * (self.calculate_f(x + diff) - self.calculate_f(x - diff)) / (2 * self.smoothing_parameter_f)
        else:
            gradient_f = 2 * self.a * x + self.b

        return gradient_f * self.objective_weight

    def calculate_g(self, x: np.ndarray):
        self.g_tmp[:self.agent_num] = np.array(
            [
                x - (self.current_demand / self.agent_num + 10),
                -x + (self.current_demand / self.agent_num - 10)
            ]
        ).T
        return self.g_tmp * self.constraint_weight

    def calculate_gradient_g(self, x: np.ndarray):
        if self.is_bandit:
            random_number = 2 * self.maximum_random_num_bandit * np.random.rand(self.expand_agent_num) - self.maximum_random_num_bandit
            diff = random_number[:self.agent_num] * self.smoothing_parameter_f
            self.gradient_g_tmp = random_number.reshape(-1, 1).repeat(self.constraint_dimensions, axis=1)\
                * (self.calculate_g(x + diff) - self.calculate_g(x - diff)) / (2 * self.smoothing_parameter_f)
        else:
            self.gradient_g_tmp[:self.agent_num] = np.array([[1, -1]] * self.agent_num)

        return self.gradient_g_tmp * self.constraint_weight

    def calculate_alpha(self, iteration: int):
        if iteration != 0:
            return 1 / np.sqrt(iteration)
        else:
            return 1

    def calculate_beta(self, iteration: int):
        if iteration != 0:
            return 1 / np.sqrt(iteration)
        else:
            return 1

    def projection_x(self, x: np.ndarray):
        # Check if x in the constraint set.
        # Check lower bound.
        x = np.where(x > self.x_lower, x, self.x_lower)
        # Check upper bound.
        x = np.where(x < self.x_upper, x, self.x_upper)
        return x

    def projection_lambda(self, lambda_: np.ndarray):
        lambda_ = np.where(lambda_ > 0, lambda_, 0)
        return lambda_

    def update(self, iteration: int, weight_matrix: np.ndarray):
        self.w = np.dot(weight_matrix, self.w)
        self.hat_lambda = np.dot(weight_matrix, self.lambda_)
        self.hat_y = np.dot(weight_matrix, self.y)

        denominator_w = self.w.reshape(-1, 1).repeat(2, axis=1)
        alpha = self.calculate_alpha(iteration)
        beta = self.calculate_beta(iteration)

        self.s = self.calculate_gradient_f(self.x)\
            + np.sum((self.calculate_gradient_g(self.x) * self.hat_lambda / denominator_w), axis=1)[:self.agent_num]
        self.x = self.projection_x(self.x - alpha * self.s)

        self.lambda_ = self.projection_lambda(self.hat_lambda + alpha * (self.hat_y / denominator_w - beta * self.hat_lambda))
        self.g = self.calculate_g(self.x)
        self.y = self.hat_y + self.g - self.pre_g

    def calculate_optimal_value(self):
        x = cvx.Variable(self.agent_num)

        objective = cvx.Minimize(self.a @ cvx.square(x) + self.b @ x + np.sum(self.c))
        # Add constraint set of x.
        constraints = [self.x_lower <= x, x <= self.x_upper]
        # Add constraints about the sum of x.
        constraints += [(cvx.sum(x) - self.current_demand + 10) * self.constraint_weight <= 0]
        constraints += [(self.current_demand - 10 - cvx.sum(x)) * self.constraint_weight <= 0]

        problem = cvx.Problem(objective, constraints)
        optimal_cost = problem.solve()
        optimal_x = x.value

        return optimal_cost, optimal_x

    def save_data_as_csv(self, save_dir: Path, file_name='result.csv') -> None:
        if file_name.split('.')[-1] != 'csv':
            raise ValueError('Invalid suffix is set. Please set to csv.')

        save_dir.mkdir(exist_ok=True, parents=True)

        df = pd.DataFrame(
                {
                    'x': self.x_list,
                    'optimal_x': self.optimal_x_list,
                    'regret': self.regret_list,
                    'constraint_violation': self.constraint_violation_list,
                }
            )

        df.to_csv(save_dir / file_name, index=False)
