import time

import cvxpy
import numpy as np

np.random.seed(0)
# 制約式の次元
m = 2
# kappaの値
kappa = 0.24


class Algorithm:
    def __init__(self, x, a, b, c, delay_max, N, iteration, stepsize_a=1, stepsize_b=1, objective_weight=1, weight_constraints=1):
        self.delay_max = delay_max
        self.N = N
        self.iteration = iteration
        self.k = 0

        self.size_a = stepsize_a
        self.size_b = stepsize_b

        self.n = self.delay_max * N

        # 目的関数，制約関数の重み
        self.objective_weight = objective_weight
        self.constraint_weight = weight_constraints

        # コスト関数の係数
        # self.a = np.random.uniform(0.025, 0.03, N)
        # self.b = np.random.uniform(15, 20, N)
        # self.c = np.random.uniform(25, 30, N)

        # 制約集合(各x[i]がとり得る範囲)
        self.p_i_max = np.random.uniform(300, 350, N)
        self.p_i_min = np.random.uniform(50, 80, N)

        # 制約関数のd？
        # demand_list = [10 * np.sin(i * np.pi / 64) for i in range(iteration)]
        # self.P = (290 * self.N * np.ones(iteration)) + demand_list + np.random.normal(0, 0.1, iteration)
        self.P = 290 * self.N

        # 変数の初期値
        # self.x = np.random.randint(200, 300, N)
        self.x = x

        # 各値の初期化
        self.w = np.ones(self.n)

        # self.mu = np.zeros((m, self.N))
        self.mu = np.zeros((m, self.n))
        self.hat_mu = np.zeros((m, self.n))
        # self.hat_mu = np.zeros(N)

        self.s = np.zeros(N)

        self.g = np.zeros((m, self.N))

        self.g = self._g(self.x)
        self.g_old = self.g
        # self.y = self.g
        self.hat_y = np.zeros((m, self.n))
        self.y = np.zeros((m, self.n))
        for i in range(self.N):
            for l_ in range(m):
                self.y[l_][i] = self.hat_y[l_][i] + self.g[l_][i]

        # リグレット
        self.Reg = 0
        self.result_Reg = [0]

        # 制約違反
        self.Regc = 0
        self.result_Regc = [0]

        self.val = 0

        # 制約違反の値
        self.constraint_value = [0, 0]
        self.sum_g = [0, 0]

        # print(f'{self.w=}')
        # print(f'{self.hat_lambda=}')
        # print(f'{self.hat_y=}')

        # print(f'{self.s=}')
        # print(f'{self.x=}')

        # print(f'{self.mu.T=}')
        # print(f'{self.y.T=}')

    def _f(self, x):
        return self.a * x ** 2 + self.b * x + self.c

    def _g(self, x):
        return np.array([
            (x - (self.P / self.N + 10)) * self.constraint_weight,
            (-x + (self.P / self.N - 10)) * self.constraint_weight
        ])

    def _D_f(self, x):
        # print('Calculate gradient')
        # print(f'{self.a=}')
        # print(f'{self.b=}')
        return 2 * self.a * x + self.b

    def _D_g(self):
        return np.array([self.constraint_weight, -self.constraint_weight])

    def _alpha(self):
        if self.k != 0:
            return self.size_a / np.sqrt(self.k)
        else:
            return self.size_a

    def _beta(self):
        # self.size_b = 0.02
        if self.k != 0:
            return self.size_b / ((self.k)**kappa)
        else:
            return self.size_b

    def _projection_mu(self):
        # print(" ")
        # print(self.w)
        for j in range(m):
            for i in range(self.n):
                if i == j == 0:
                    print(f'{self.hat_mu[:, i]=}')
                    print('alpha=', self._alpha())
                    print(f'{self.hat_y[:, i]=}')
                    print(f'{self.w[i]=}')
                    print('beta=', self._beta())
                    print(f'{self.hat_mu[:, i]=}')
                self.mu[j][i] = self.hat_mu[j][i] + self._alpha() * ((self.hat_y[j][i] / self.w[i]) - self._beta() * self.hat_mu[j][i])

        for i in range(self.n):
            if self.mu[0][i] >= 0:
                if self.mu[1][i] < 0:
                    self.mu[1][i] = 0
            else:
                if self.mu[1][i] >= 0:
                    self.mu[0][i] = 0
                else:
                    self.mu[0][i] = 0
                    self.mu[1][i] = 0

    def _projection_x(self):
        self.x = self.x - self._alpha() * self.s

        flag = 0

        for i in range(self.N):
            if self.p_i_min[i] < self.x[i] and self.x[i] < self.p_i_max[i]:
                pass
            elif self.x[i] <= self.p_i_min[i]:
                self.x[i] = self.p_i_min[i]
            elif self.p_i_max[i] <= self.x[i]:
                self.x[i] = self.p_i_max[i]
            else:
                flag = 1

        if flag == 1:
            print(' ')
            print('x : {}'.format(self.x))
            print('s : {}'.format(self.s))
            print('d_f : {}'.format(self.d_f))
            print('d_g : {}'.format(self.d_g))
            print('hat_mu : {}'.format(self.hat_mu))
            print('hat_y : {}'.format(self.hat_y))
            print('mu : {}'.format(self.mu))
            print('w : {}'.format(self.w))

            time.sleep(10)

    def update(self, k, weight_matrix, a, b, c):
        self.k = k
        self.weight_matrix = weight_matrix

        # 目的関数の係数をupdate
        # self.a = np.random.uniform(0.025, 0.03, self.N)
        # self.b = np.random.uniform(15, 20, self.N)
        # self.c = np.random.uniform(25, 30, self.N)
        # print(f'{self.a=}')
        # print(f'{self.b=}')
        # print(f'{self.c=}')
        self.a = a
        self.b = b
        self.c = c

        # 制約値をupdate
        self.P = 290 * self.N + 10 * np.sin(k * np.pi / 64)

        # 勾配の計算
        self.d_f = self._D_f(self.x)
        self.d_g = self._D_g()

        # 各変数の更新
        self.w = np.dot(weight_matrix, self.w)

        for l_ in range(m):
            self.hat_mu[l_] = np.dot(weight_matrix, self.mu[l_])
            self.hat_y[l_] = np.dot(weight_matrix, self.y[l_])

        gradient_g_plus = []
        for i in range(self.N):
            self.s[i] = self.d_f[i] + np.dot(self.d_g, (self.hat_mu.T[i] / self.w[i]))
            gradient_g_plus.append(np.dot(self.d_g, (self.hat_mu.T[i] / self.w[i])))

        # xの計算
        self._projection_x()

        # muの計算
        self._projection_mu()

        # yの計算
        g_old = np.copy(self.g)
        self.g = self._g(self.x)

        self.y = np.copy(self.hat_y)
        for i in range(self.N):
            for l_ in range(m):
                self.y[l_][i] = self.hat_y[l_][i] + self.g[l_][i] - g_old[l_][i]

        # 最適値の計算
        f_opt, xc = self._calc_optimal_value()
        self.f = self._f(self.x)

        # print(' ')
        # print(f'        x: {self.x}')
        # print(f'optimal x: {xc}')

        # リグレットの計算
        self.Reg += np.abs(np.sum(self.f) - f_opt)
        if self.k != 0:
            # self.result_Reg.append(np.abs(self.Reg / self.k))
            self.result_Reg.append(self.Reg / self.k)
        else:
            # self.result_Reg.append(np.abs(self.Reg))
            self.result_Reg.append(self.Reg)

        # 制約違反の計算
        g = np.copy(self.g)

        # sum_g = np.sum(g, axis=1)
        # 各時刻，各エージェントの制約式の値の和
        self.Regc += np.sum(g, axis=1)
        # 正のところだけ取り出して，ノルムを計算
        regc = np.linalg.norm(self.Regc[self.Regc >= 0])
        # print(' ')
        # print(regc / self.k)
        # self.Regc += np.linalg.norm(sum_g[sum_g >= 0])
        # for i in range(m):
        #     tmp = 0
        #     for j in range(self.N):
        #         tmp += g[i][j]
        #     if tmp > 0:
        #         self.Regc += tmp

        # print(self._alpha())
        # print(self._beta())

        # print(f'{self.w=}')
        print(f'{self.hat_mu.T=}')
        # print(f'{self.hat_y.T=}')

        # print(f'{self.s=}')
        # print(f'{self.x=}')

        print(f'{self.mu.T=}')
        # print(f'{self.y.T=}')

        # print(f'{self.d_f=}')
        # print(f'{self.d_g=}')
        # print(f'{gradient_g_plus=}')

        if self.k != 0:
            self.result_Regc.append(regc / self.k)
        else:
            self.result_Regc.append(regc)

        self.val = self.Regc

        # if (self.k % fig_interval == 0 and self.k != 0) or self.k + 1 == self.iteration:
        #     print(" ")
        #     print("x : {}".format(self.x))
        #     print("xc : {}".format(xc))
        #     print("s : {}".format(self.s))
        #     print("d_f : {}".format(self.d_f))
        #     print("d_g : {}".format(self.d_g))
        #     print("hat_mu : {}".format(self.hat_mu))
        #     print("mu_old : {}".format(self.mu_old))
        #     print("hat_y : {}".format(self.hat_y))
        #     print("y_old : {}".format(self.y_old))
        #     print("mu : {}".format(self.mu))
        #     print("w : {}".format(self.w))
        #     print("f : {}".format(self.f))
        #     print("sum f : {}".format(np.sum(self.f)))
        #     print("fc : {}".format(f_opt))
        #     print("Reg : {}".format(self.Reg))
        #     print("add reg : {}".format(self.Reg / self.k))

        return self.result_Reg, self.result_Regc

    def _calc_optimal_value(self):
        xcvx = cvxpy.Variable(self.N)

        objective = cvxpy.Minimize(0)
        constraints = []
        for i in range(self.N):
            objective += cvxpy.Minimize(self.a[i] * cvxpy.square(xcvx[i]) + self.b[i] * xcvx[i] + self.c[i])
            constraints += [self.p_i_min[i] <= xcvx[i], xcvx[i] <= self.p_i_max[i]]
        constraints += [(cvxpy.sum(xcvx) - (self.P + 10 * self.N)) * self.constraint_weight <= 0]
        constraints += [(-cvxpy.sum(xcvx) + (self.P - 10 * self.N)) * self.constraint_weight <= 0]
        # constraints += [cvxpy.sum(xcvx) <= self.P + 10 * self.N - self.constraint_value[0]]
        # constraints += [cvxpy.sum(xcvx) >= self.P - 10 * self.N - self.constraint_value[1]]

        prob = cvxpy.Problem(objective, constraints)
        fopt = prob.solve()
        # print(fopt)
        xcvx_solved = xcvx.value
        # print(xcvx_solved)
        # print(np.sum(xcvx_solved))

        # 制約関数の累積値の計算用
        # self.constraint_value += np.sum(self._g(xcvx_solved), axis=1)
        # print(f'制約値: {self.constraint_value}')

        return fopt, xcvx_solved
