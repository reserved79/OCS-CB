

import scipy.special as sc_special
import numpy as np
from numpy.random import random as rand




def obl_cuckoo_search(n, m, fit_func, lower_boundary, upper_boundary, iter_num=50, pa=0.25, beta=1.5, step_size=0.1):

    nests = generate_nests(n, m, lower_boundary, upper_boundary)
    obl_nests = ob_position(nests, n, m, upper_boundary, lower_boundary)
    nests_total = np.concatenate((nests, obl_nests), axis=0)

    fitness_total = calc_fitness(fit_func, nests_total)
    stord = sorted(range(len(fitness_total)), key=lambda k: fitness_total[k], reverse=False)
    nests = nests_total[stord[0: n]]

    fitness = calc_fitness(fit_func, nests)
    best_t = np.ones((iter_num, 1))

    best_nest_index = np.argmin(fitness)
    best_fitness = fitness[best_nest_index]
    best_nest = nests[best_nest_index].copy()

    for i in range(iter_num):
        print(i, '次迭代：')
        nests = update_nests(fit_func, lower_boundary, upper_boundary, nests, best_nest, fitness, step_size)
        nests = abandon_nests(nests, lower_boundary, upper_boundary, pa)

        fitness = calc_fitness(fit_func, nests)

        min_nest_index = np.argmin(fitness)
        min_fitness = fitness[min_nest_index]
        min_nest = nests[min_nest_index]

        if (min_fitness < best_fitness):
            best_nest = min_nest.copy()
            best_fitness = min_fitness

        best_t[i] = best_fitness
        # print(i, '次迭代：')
        # print(best_fitness)

    return (best_nest, best_fitness, best_t)


def generate_nests(n, m, lower_boundary, upper_boundary):

    lower_boundary = np.array(lower_boundary)
    upper_boundary = np.array(upper_boundary)
    nests = np.empty((n, m))

    for each_nest in range(n):
        nests[each_nest] = lower_boundary + np.array([np.random.rand() for _ in range(m)]) * (
                    upper_boundary - lower_boundary)

    return nests


def update_nests(fit_func, lower_boundary, upper_boundary, nests, best_nest, fitness, step_coefficient):

    lower_boundary = np.array(lower_boundary)
    upper_boundary = np.array(upper_boundary)
    n, m = nests.shape

    steps = levy_flight(n, m, 1.5)
    new_nests = nests.copy()
    obl_nests = nests.copy()

    for each_nest in range(n):

        step_size = step_coefficient * steps[each_nest] * (nests[each_nest] - best_nest)
        step_direction = np.random.rand(m)
        new_nests[each_nest] += step_size * step_direction

        new_nests[each_nest][new_nests[each_nest] < lower_boundary] = lower_boundary[
            new_nests[each_nest] < lower_boundary]
        new_nests[each_nest][new_nests[each_nest] > upper_boundary] = upper_boundary[
            new_nests[each_nest] > upper_boundary]

    if (np.random.rand() < 0.3):
        for each_nest in range(n):
            obl_nests[each_nest] = ob_position1(new_nests[each_nest], n, m, upper_boundary, lower_boundary)
            obl_nests[each_nest][obl_nests[each_nest] < lower_boundary] = lower_boundary[
                obl_nests[each_nest] < lower_boundary]
            obl_nests[each_nest][obl_nests[each_nest] > upper_boundary] = upper_boundary[
                obl_nests[each_nest] > upper_boundary]

            if fit_func(obl_nests[each_nest]) < fit_func(new_nests[each_nest]):
                new_nests[each_nest] = obl_nests[each_nest]

    new_fitness = calc_fitness(fit_func, new_nests)
    nests[new_fitness < fitness] = new_nests[new_fitness < fitness]

    return nests


def abandon_nests(nests, lower_boundary, upper_boundary, pa):

    lower_boundary = np.array(lower_boundary)
    upper_boundary = np.array(upper_boundary)
    n, m = nests.shape
    obl_nests = nests.copy()

    for each_nest in range(n):
        if (np.random.rand() > pa):
            step_size = np.random.rand() * (nests[np.random.randint(0, n)] - nests[np.random.randint(0, n)])
            nests[each_nest] += step_size

            nests[each_nest][nests[each_nest] < lower_boundary] = lower_boundary[nests[each_nest] < lower_boundary]
            nests[each_nest][nests[each_nest] > upper_boundary] = upper_boundary[nests[each_nest] > upper_boundary]

    return nests


def levy_flight(n, m, beta):

    sigma_u = (sc_special.gamma(1 + beta) * np.sin(np.pi * beta / 2) / (
                sc_special.gamma((1 + beta) / 2) * beta * (2 ** ((beta - 1) / 2)))) ** (1 / beta)
    sigma_v = 1

    u = np.random.normal(0, sigma_u, (n, m))
    v = np.random.normal(0, sigma_v, (n, m))

    steps = u / ((np.abs(v)) ** (1 / beta))

    return steps


def calc_fitness(fit_func, nests):

    n, m = nests.shape
    fitness = np.empty(n)

    for each_nest in range(n):
        fitness[each_nest] = fit_func(nests[each_nest])

    return fitness




def ob_position(positions, search_agents_no, dim, ub, lb):
    boundary_no = np.shape(ub)[0]
    if boundary_no == 1:
        positions = (ub + lb) - positions
        return positions
    elif boundary_no > 1:
        for i in range(dim):
            ub_i = ub[i]
            lb_i = lb[i]
            positions[:, i] = (ub_i + lb_i) - positions[:, i]
    return positions

def ob_position1(positions, search_agents_no, dim, ub, lb):
    boundary_no = np.shape(ub)[0]
    if boundary_no == 1:
        positions = (ub + lb) - positions
        return positions
    elif boundary_no > 1:
        for i in range(dim):
            ub_i = ub[i]
            lb_i = lb[i]
            positions[i] = (ub_i + lb_i) - positions[i]
    return positions

