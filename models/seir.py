import numpy as np
import matplotlib.pyplot as plt


def seir1(x, t):
    S, E, Is, Ias, Q, Qprime, C, Rwd, D, Rd = x
    n = sum(x)
    b1 = 0.5
    b2 = 0.1458
    b3 = 0.1458
    b4 = 0.05
    u = 0.5
    v = 0.05
    e = 0.5
    r = 0.3
    g = 0.0001  # rate at which a fraction of recovered individuals lose their immunity
    f = 0.2  # rate at which fraction of carriers gets re-infected
    p = 0.01
    a = 0.42  # disease transmission rate
    l1 = 0.25
    l2 = 0.2
    l3 = 0.1
    l4 = 0.05
    l5 = 0.2
    n1 = 0.1
    n2 = 0.1
    n3 = 0.05
    n4 = 0.01
    n5 = 0.1
    #S, E, Is, Ias, Q, Qprime, C, Rwd, D, Rd = x
    dx = np.zeros(11)
    dx[0] = -a * (S * (Is + Ias + C) / n) + (g * Rwd)
    dx[1] = a * (S * (Is + Ias + C) / n) - (u * E)
    dx[2] = (r * u * E) - (e * Is) + (f * C) - (l1 * Is) - (n1 * Is)
    dx[3] = ((1-r) * u * E) - (b3 * Ias) - (l3 * Ias) - (n3 * Ias)
    dx[4] = (e * Is) - (b1 * Q) - (v * Q) - (p * Q) - (l2 * Q) - (n2 * Q)
    dx[5] = (p * Q) - (b4 * Qprime) - (l5 * Qprime) - (n5 * Qprime)
    dx[6] = (v * Q) - (f * C) - (b2 * C) - (l4 * C) - (n4 * C)
    dx[7] = (b1 * Q) + (b3 * Ias) + (b2 * C) - (g * Rwd)
    dx[8] = (l1 * Is) + (l2 * Q) + (l3 * Ias) + (l4 * C) + (l5 * Qprime)
    dx[9] = (n1 * Is) + (n2 * Q) + (n3 * Ias) + (n4 * C) + (n5 * Qprime)
    return dx


def seir2(x, t):
    S, E, I, Q, R, D, V = x
    # Vaccination rate is vaccinated/population/days
    v = 2300  # New births and new residents
    b1 = 8.58 * 10**-9  # Transmission rate before intervention
    b2 = 3.43 * 10**-9  # Transmission rate during and after intervention
    a = 3.5 * 10**-4  # Vaccination rate
    u = 3 * 10**-5  # Natural death rate
    gamma = 1/5.5  # Incubation period
    sigma = 0.05  # Vaccine inefficacy
    delta = 1/3.8  # Infection time
    k = 0.014  # Case fatality rate
    lamb = 1/10  # Recovery time
    p = 1/15  # Time until death
    dx = np.zeros(7)
    dx[0] = (v) - (b2 * S * I) - (a * S) - (u * S)
    dx[1] = (b2 * S * I) - (gamma * E) + (sigma * b2 * V * I) - (u * E)
    dx[2] = (gamma * E) - (delta * I) - (u * I)
    dx[3] = (delta * I) - ((1-k) * lamb * Q) - (k * p * Q) - (u * Q)
    dx[4] = ((1 - k) * lamb * Q) - (u * R)
    dx[5] = k * p * Q
    dx[6] = (a * S) - (sigma * b2 * V * I) - (u * V)
    return dx
