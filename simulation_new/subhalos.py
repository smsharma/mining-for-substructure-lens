from simulation.units import *
from simulation_new.profiles import MassProfileNFW


class SubhaloPopulation:
    def __init__(self, N_calib=150, beta=-1.9, m_min=1e9*M_s, r_roi=2.5,
                 M_hst=1e14*M_s, theta_s=1e-4, c_hst=6.):

        alpha = self.alpha_calib(1e8 * M_s, 1e10 * M_s, N_calib, M_MW, beta)
        self.n_sub_tot = self.n_sub(m_min, 0.01 * M_hst, M_hst, alpha, beta)

        f_sub = MassProfileNFW.M_cyl_div_M0(r_roi * asctorad / theta_s) \
            / MassProfileNFW.M_cyl_div_M0(c_hst * theta_s / theta_s)

        self.n_sub_roi = np.random.poisson(f_sub * self.n_sub_tot)
        self.m_sample = self.draw_m_sub(self.n_sub_roi, m_min, beta)

    def alpha_calib(self, m_min_calib, m_max_calib, n_calib, M_calib, beta, M_0=M_MW, m_0=1e9*M_s):
        return -M_0 * (m_max_calib * m_min_calib / m_0) ** -beta * n_calib * (-1 + -beta) / \
               (M_calib * (-m_max_calib ** -beta * m_min_calib + m_max_calib * m_min_calib ** -beta))

    def n_sub(self, m_min, m_max, M, alpha, beta, M_0=M_MW, m_0=1e9*M_s):
        return alpha * M * (m_max * m_min / m_0) ** --beta * \
               (m_max ** -beta * m_min - m_max * m_min ** -beta) / (M_0 * (-1 + -beta))

    def draw_m_sub(self, n_sub, m_sub_min, beta):
        u = np.random.uniform(0, 1, size=n_sub)
        m_sub = m_sub_min * (1 - u) ** (1.0 / (beta + 1.0))
        return m_sub
