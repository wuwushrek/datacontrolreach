# Use float64 precision
from jax.config import config
config.update("jax_enable_x64", True)

import numpy as np
from datacontrolreach import jumpy as jp

default_aircraft = {
    "s": 300,
    "b": 30,
    "cbar": 11.32,
    "rm": 1.57e-3,
    "xcgref": 0.35,
    "xcg": 0.35,
    "he": 160.0,
    "c1": -.770,
    "c2": 0.02755,
    "c3": 1.055e-4,
    "c4": 1.642e-6,
    "c5": .9604,
    "c6": 1.759e-2,
    "c7": 1.792e-5,
    "c8": -.7336,
    "c9": 1.587e-5,
    "rtod": 57.29578,
    "g": 32.17,
    # center of gravity multiplier
    'xcg_mult': 1.0,
    # other aerodynmic coefficient multipliers
    'cxt_mult': 1.0,
    'cyt_mult': 1.0,
    'czt_mult': 1.0,
    'clt_mult': 1.0,
    'cmt_mult': 1.0,
    'cnt_mult': 1.0,
    'model': 'morelli',
}


# thtlc, el, ail, rdr = u[0], u[1], u[2], u[3]
# vt, alpha, beta, phi, theta, p, q, r = x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7]
# alt, power = x[8], x[9]
# intNz, intPs = x[10], x[11]

def known_terms(x, uval, t=None):
    """ Define the known terms of the F-16 dynamics according to the general
        rigid-body equations of motion of aircraft
        :param x : The state of the system
        :param u : The control signal to apply
        :param t : Time instant at which the vector field is evaluated
    """
    # Some parameters
    s, b, cbar, rm, xcgref, xcg, he, c1, c2, c3, c4, c5, c6, c7, c8, c9, rtod, g =\
            (default_aircraft[p] for p in 's b cbar rm xcgref xcg he c1 c2 c3 c4 c5 c6 c7 c8 c9 rtod g'.split())
    xcg_mult, cxt_mult, cyt_mult, czt_mult, clt_mult, cmt_mult, cnt_mult =\
            (default_aircraft[p] for p in 'xcg_mult cxt_mult cyt_mult czt_mult clt_mult cmt_mult cnt_mult'.split())

    # Get some of the states
    vt, phi, theta, p, q, r, alt = x[0],  x[3], x[4], x[5], x[6], x[7], x[8]
    power = x[9]

    # Pressure from speed and altitude
    qbar = _qbar(vt, alt)

    tvt = .5 / vt
    b2v = b * tvt

    # Save these coefficients -> Use in the moment computations
    cq = cbar * q * tvt
    cr = b2v * r
    cp = b2v * p

    # Axial speed and cos, sin of euler angles
    cbta = jp.cos(x[2])
    calp = jp.cos(x[1])
    salp = jp.sin(x[1])
    u = vt * calp * cbta
    v = vt * jp.sin(x[2])
    w = vt * salp * cbta
    sth = jp.sin(theta)
    cth = jp.cos(theta)
    sph = jp.sin(phi)
    cph = jp.cos(phi)

    gcth = g * cth
    qsph = q * sph

    # Axial velocity known part
    udot = r * v-q * w-g * sth
    vdot = p * w-r * u + gcth * sph
    wdot = q * u-p * v + gcth * cph

    # kinematics
    phi_dot = p + (sth/cth) * (qsph + r * cph)
    theta_dot = q * cph-r * sph

    # moments
    p_dot = (c2 * p + c1 * r + c4 * he) * q
    q_dot = (c5 * p-c7 * he) * r + c6 * (r * r-p * p)
    r_dot = (c8 * p-c2 * r + c9 * he) * q

    # navigation
    s5 = sph * cth
    s8 = cph * cth
    alt_dot = u * sth-v * s5-w * s8      # vertical speed

    Ps = p * calp+ r * salp

    return {'u' : u, 'v' : v, 'w' : w, 'udot' : udot, 'vdot' : vdot, 'wdot' : wdot, 'phi_dot' : phi_dot, 
            'theta_dot' : theta_dot, 'p_dot' : p_dot, 'q_dot' : q_dot, 
            'r_dot' : r_dot, 'alt_dot' : alt_dot, 'control' : uval,
            'qbar' : qbar, 'cbta' : cbta, 'cq' : cq, 'cr' : cr, 'cp' : cp, 'Ps' : Ps, 'power' : power, 'vt' : vt}


def composed_dynamics_full(kTerms, unkTerms):
    """ Composed the dynamics with the kTerms and an estimation of the unknown terms
    """
    # Get some known terms
    u, v, w, _udot, _vdot, _wdot, phi_dot, theta_dot, _p_dot, _q_dot, _r_dot, alt_dot, qbar, control, Ps, cbta, cq, cr, cp, power, vt= \
        [kTerms[v] for v in ('u','v','w','udot','vdot','wdot','phi_dot', 'theta_dot', 'p_dot', 'q_dot', 'r_dot',
                                'alt_dot', 'qbar', 'control', 'Ps', 'cbta', 'cq', 'cr', 'cp', 'power', 'vt', ) ]

    # Some constants
    s, b, rm, g, c3, c4, c7, c9, cbar, xcgref, xcg = (default_aircraft[p] for p in ('s','b','rm','g','c3','c4','c7','c9','cbar', 'xcgref', 'xcg'))

    # Some temporary variables
    qs = qbar * s
    qsb = qs * b
    rmqs = rm * qs

    # The states of the system
    thtl, el, ail, rdr = control[0], control[1], control[2], control[3]
    # vt, alpha, beta, phi, theta, p, q, r = x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7]
    # alt, power = x[8], x[9]
    # intNz, intPs = x[10], x[11]

    # Thrust unknown coefficients and thrust value
    _thrust0, _thrust1 = (unkTerms[v] for v in ('_thrust0', '_thrust1')) 
    thrust = _thrust0 + _thrust1 * power * 0.02

    # Power dot value
    cpow = 64.94 * thtl
    diff_power = cpow - power
    power_dot = _rtau(diff_power) * diff_power

    # Gather estimates for the forces
    _cxt_0, _cxt_de, _cxt_de2, _cxt_q = (unkTerms[v] for v in ('cxt_0', 'cxt_de', 'cxt_de2', 'cxt_q'))
    cxt = _cxt_0 + _cxt_de * el + _cxt_de2 * el**2 + _cxt_q * cq

    _cyt_0, _cyt_da, _cyt_dr, _cyt_p, _cyt_r = (unkTerms[v] for v in ('cyt_0', 'cyt_da', 'cyt_dr', 'cyt_p', 'cyt_r'))
    cyt = _cyt_0 + _cyt_da * ail + _cyt_dr * rdr + _cyt_p * cp + _cyt_r * cr

    _czt_0, _czt_de, _czt_q = (unkTerms[v] for v in ('czt_0', 'czt_de', 'czt_q'))
    czt = _czt_0 + _czt_de * el + _czt_q * cq

    # Gather estimates for the moment
    _clt_0, _clt_da, _clt_dr, _clt_p, _clt_r = (unkTerms[v] for v in ('clt_0', 'clt_da', 'clt_dr', 'clt_p', 'clt_r'))
    clt = _clt_0 + _clt_da * ail + _clt_dr * rdr + _clt_p * cp + _clt_r * cr

    _cmt_0, _cmt_de, _cmt_de2, _cmt_de3, _cmt_q = (unkTerms[v] for v in ('cmt_0', 'cmt_de', 'cmt_de2', 'cmt_de3', 'cmt_q'))
    cmt = _cmt_0 + _cmt_de * el + _cmt_de2 * el**2 + _cmt_de3 * el**3 + _cmt_q * cq + 2 * czt * (xcgref-xcg)

    _cnt_0, _cnt_da, _cnt_dr, _cnt_p, _cnt_r = (unkTerms[v] for v in ('cnt_0', 'cnt_da', 'cnt_dr', 'cnt_p', 'cnt_r'))
    cnt = _cnt_0 + _cnt_da * ail + _cnt_dr * rdr + _cnt_p * cp + _cnt_r * cr - 2 * cyt * (xcgref-xcg) * (cbar/b)
    # print(cxt , cyt, czt, clt, cmt, cnt)

    # Get acceleration and moments
    ax, ay, az = rmqs * cxt, rmqs * cyt, rmqs * czt
    _udot_unk = ax + rm * thrust
    _vdot_unk = ay
    _wdot_unk = az

    udot = _udot + _udot_unk
    vdot = _vdot + _vdot_unk
    wdot = _wdot + _wdot_unk
    dum = (u * u + w * w)

    vt_dot = (u * udot + v * vdot + w * wdot)/vt
    alpha_dot = (u * wdot-w * udot)/dum
    beta_dot = (vt * vdot-v * vt_dot) * cbta/dum

    _p_dot_unk = qsb * (c3 * clt + c4 * cnt)
    _q_dot_unk = qs * cbar * c7 * cmt
    _r_dot_unk = qsb * (c4 * clt + c9 * cnt)

    p_dot = _p_dot + _p_dot_unk
    q_dot = _q_dot + _q_dot_unk
    r_dot = _r_dot + _r_dot_unk

    xa = 15.0                   # sets distance normal accel is in front of the c.g. (xa = 15.0 at pilot)
    az = az - xa * q_dot        # moves normal accel in front of c.g.
    Nz = (-az / g) - 1.0        # zeroed at 1 g, positive g = pulling up
    # Ps = p * jp.cos(alpha) + r * jp.sin(alpha)

    return jp.array([vt_dot, alpha_dot, beta_dot, phi_dot, theta_dot, p_dot, q_dot, r_dot, alt_dot, power_dot, Nz, Ps])

############ Utility functions from the F-16 simulator ###############
def _qbar(vt, alt):
    '''
    Computes the pressure.

    Non-linearities: alt * vt^2
    '''
    ro = 2.377e-3
    # rho = freestream mass density
    rho = ro * _tfac(alt) ** 4.14
    # qbar = dynamic pressure
    return .5 * rho * vt * vt

def _tfac(alt):
    '''
    Non-linearities
    '''
    return 1 - .703e-5 * alt

def _rtau(dp):
    '''
    Non-linearities: discontinuous w.r.t. dp and saturation at both ends
    '''
    # Should be careful about it in particular the cases where part is below 25 and not
    one_dp = jp.ones_like(dp)
    return jp.where(dp <= 25, one_dp, jp.where(dp >= 50, .1 * one_dp, 1.9 - .036 * dp))

