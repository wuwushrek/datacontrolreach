from jax.config import config
config.update("jax_enable_x64", True)

import jax
import datacontrolreach.jumpy as jp
from datacontrolreach.interval import Interval
import mpmath
from mpmath import iv
from mpmath import matrix

import numpy as np
# parser = argparse.ArgumentParser('Learning Midpoint of Stiff Linear System')
# parser.add_argument('--seed',  type=int, default=201)
# args = parser.parse_args()

EPS_PRECISION = 1e-6
atol = 1e-10
rtol = 1e-10

def elemwise_mult(mA_iv, mB_iv):
	return iv.matrix( [ [ mA_iv[i,j] * mB_iv[i,j] for j in range(mA_iv.cols)] for i in range(mA_iv.rows) ])

def elemwise_div(mA_iv, mB_iv):
	return iv.matrix( [ [ mA_iv[i,j] / mB_iv[i,j] for j in range(mA_iv.cols)] for i in range(mA_iv.rows) ])

def elemwise_pow(mA_iv, powval):
	return iv.matrix( [ [ mA_iv[i,j] ** powval for j in range(mA_iv.cols)] for i in range(mA_iv.rows) ])

def elemwise_abs(mA_iv):
	return iv.matrix( [ [ abs(mA_iv[i,j]) for j in range(mA_iv.cols)] for i in range(mA_iv.rows) ])

def elemwise_sqrt(mA_iv):
	return iv.matrix( [ [ iv.sqrt(mA_iv[i,j]) for j in range(mA_iv.cols)] for i in range(mA_iv.rows) ])

def elemwise_cos(mA_iv):
	return iv.matrix( [ [ iv.cos(mA_iv[i,j]) for j in range(mA_iv.cols)] for i in range(mA_iv.rows) ])

def elemwise_sin(mA_iv):
	return iv.matrix( [ [ iv.sin(mA_iv[i,j]) for j in range(mA_iv.cols)] for i in range(mA_iv.rows) ])

def compare_intervals(mA, mA_iv):
	if isinstance(mA_iv, Interval):
		ma_iv_lb, ma_iv_ub = mA_iv.lb, mA_iv.ub
	else:
		ma_iv_lb, ma_iv_ub = iv2np(mA_iv)
	# print(ma_iv_lb.flatten() - mA.lb.flatten())
	return np.allclose(ma_iv_lb.flatten(), mA.lb.flatten(), rtol=rtol, atol=atol) and np.allclose(ma_iv_ub.flatten(), mA.ub.flatten(), rtol=rtol, atol=atol)

def np2iv(int_lb, int_ub):
	if int_lb.ndim == 1:
		mB = iv.matrix([iv.mpf([lb,ub]) for lb, ub in zip(int_lb, int_ub)])
	else:
		mB = iv.matrix([[iv.mpf([lb,ub]) for lb, ub in zip(l_lb, l_ub)] for l_lb, l_ub in zip(int_lb, int_ub)])
	return mB

def iv2np(mA):
	res_lb = np.zeros((mA.rows, mA.cols))
	res_ub = np.zeros((mA.rows, mA.cols))
	for i in range(mA.rows):
		for j in range(mA.cols):
			res_lb[i,j] = mA[i,j].a.a
			res_ub[i,j] = mA[i,j].b.a
	return res_lb, res_ub

def rd_interval(n_dim):
	int_lb = np.random.uniform(low=-10.0, high=10.0, size=n_dim)
	width = np.random.uniform(low=0.0, high=5.0, size=n_dim)
	int_ub = int_lb + width
	return int_lb, int_ub

def rd_pos_interval(n_dim):
	int_lb = np.random.uniform(low=0.0, high=10.0, size=n_dim)
	width = np.random.uniform(low=0.0, high=5.0, size=n_dim)
	int_ub = int_lb + width
	return int_lb, int_ub

def rd_nzero_interval(n_dim):
	int_lb = np.random.uniform(low=-10.0, high=10.0, size=n_dim)
	lb = np.zeros(n_dim)
	ub = np.maximum(-int_lb , 5.0) / 2
	width = np.random.uniform(low=lb, high=ub, size=n_dim)
	int_ub = int_lb + width
	return int_lb, int_ub

def test_add(seed):
	np.random.seed(seed)
	# Generate the data for test
	size_arr = np.random.randint(low=1, high=3)
	n_dim = tuple(np.random.randint(low=1, high=21, size=size_arr))
	int_lb_1, int_ub_1 = rd_interval(n_dim)
	int_lb_2, int_ub_2 = rd_interval(n_dim)
	# Build the interval matrices
	mA = Interval(lb=int_lb_1, ub= int_ub_1)
	mB = Interval(lb=int_lb_2, ub= int_ub_2)
	m_res = mA + mB
	m_res_1 = 1 + mA
	m_res_2 = mA + 2.0
	# Build the interval matrices
	mA_iv = np2iv(int_lb_1, int_ub_1)
	mB_iv = np2iv(int_lb_2, int_ub_2)
	m_res_iv = mA_iv + mB_iv
	m_res_1_iv = 1 + mA_iv
	m_res_2_iv = mA_iv + 2.0
	assert compare_intervals(m_res, m_res_iv), 'Addition between intervals fails : {} , {}\n'.format(m_res, m_res_iv)
	assert compare_intervals(m_res_1, m_res_1_iv), 'Addition between scalar and intervals fails : {} , {}\n'.format(m_res_1, m_res_1_iv)
	assert compare_intervals(m_res_2, m_res_2_iv), 'Addition between intervals and scalar fails : {} , {}\n'.format(m_res_2, m_res_2_iv)

def test_sub(seed):
	np.random.seed(seed)
	# Generate the data for test
	size_arr = np.random.randint(low=1, high=3)
	n_dim = tuple(np.random.randint(low=1, high=21, size=size_arr))
	int_lb_1, int_ub_1 = rd_interval(n_dim)
	int_lb_2, int_ub_2 = rd_interval(n_dim)
	# Build the interval matrices
	mA = Interval(lb=int_lb_1, ub= int_ub_1)
	mB = Interval(lb=int_lb_2, ub= int_ub_2)
	m_res = mA - mB
	m_res_1 = 1 - mA
	m_res_2 = mA - 2.0
	# Build the interval matrices
	mA_iv = np2iv(int_lb_1, int_ub_1)
	mB_iv = np2iv(int_lb_2, int_ub_2)
	m_res_iv = mA_iv - mB_iv
	m_res_1_iv = 1 - mA_iv
	m_res_2_iv = mA_iv - 2.0
	assert compare_intervals(m_res, m_res_iv), 'Substraction between intervals fails : {} , {}\n'.format(m_res, m_res_iv)
	assert compare_intervals(m_res_1, m_res_1_iv), 'Substraction between scalar and intervals fails : {} , {}\n'.format(m_res_1, m_res_1_iv)
	assert compare_intervals(m_res_2, m_res_2_iv), 'Substraction between intervals and scalar fails : {} , {}\n'.format(m_res_2, m_res_2_iv)

def test_mul(seed):
	np.random.seed(seed)
	# Generate the data for test
	size_arr = np.random.randint(low=1, high=3)
	n_dim = tuple(np.random.randint(low=1, high=21, size=size_arr))
	int_lb_1, int_ub_1 = rd_interval(n_dim)
	int_lb_2, int_ub_2 = rd_interval(n_dim)
	# Build the interval matrices
	mA = Interval(lb=int_lb_1, ub= int_ub_1)
	mB = Interval(lb=int_lb_2, ub= int_ub_2)
	m_res = mA * mB
	m_res_1 = 0.4 * mA
	m_res_2 = mA * -3.0
	# Build the interval matrices
	mA_iv = np2iv(int_lb_1, int_ub_1)
	mB_iv = np2iv(int_lb_2, int_ub_2)
	m_res_iv = elemwise_mult(mA_iv, mB_iv)
	m_res_1_iv = 0.4 * mA_iv
	m_res_2_iv = mA_iv * -3.0
	assert compare_intervals(m_res, m_res_iv), 'Multiplication between intervals fails : {} , {}\n'.format(m_res, m_res_iv)
	assert compare_intervals(m_res_1, m_res_1_iv), 'Multiplication between scalar and intervals fails : {} , {}\n'.format(m_res_1, m_res_1_iv)
	assert compare_intervals(m_res_2, m_res_2_iv), 'Multiplication between intervals and scalar fails : {} , {}\n'.format(m_res_2, m_res_2_iv)

def test_div(seed):
	np.random.seed(seed)
	# Generate the data for test
	size_arr = np.random.randint(low=1, high=3)
	n_dim = tuple(np.random.randint(low=1, high=21, size=size_arr))
	int_lb_1, int_ub_1 = rd_interval(n_dim)
	int_lb_2, int_ub_2 = rd_nzero_interval(n_dim)
	# Build the interval matrices
	mA = Interval(lb=int_lb_1, ub= int_ub_1) + 10.1
	mB = Interval(lb=int_lb_2, ub= int_ub_2) + 10.1
	m_res = mA / mB
	m_res_1 = 0.4 / mB
	m_res_2 = mA / -3.0
	# Build the interval matrices
	mA_iv = np2iv(int_lb_1, int_ub_1) + 10.1
	mB_iv = np2iv(int_lb_2, int_ub_2) + 10.1
	m_res_iv = elemwise_div(mA_iv, mB_iv)
	m_res_1_iv = elemwise_div(np2iv(np.full(n_dim,0.4), np.full(n_dim,0.4)), mB_iv)
	m_res_2_iv = mA_iv / -3.0
	assert compare_intervals(m_res, m_res_iv), 'Division between intervals fails : {} , {}\n'.format(m_res, m_res_iv)
	assert compare_intervals(m_res_1, m_res_1_iv), 'Division between scalar and intervals fails : {} , {}\n'.format(m_res_1, m_res_1_iv)
	assert compare_intervals(m_res_2, m_res_2_iv), 'Division between intervals and scalar fails : {} , {}\n'.format(m_res_2, m_res_2_iv)

def test_pow(seed):
	np.random.seed(seed)
	# Generate the data for test
	size_arr = np.random.randint(low=1, high=3)
	n_dim = tuple(np.random.randint(low=1, high=21, size=size_arr))
	int_lb_1, int_ub_1 = rd_interval(n_dim)
	rand_pow = np.random.randint(low=0, high=5)
	# Build the interval matrices
	mA = Interval(lb=int_lb_1, ub= int_ub_1)
	m_res = mA ** rand_pow
	# Build the interval matrices
	mA_iv = np2iv(int_lb_1, int_ub_1)
	m_res_iv = elemwise_pow(mA_iv,  rand_pow)
	assert compare_intervals(m_res, m_res_iv), 'Power between intervals and positive number fails : {} , {}\n'.format(m_res, m_res_iv)

def test_abs(seed):
	np.random.seed(seed)
	# Generate the data for test
	size_arr = np.random.randint(low=1, high=3)
	n_dim = tuple(np.random.randint(low=1, high=21, size=size_arr))
	int_lb_1, int_ub_1 = rd_interval(n_dim)
	# Build the interval matrices
	mA = Interval(lb=int_lb_1, ub= int_ub_1)
	m_res = abs(mA)
	# Build the interval matrices
	mA_iv = np2iv(int_lb_1, int_ub_1)
	m_res_iv = elemwise_abs(mA_iv)
	assert compare_intervals(m_res, m_res_iv), 'Absolute value of an interval number fails : {} , {}\n'.format(m_res, m_res_iv)

def test_width(seed):
	np.random.seed(seed)
	# Generate the data for test
	size_arr = np.random.randint(low=1, high=3)
	n_dim = tuple(np.random.randint(low=1, high=21, size=size_arr))
	int_lb_1, int_ub_1 = rd_interval(n_dim)
	# Build the interval matrices
	mA = Interval(lb=int_lb_1, ub= int_ub_1)
	m_res = mA.width
	# Build the interval matrices
	mA_iv = np2iv(int_lb_1, int_ub_1)
	m_res_iv = np.array( [ [ float(mA_iv[i,j].delta.a) for j in range(mA_iv.cols)] for i in range(mA_iv.rows) ])
	assert np.allclose(m_res.flatten(), m_res_iv.flatten(), rtol=rtol, atol=atol), 'Width of an interval fails : {} , {}\n'.format(m_res, m_res_iv)

def test_contains(seed):
	np.random.seed(seed)
	# Generate the data for test
	size_arr = np.random.randint(low=1, high=3)
	n_dim = tuple(np.random.randint(low=1, high=21, size=size_arr))
	int_lb_1, int_ub_1 = rd_interval(n_dim)
	int_lb_2, int_ub_2 = rd_interval(n_dim)

	# Build the interval matrices
	mA = Interval(lb=int_lb_1, ub= int_ub_1)
	mB = Interval(lb=int_lb_2, ub= int_ub_2)

	m_res = mA.contains(mB)

	# Build the interval matrices
	mA_iv = np2iv(int_lb_1, int_ub_1)
	mB_iv = np2iv(int_lb_2, int_ub_2)

	m_res_iv = np.array( [ [ mB_iv[i,j] in mA_iv[i,j] for j in range(mA_iv.cols)] for i in range(mA_iv.rows) ])
	assert np.allclose(m_res.flatten(), m_res_iv.flatten(), rtol=rtol, atol=atol), 'Test if an interval contains another interval fails : {} , {}\n'.format(m_res, m_res_iv)

def test_sqrt(seed):
	np.random.seed(seed)
	# Generate the data for test
	size_arr = np.random.randint(low=1, high=3)
	n_dim = tuple(np.random.randint(low=1, high=21, size=size_arr))
	int_lb_1, int_ub_1 = rd_pos_interval(n_dim)

	# Build the interval matrices
	mA = Interval(lb=int_lb_1, ub= int_ub_1)

	m_res = mA.sqrt()

	# Build the interval matrices
	mA_iv = np2iv(int_lb_1, int_ub_1)

	m_res_iv = elemwise_sqrt(mA_iv)
	assert compare_intervals(m_res, m_res_iv), 'Absolute value of an interval number fails : {} , {}\n'.format(m_res, m_res_iv)

def test_norm(seed):
	np.random.seed(seed)
	# Generate the data for test
	size_arr = np.random.randint(low=1, high=3)
	n_dim = tuple(np.random.randint(low=1, high=21, size=size_arr))
	int_lb_1, int_ub_1 = rd_pos_interval(n_dim)

	# Build the interval matrices
	mA = Interval(lb=int_lb_1, ub= int_ub_1)

	m_res = mA.norm()

	# Build the interval matrices
	mA_iv = np2iv(int_lb_1, int_ub_1)

	m_res_iv = iv.matrix([[iv.sqrt( sum([ mA_iv[i,j]**2 for j in range(mA_iv.cols) for i in range(mA_iv.rows)])) ]])
	assert compare_intervals(m_res, m_res_iv), 'Norm value of an interval fails : {} , {}\n'.format(m_res, m_res_iv)

def test_cos(seed):
	np.random.seed(seed)
	# Generate the data for test
	size_arr = np.random.randint(low=1, high=3)
	n_dim = tuple(np.random.randint(low=1, high=21, size=size_arr))
	int_lb_1, int_ub_1 = rd_interval(n_dim)

	# Build the interval matrices
	mA = Interval(lb=int_lb_1, ub= int_ub_1)
	# print(mA)

	m_res = mA.cos()

	# Build the interval matrices
	mA_iv = np2iv(int_lb_1, int_ub_1)

	m_res_iv = elemwise_cos(mA_iv)
	assert compare_intervals(m_res, m_res_iv), 'Cos value of an interval fails : {} , {}\n'.format(m_res, m_res_iv)

def test_sin(seed):
	np.random.seed(seed)
	# Generate the data for test
	size_arr = np.random.randint(low=1, high=3)
	n_dim = tuple(np.random.randint(low=1, high=21, size=size_arr))
	int_lb_1, int_ub_1 = rd_interval(n_dim)

	# Build the interval matrices
	mA = Interval(lb=int_lb_1, ub= int_ub_1)
	# print(mA)

	m_res = mA.sin()

	# Build the interval matrices
	mA_iv = np2iv(int_lb_1, int_ub_1)

	m_res_iv = elemwise_sin(mA_iv)
	assert compare_intervals(m_res, m_res_iv), 'Sin value of an interval fails : {} , {}\n'.format(m_res, m_res_iv)

def test_comparison(seed):
	np.random.seed(seed)
	# Generate the data for test
	size_arr = np.random.randint(low=1, high=3)
	n_dim = tuple(np.random.randint(low=1, high=21, size=size_arr))
	int_lb_1, int_ub_1 = rd_interval(n_dim)
	int_lb_2, int_ub_2 = rd_interval(n_dim)

	# Build the interval matrices
	mA = Interval(lb=int_lb_1, ub= int_ub_1)
	mB = Interval(lb=int_lb_2, ub= int_ub_2)
	res_leq = mA <= mB
	res_eq = mA == mB
	res_geq = mA >= mB

	# # Build the interval matrices
	mA_iv = np2iv(int_lb_1, int_ub_1)
	mB_iv = np2iv(int_lb_2, int_ub_2)
	res_leq_iv = np.array( [ [ mA_iv[i,j] <= mB_iv[i,j] if (mA_iv[i,j] <= mB_iv[i,j]) is not None else False  for j in range(mA_iv.cols)] for i in range(mA_iv.rows) ])
	res_eq_iv = np.array( [ [ mA_iv[i,j] == mB_iv[i,j] if (mA_iv[i,j] == mB_iv[i,j]) is not None else False  for j in range(mA_iv.cols)] for i in range(mA_iv.rows) ])
	res_geq_iv = np.array( [ [ mA_iv[i,j] >= mB_iv[i,j] if (mA_iv[i,j] >= mB_iv[i,j]) is not None else False  for j in range(mA_iv.cols)] for i in range(mA_iv.rows) ])

	assert np.allclose(res_leq.flatten(), res_leq_iv.flatten(), rtol=rtol, atol=atol), 'LEQ intervals fails : {} , {}\n'.format(res_leq, res_leq_iv)
	assert np.allclose(res_eq.flatten(), res_eq_iv.flatten(), rtol=rtol, atol=atol), 'EQ intervals fails : {} , {}\n'.format(res_eq, res_eq_iv)
	assert np.allclose(res_geq.flatten(), res_geq_iv.flatten(), rtol=rtol, atol=atol), 'LEQ intervals fails : {} , {}\n'.format(res_geq, res_geq_iv)

def test_abs(seed):
	np.random.seed(seed)
	# Generate the data for test
	size_arr = np.random.randint(low=1, high=3)
	n_dim = tuple(np.random.randint(low=1, high=21, size=size_arr))
	int_lb_1, int_ub_1 = rd_interval(n_dim)
	# Build the interval matrices
	mA = Interval(lb=int_lb_1, ub= int_ub_1)
	m_res = abs(mA)
	# Build the interval matrices
	mA_iv = np2iv(int_lb_1, int_ub_1)
	m_res_iv = elemwise_abs(mA_iv)
	assert compare_intervals(m_res, m_res_iv), 'Absolute value of an interval number fails : {} , {}\n'.format(m_res, m_res_iv)

def test_matrixmul(seed):
	np.random.seed(seed)
	# Generate the data for test
	n_dim1 = tuple(np.random.randint(low=1, high=21, size=2))
	n_dim2 = tuple(np.random.randint(low=1, high=21, size=2))
	n_dim2 = (n_dim1[1], n_dim2[1])
	int_lb_1, int_ub_1 = rd_interval(n_dim1)
	int_lb_2, int_ub_2 = rd_interval(n_dim2)
	int_lb_3, int_ub_3 = rd_interval((n_dim1[1],))
	int_lb_4, int_ub_4 = rd_interval((n_dim1[0],))
	int_lb_5, int_ub_5 = rd_interval((n_dim1[0],n_dim1[1], n_dim2[1]))


	# Build the interval matrices
	mA = Interval(lb=int_lb_1, ub= int_ub_1)
	mB = Interval(lb=int_lb_2, ub= int_ub_2)
	mC = Interval(lb=int_lb_3, ub= int_ub_3)
	mD = Interval(lb=int_lb_4, ub= int_ub_4)
	mE = Interval(lb=int_lb_5, ub= int_ub_5)

	res_1 = mA @ mB
	res_2 = mA @ mC
	res_3 = mD @ mA

	res_4 = mE @ mC


	mA_iv = np2iv(int_lb_1, int_ub_1)
	mB_iv = np2iv(int_lb_2, int_ub_2)
	mC_iv = np2iv(int_lb_3.reshape(-1,1), int_ub_3.reshape(-1,1))
	mD_iv = np2iv(int_lb_4.reshape(1,-1), int_ub_4.reshape(1, -1))
	# mE_iv = np2iv(int_lb_5, int_ub_5)

	res_1_iv = mA_iv * mB_iv
	res_2_iv = mA_iv * mC_iv
	res_3_iv = mD_iv * mA_iv
	res_4_iv = np2iv(*mul_iTv(int_lb_5, int_ub_5, int_lb_3, int_ub_3))

	assert compare_intervals(res_1, res_1_iv), 'Mat interval product Mat interval fails : {} , {}\n'.format(res_1, res_1_iv)
	assert compare_intervals(res_2, res_2_iv), 'Mat interval product vector interval fails : {} , {}\n'.format(res_2, res_2_iv)
	assert compare_intervals(res_3, res_3_iv), 'vect interval product Mat interval fails : {} , {}\n'.format(res_3, res_3_iv)
	assert compare_intervals(res_4, res_4_iv), 'Tensor interval product Mat interval fails : {} , {}\n'.format(res_4, res_4_iv)

def test_jit(seed):
	np.random.seed(seed)
	# Generate the data for test
	size_arr = np.random.randint(low=1, high=3)
	n_dim = tuple(np.random.randint(low=1, high=21, size=size_arr))
	int_lb_1, int_ub_1 = rd_interval(n_dim)
	int_lb_2, int_ub_2 = rd_interval(n_dim)
	# Build the interval matrices
	mA = Interval(lb=int_lb_1, ub= int_ub_1)
	mB = Interval(lb=int_lb_2, ub= int_ub_2)

	f1 = lambda x, y : x * y
	f1jit = jax.jit(f1)

	# Jit of a function
	res = f1(mA[0], mB)
	resjit = f1jit(mA[0], mB)

	# Jax jvp
	fval, fjvp = jax.jit(lambda y : jax.jvp(lambda x :  x[0]*mB, (y,), (y, )))(mA)
	assert compare_intervals(res, resjit), 'Jit interval result fails : {} , {}\n'.format(res, resjit)
	assert compare_intervals(res, fval), 'Jvp fval result fails : {} , {}\n'.format(res, fval)
	assert compare_intervals(res, fjvp), 'Jvp fval result fails : {} , {}\n'.format(res, fjvp)


def mul_i(x_lb, x_ub, y_lb, y_ub):
	"""Define the multiplication between an interval x=(x_lb,x_ub) and an
	   interval given by y = (y_lb, y_ub)
	"""
	val_1 = x_lb * y_lb
	val_2 = x_lb * y_ub
	val_3 = x_ub * y_lb
	val_4 = x_ub * y_ub
	return np.minimum(val_1, np.minimum(val_2, np.minimum(val_3,val_4))), np.maximum(val_1, np.maximum(val_2, np.maximum(val_3,val_4)))



def mul_iTv(x_lb, x_ub, y_lb, y_ub):
	""" Define the multiplication between an interval Tensor x=(x_lb,x_ub) and an
		interval vector y = (y_lb, y_ub)
	"""
	res_lb = np.empty((x_lb.shape[0],x_lb.shape[2]))
	res_ub = np.empty((x_lb.shape[0],x_lb.shape[2]))
	for i in range(x_lb.shape[0]):
		for k in range(x_lb.shape[2]):
			res_i_lb, res_i_ub = 0 , 0
			for l in range(x_lb.shape[1]):
				t_lb, t_ub = mul_i(x_lb[i,l,k], x_ub[i,l,k], y_lb[l], y_ub[l])
				res_i_lb += t_lb
				res_i_ub += t_ub
			res_lb[i,k] = res_i_lb
			res_ub[i,k] = res_i_ub
	return res_lb, res_ub


# Jacobian tests
def test_jacobian_add(seed):
	np.random.seed(seed)
	# Generate the data for test
	size_arr = np.random.randint(low=1, high=3)
	n_dim = tuple(np.random.randint(low=1, high=21, size=size_arr))
	int_lb_1, int_ub_1 = rd_interval(n_dim)
	int_lb_2, int_ub_2 = rd_interval(n_dim)
	# Build the intervals
	interval1 = Interval(lb=int_lb_1, ub= int_ub_1)
	interval2 = Interval(lb=int_lb_2, ub= int_ub_2)

	tangent0 = Interval( np.full(np.shape(interval1), 0.0), np.full(np.shape(interval1), 0.0))  # is an all 0's matrix
	tangent1 = Interval( np.full(np.shape(interval1), 1.0), np.full(np.shape(interval1), 1.0))  # is an all 1's matrix

	# test interval + interval
	value, derivative = jax.jvp(Interval.__add__, (interval1, interval2), (tangent0, tangent1))
	assert compare_intervals(tangent1, derivative), 'Jacobian for Addition between intervals fails : {} , {}\n'.format(interval1, derivative)

	value, derivative = jax.jvp(Interval.__add__, (interval1, interval2), (tangent1, tangent0))
	assert compare_intervals(tangent1, derivative), 'Jacobian for Addition between intervals fails : {} , {}\n'.format(interval1, derivative)

	# test interval + number
	value, derivative = jax.jvp(Interval.__add__, (interval1, 2.0), (tangent0, 1.0))
	assert compare_intervals(tangent1, derivative), 'Jacobian for Addition between interval and scalar fails : {} , {}\n'.format(tangent1, derivative)

	value, derivative = jax.jvp(Interval.__add__, (interval1, 2.0), (tangent1, 0.0))
	assert compare_intervals(tangent1, derivative), 'Jacobian for Addition between interval and scalar fails : {} , {}\n'.format(tangent1, derivative)

	# test number + interval
	value, derivative = jax.jvp(Interval.__add__, (1.0, interval2), (0.0, tangent1))
	assert compare_intervals(tangent1, derivative), 'Jacobian for Addition between scalar and interval fails : {} , {}\n'.format(tangent1, derivative)

	value, derivative = jax.jvp(Interval.__add__, (1.0, interval2), (1.0, tangent0))
	assert compare_intervals(tangent1, derivative), 'Jacobian for Addition between scalar and interval fails : {} , {}\n'.format(tangent1, derivative)

def test_jacobian_sub(seed):
	np.random.seed(seed)
	# Generate the data for test
	size_arr = np.random.randint(low=1, high=3)
	n_dim = tuple(np.random.randint(low=1, high=21, size=size_arr))
	int_lb_1, int_ub_1 = rd_interval(n_dim)
	int_lb_2, int_ub_2 = rd_interval(n_dim)
	# Build the intervals
	interval1 = Interval(lb=int_lb_1, ub= int_ub_1)
	interval2 = Interval(lb=int_lb_2, ub= int_ub_2)

	tangent0 = Interval( np.full(np.shape(interval1), 0.0), np.full(np.shape(interval1), 0.0))  # is an all 0's matrix
	tangent1 = Interval( np.full(np.shape(interval1), 1.0), np.full(np.shape(interval1), 1.0))  # is an all 1's matrix

	# test interval + interval
	value, derivative = jax.jvp(Interval.__sub__, (interval1, interval2), (tangent0, tangent1))
	assert compare_intervals(-tangent1, derivative), 'Jacobian for Subtraction between intervals fails : {} , {}\n'.format(interval1, derivative)

	value, derivative = jax.jvp(Interval.__sub__, (interval1, interval2), (tangent1, tangent0))
	assert compare_intervals(tangent1, derivative), 'Jacobian for Subtraction between intervals fails : {} , {}\n'.format(interval1, derivative)

	# test interval + number
	value, derivative = jax.jvp(Interval.__sub__, (interval1, 2.0), (tangent0, 1.0))
	assert compare_intervals(-tangent1, derivative), 'Jacobian for Subtraction between interval and scalar fails : {} , {}\n'.format(tangent1, derivative)

	value, derivative = jax.jvp(Interval.__sub__, (interval1, 2.0), (tangent1, 0.0))
	assert compare_intervals(tangent1, derivative), 'Jacobian for Subtraction between interval and scalar fails : {} , {}\n'.format(tangent1, derivative)

	# test number + interval
	value, derivative = jax.jvp(Interval.__sub__, (1.0, interval2), (0.0, tangent1))
	assert compare_intervals(-tangent1, derivative), 'Jacobian for Subtraction between scalar and interval fails : {} , {}\n'.format(tangent1, derivative)

	value, derivative = jax.jvp(Interval.__sub__, (1.0, interval2), (1.0, tangent0))
	assert compare_intervals(tangent1, derivative), 'Jacobian for Subtraction between scalar and interval fails : {} , {}\n'.format(tangent1, derivative)


def test_jacobian_mul(seed):
	np.random.seed(seed)
	# Generate the data for test
	size_arr = np.random.randint(low=1, high=3)
	n_dim = tuple(np.random.randint(low=1, high=21, size=size_arr))
	int_lb_1, int_ub_1 = rd_interval(n_dim)
	int_lb_2, int_ub_2 = rd_interval(n_dim)
	# Build the intervals
	interval1 = Interval(lb=int_lb_1, ub= int_ub_1)
	interval2 = Interval(lb=int_lb_2, ub= int_ub_2)

	tangent0 = Interval( np.full(np.shape(interval1), 0.0), np.full(np.shape(interval1), 0.0))  # is an all 0's matrix
	tangent1 = Interval( np.full(np.shape(interval1), 1.0), np.full(np.shape(interval1), 1.0))  # is an all 1's matrix

	# test interval + interval
	value, derivative = jax.jvp(Interval.__mul__, (interval1, interval2), (tangent0, tangent1))
	assert compare_intervals(interval1, derivative), 'Jacobian for Multiplication between intervals fails : {} , {}\n'.format(interval1, derivative)

	value, derivative = jax.jvp(Interval.__mul__, (interval1, interval2), (tangent1, tangent0))
	assert compare_intervals(interval2, derivative), 'Jacobian for Multiplication between intervals fails : {} , {}\n'.format(interval2, derivative)

	# test interval + number
	value, derivative = jax.jvp(Interval.__mul__, (interval1, 2.0), (tangent0, 1.0))
	assert compare_intervals(interval1, derivative), 'Jacobian for Multiplication between interval and scalar fails : {} , {}\n'.format(interval1, derivative)

	value, derivative = jax.jvp(Interval.__mul__, (interval1, 2.0), (tangent1, 0.0))
	assert compare_intervals(2*tangent1, derivative), 'Jacobian for Multiplication between interval and scalar fails : {} , {}\n'.format(interval2, derivative)

	# test number + interval
	value, derivative = jax.jvp(Interval.__mul__, (1.0, interval2), (0.0, tangent1))
	assert compare_intervals(tangent1, derivative), 'Jacobian for Multiplication between scalar and interval fails : {} , {}\n'.format(tangent1, derivative)

	value, derivative = jax.jvp(Interval.__mul__, (1.0, interval2), (1.0, tangent0))
	assert compare_intervals(interval2, derivative), 'Jacobian for Multiplication between scalar and interval fails : {} , {}\n'.format(interval2, derivative)


def test_jacobian_div(seed):
	np.random.seed(seed)
	# Generate the data for test
	size_arr = np.random.randint(low=1, high=3)
	n_dim = tuple(np.random.randint(low=1, high=21, size=size_arr))
	int_lb_1, int_ub_1 = rd_interval(n_dim)
	int_lb_2, int_ub_2 = rd_interval(n_dim)
	# Build the intervals
	interval1 = Interval(lb=int_lb_1, ub= int_ub_1) + 10.1
	interval2 = Interval(lb=int_lb_2, ub= int_ub_2) + 10.1

	tangent0 = Interval( np.full(np.shape(interval1), 0.0), np.full(np.shape(interval1), 0.0))  # is an all 0's matrix
	tangent1 = Interval( np.full(np.shape(interval1), 1.0), np.full(np.shape(interval1), 1.0))  # is an all 1's matrix

	# test interval + interval
	value, derivative = jax.jvp(Interval.__div__, (interval1, interval2), (tangent0, tangent1))
	assert compare_intervals(-interval1 /(interval2 * interval2), derivative), 'Jacobian for division between intervals fails : {} , {}\n'.format(-interval1 /(interval2 * interval2), derivative)

	value, derivative = jax.jvp(Interval.__div__, (interval1, interval2), (tangent1, tangent0))
	assert compare_intervals(1/interval2, derivative), 'Jacobian for division between intervals fails : {} , {}\n'.format(interval2, derivative)

	# test interval + number
	value, derivative = jax.jvp(Interval.__div__, (interval1, 2.0), (tangent0, 1.0))
	assert compare_intervals(-interval1 /(4.0  * tangent1), derivative), 'Jacobian for division between interval and scalar fails : {} , {}\n'.format(-interval1 / (4.0 * tangent1), derivative)

	value, derivative = jax.jvp(Interval.__div__, (interval1, 2.0), (tangent1, 0.0))
	assert compare_intervals(1/(2*tangent1), derivative), 'Jacobian for division between interval and scalar fails : {} , {}\n'.format(interval2, derivative)

	# test number + interval
	value, derivative = jax.jvp(Interval.__div__, (1.0, interval2), (0.0, tangent1))
	assert compare_intervals(-tangent1 /(interval2 * interval2), derivative), 'Jacobian for division between scalar and interval fails : {} , {}\n'.format(tangent1, derivative)

	value, derivative = jax.jvp(Interval.__div__, (1.0, interval2), (1.0, tangent0))
	assert compare_intervals(1/interval2, derivative), 'Jacobian for division between scalar and interval fails : {} , {}\n'.format(interval2, derivative)


def test_jacobian_pow(seed):
	np.random.seed(seed)
	# Generate the data for test
	size_arr = np.random.randint(low=1, high=3)
	n_dim = tuple(np.random.randint(low=1, high=21, size=size_arr))
	int_lb_1, int_ub_1 = rd_interval(n_dim)
	int_lb_2, int_ub_2 = rd_interval(n_dim)
	# Build the intervals
	interval1 = Interval(lb=int_lb_1, ub= int_ub_1) + 10.1

	tangent0 = Interval( np.full(np.shape(interval1), 0.0), np.full(np.shape(interval1), 0.0))  # is an all 0's matrix
	tangent1 = Interval( np.full(np.shape(interval1), 1.0), np.full(np.shape(interval1), 1.0))  # is an all 1's matrix

	# test interval + number
	value, derivative = jax.jvp(Interval.__pow__, (interval1, 2.0), (tangent0, 1.0))
	assert compare_intervals( interval1**2.0 * Interval.log(interval1), derivative), '1. Jacobian for power of interval to scalar fails : {} , {}\n'.format( interval1**2.0 * Interval.log(interval1), derivative)

	value, derivative = jax.jvp(Interval.__pow__, (interval1, 2.0), (tangent1, 0.0))
	assert compare_intervals(2.0 * interval1 ** (2.0-1), derivative), '2. Jacobian for power of interval to scalar fails : {} , {}\n'.format(2.0 * interval1 ** (2.0-1), derivative)

	value, derivative = jax.jvp(Interval.__pow__, (interval1, 1.5), (tangent0, 1.0))
	assert compare_intervals( interval1**1.5 * Interval.log(interval1), derivative), '3. Jacobian for power of interval to scalar fails : {} , {}\n'.format( interval1**1.5 * Interval.log(interval1), derivative)

	value, derivative = jax.jvp(Interval.__pow__, (interval1, 1.5), (tangent1, 0.0))
	assert compare_intervals(1.5 * interval1 ** (1.5-1), derivative), '4. Jacobian for power of interval to scalar fails : {} , {}\n'.format(1.5 * interval1 ** (1.5-1), derivative)


def test_jacobian_cos(seed):
	np.random.seed(seed)
	# Generate the data for test
	size_arr = np.random.randint(low=1, high=3)
	n_dim = tuple(np.random.randint(low=1, high=21, size=size_arr))
	int_lb_1, int_ub_1 = rd_interval(n_dim)
	# Build the intervals
	interval1 = Interval(lb=int_lb_1, ub= int_ub_1)

	tangent1 = Interval( np.full(np.shape(interval1), 1.0), np.full(np.shape(interval1), 1.0))  # is an all 1's matrix

	# test sin
	value, derivative = jax.jvp(Interval.cos, (interval1,), (tangent1,))
	assert compare_intervals( -Interval.sin(interval1), derivative), 'Jacobian for cos of interval fails: {} , {}\n'.format( -Interval.sin(interval1), derivative)

	# test cos
	value, derivative = jax.jvp(Interval.sin, (interval1,), (tangent1,))
	assert compare_intervals( Interval.cos(interval1), derivative), 'Jacobian for cos of interval fails: {} , {}\n'.format( Interval.cos(interval1), derivative)

	# test tan = causes divide by 0 error
	# value, derivative = jax.jvp(Interval.tan, (interval1,), (tangent1,))
	# assert compare_intervals( 1.0/Interval.cos(interval1) ** 2, derivative), 'Jacobian for cos of interval fails: {} , {}\n'.format( 1.0/Interval.cos(interval1) ** 2, derivative)

	# test abs
	value, derivative = jax.jvp(Interval.__abs__, (interval1,), (tangent1,))
	solution = jp.where(interval1 > 0, tangent1, jp.where(interval1 < 0, -tangent1, tangent1 * Interval(-1., 1.)))
	assert compare_intervals(solution, derivative), 'Jacobian for cos of interval fails: {} , {}\n'.format(solution, derivative)
