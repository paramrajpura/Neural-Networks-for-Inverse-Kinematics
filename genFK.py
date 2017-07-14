import numpy as np
import random
from sympy import *
import transformations as trans
import csv

data = []
for i in range(1000):
	# Define DH param symbols
	q1, q2, q3, q4, q5, q6, q7 = symbols('q1:8')
	d1, d2, d3, d4, d5, d6, d7 = symbols('d1:8')
	a0, a1, a2, a3, a4, a5, a6 = symbols('a0:7')
	alpha0, alpha1, alpha2, alpha3, alpha4, alpha5, alpha6 = symbols('alpha0:7')

	deg2rad = np.pi/180.0
	rad2deg = 180.0/np.pi
	a1 = random.uniform(-185*deg2rad, 185*deg2rad)
	a2 = random.uniform(-45*deg2rad, 85*deg2rad)
	a3 = random.uniform(-150*deg2rad, 150*deg2rad)#-210*deg2rad, 65*deg2rad
	a4 = random.uniform(-350*deg2rad, 350*deg2rad)
	a5 = random.uniform(-125*deg2rad, 125*deg2rad)
	a6 = random.uniform(-350*deg2rad, 350*deg2rad)

	#q1 = 0
	#q2 = 0
	#q3 = 0
	#q4 = 0
	#q5 = 0
	#q6 = 0

	#print('Angles_Y',q1*rad2deg,q2*rad2deg,q3*rad2deg,q4*rad2deg,q5*rad2deg,q6*rad2deg)
	# Joint angle symbols
	# DH Parameters
	s = {alpha0: 0,      a0:   0,    d1: 0.75, q1:a1,
		 alpha1: -np.pi/2,  a1: 0.35,   d2: 0,     q2: a2-pi/2,
		 alpha2: 0,      a2: 1.25,   d3: 0, q3:a3,
		 alpha3: -np.pi/2,  a3: -0.054, d4: 1.5, q4:a4,
		 alpha4: np.pi/2,   a4: 0   ,   d5: 0, q5:a5,
		 alpha5: -np.pi/2,  a5: 0   ,   d6: 0, q6:a6,
		 alpha6: 0,      a6: 0   ,   d7: 0.303, q7: 0}


	# Modified DH params



	# Define Modified DH Transformation matrix



	# Create individual transformation matrices
	T0_1 = Matrix([[             cos(q1),            -sin(q1),            0,              a0],
		           [ sin(q1)*cos(alpha0), cos(q1)*cos(alpha0), -sin(alpha0), -sin(alpha0)*d1],
		           [ sin(q1)*sin(alpha0), cos(q1)*sin(alpha0),  cos(alpha0),  cos(alpha0)*d1],
		           [                   0,                   0,            0,               1]])
	T0_1 = T0_1.subs(s)

	T1_2 = Matrix([[             cos(q2),            -sin(q2),            0,              a1],
		           [ sin(q2)*cos(alpha1), cos(q2)*cos(alpha1), -sin(alpha1), -sin(alpha1)*d2],
		           [ sin(q2)*sin(alpha1), cos(q2)*sin(alpha1),  cos(alpha1),  cos(alpha1)*d2],
		           [                   0,                   0,            0,               1]])
	T1_2 = T1_2.subs(s)

	T2_3 = Matrix([[             cos(q3),            -sin(q3),            0,              a2],
		           [ sin(q3)*cos(alpha2), cos(q3)*cos(alpha2), -sin(alpha2), -sin(alpha2)*d3],
		           [ sin(q3)*sin(alpha2), cos(q3)*sin(alpha2),  cos(alpha2),  cos(alpha2)*d3],
		           [                   0,                   0,            0,               1]])
	T2_3 = T2_3.subs(s)

	T3_4 = Matrix([[             cos(q4),            -sin(q4),            0,              a3],
		           [ sin(q4)*cos(alpha3), cos(q4)*cos(alpha3), -sin(alpha3), -sin(alpha3)*d4],
		           [ sin(q4)*sin(alpha3), cos(q4)*sin(alpha3),  cos(alpha3),  cos(alpha3)*d4],
		           [                   0,                   0,            0,               1]])
	T3_4 = T3_4.subs(s)


	T4_5 = Matrix([[             cos(q5),            -sin(q5),            0,              a4],
		           [ sin(q5)*cos(alpha4), cos(q5)*cos(alpha4), -sin(alpha4), -sin(alpha4)*d5],
		           [ sin(q5)*sin(alpha4), cos(q5)*sin(alpha4),  cos(alpha4),  cos(alpha4)*d5],
		           [                   0,                   0,            0,               1]])
	T4_5 = T4_5.subs(s)


	T5_6 = Matrix([[             cos(q6),            -sin(q6),            0,              a5],
		           [ sin(q6)*cos(alpha5), cos(q6)*cos(alpha5), -sin(alpha5), -sin(alpha5)*d6],
		           [ sin(q6)*sin(alpha5), cos(q6)*sin(alpha5),  cos(alpha5),  cos(alpha5)*d6],
		           [                   0,                   0,            0,               1]])
	T5_6 = T5_6.subs(s)

	T6_G = Matrix([[             cos(q7),            -sin(q7),            0,              a6],
		           [ sin(q7)*cos(alpha6), cos(q7)*cos(alpha6), -sin(alpha6), -sin(alpha6)*d7],
		           [ sin(q7)*sin(alpha6), cos(q7)*sin(alpha6),  cos(alpha6),  cos(alpha6)*d7],
		           [                   0,                   0,            0,               1]])
	T6_G = T6_G.subs(s)

	R_x = Matrix([[1, 0, 0],
	[0, cos(pi/2), -sin(pi/2)],
	[0, sin(pi/2), cos(pi/2)]])
	R_y = Matrix([
	[cos(-pi/2),   0, sin(-pi/2)],
	[      0,      1,         0],
	[-sin(-pi/2),  0, cos(-pi/2)]])
	R_z = Matrix([[cos(pi), -sin(pi), 0],
	[sin(pi), cos(pi), 0],
	[0, 0, 1]])
	R_correction = simplify(R_z*R_y)

	T0_G = (T0_1* T1_2 * T2_3 * T3_4 * T4_5 * T5_6 * T6_G )
	#print(simplify(T0_G))
	quat1 = trans.quaternion_from_matrix(np.array(T0_G[:3,:3]*R_correction))
	#quat = [quat1[1],quat1[2],quat1[3],quat1[0]]
	(roll, pitch, yaw) =trans.euler_from_matrix(np.array(T0_G[:3,:3]*R_correction))
	#print(roll,pitch,yaw)
	#print('Angles',a1,a2,a3,a4,a5,a6)
	#print('Pose', quat)
	#print('Position',T0_G[0:4,3])
	data.append([T0_G[0,3],T0_G[1,3],T0_G[2,3],quat1[1],quat1[2],quat1[3],quat1[0],a1,a2,a3,a4,a5,a6])

with open('data.csv', "a") as output:
    writer = csv.writer(output, lineterminator='\n')
    writer.writerows(data)
#P_EE = Matrix([[-2.79193],[-1.76393],[1.48232]])
