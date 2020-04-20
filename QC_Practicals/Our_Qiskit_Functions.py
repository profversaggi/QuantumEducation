from qiskit import ClassicalRegister, QuantumRegister, QuantumCircuit, Aer, execute
import numpy as np
import math as m
import scipy as sci

S_simulator = Aer.backends(name='statevector_simulator')[0]
M_simulator = Aer.backends(name='qasm_simulator')[0] 

#Displaying Results 
def Wavefunction( obj , *args, **kwargs):
#Displays the waveftmction of the quantum system 
	if(type(obj) == QuantumCircuit ):
		statevec = execute( obj, S_simulator, shots=1 ).result().get_statevector()
	if(type(obj) == np.ndarray):
		statevec = obj
	sys = False
	NL = False
	dec = 5
	if 'precision' in kwargs:
		dec = int( kwargs['precision'] )
	if 'column' in kwargs:
		NL = kwargs['column']
	if 'systems' in kwargs:
		systems = kwargs['systems']
		sys = True
		last_sys = int(len(systems)-1)
		show_systems = []
		for s_chk in np.arange(len(systems)):
			if( type(systems[s_chk]) != int ):
				raise Exception('systems must be an array of all integers')
		if 'show_systems' in kwargs:
			show_systems = kwargs['show_systems']
			if( len(systems)!= len(show_systems) ):
				raise Exception('systems and show_systems need to be arrays of equal length')
			for ls in np.arange(len(show_systems)):
				if((show_systems[ls] != True) and (show_systems[ls] != False)):
					raise Exception('show_systems must be an array of Truth Values')
				if(show_systems[ls] == True):
					last_sys = int(ls) 
		else:
			for ss in np.arange(len(systems)):
				show_systems.append(True)
	wavefunction = ''
	qubits = int(m.log(len(statevec),2))
	for i in np.arange( int(len(statevec))):
		#print(wavefunction)
		value = round(statevec[i].real, dec) + round(statevec[i].imag, dec) * 1j
		if( (value.real != 0) or (value.imag != 0)):
			state = list(Binary(int(i),int(2**qubits)))
			state.reverse()
			state_str = ''
			#print(state)
			if( sys == True ): #Systems and SharSystems 
				k = 0 
				for s in np.arange(len(systems)):
					if(show_systems[s] == True):
						if(int(s) != last_sys):
							state.insert(int(k + systems[s]), '>|' )
							k = int(k + systems[s] + 1)
						else:
							k = int(k + systems[s]) 
					else:
						for s2 in np.arange(systems[s]):
							del state[int(k)]
			for j in np.arange(len(state)):
				if(type(state[j])!= str):
					state_str = state_str + str(int(state[j])) 
				else:
					state_str = state_str + state[j]
			#print(state_str)
			#print(value)
			if( (value.real != 0) and (value.imag != 0) ):
				if( value.imag > 0):
					wavefunction = wavefunction + str(value.real) + '+' + str(value.imag) + 'j |' + state_str + '>   '
				else:
					wavefunction = wavefunction + str(value.real) + '' + str(value.imag) + 'j |' + state_str +  '>   '
			if( (value.real !=0 ) and (value.imag ==0) ):
				wavefunction = wavefunction  + str(value.real) + '  |' + state_str + '>   '
			if( (value.real == 0) and (value.imag != 0) ):
				wavefunction = wavefunction + str(value.imag)  + 'j |' + state_str + '>   '
			if(NL):
				wavefunction = wavefunction + '\n'
		#print(NL)
	
	#print(wavefunction)
	return wavefunction


def Measurement(quantumcircuit, *args, **kwargs): 
	#Displays the measurement results of a quantum circuit 
	p_M = True
	S = 1
	ref = False
	NL = False
	if 'shots' in kwargs:
		S = int(kwargs['shots'])
	if 'return_M' in kwargs:
		ret = kwargs['return_M']
	if 'print_M' in kwargs:
		p_M = kwargs['print_M']
	if 'column' in kwargs:
		NL = kwargs['column']
	M1 = execute(quantumcircuit, M_simulator, shots=S).result().get_counts(quantumcircuit)
	M2 = {}
	k1 = list(M1.keys())
	v1 = list(M1.values())
	for k in np.arange(len(k1)):
		key_list = list(k1[k])
		new_key = ''
		for j in np.arange(len(key_list)):
			new_key = new_key+key_list[len(key_list)-(j+1)]
		M2[new_key] = v1[k]
	if(p_M):
		k2 = list(M2.keys())
		v2 = list(M2.values())
		measurements = ''
		for i in np.arange(len(k2)):
			m_str = str(v2[i])+'|'
			for j in np.arange(len(k2[i])):
				if(k2[i][j] == '0'):
					m_str = m_str + '0' 
				if(k2[i][j] == '1'):
					m_str = m_str + '1'
				if( k2[i][j] == ' ' ):
					m_str = m_str +'>|'
			m_str = m_str + '>   '
			if(NL):
				m_str = m_str + '\n'
			measurements = measurements + m_str
		#print(measurements)
		return measurements
	if(ref):
		return M2


#Math Operations
def Oplus(bit1,bit2): 
	'''Adds too bits of O's and 1's (modulo 2)'''
	bit = np.zeros(len(bit1))
	for i in np.arange( len(bit) ):
		if( (bit1[i]+bit2[i])%2 == 0 ):
			bit[i] = 0
		else: 
			bit[i] = 1
	return bit 


def Binary(number,total): 
#Converts a number to binary, right to left LSB 152 153 o
	qubits = int(m.log(total,2))
	N = number
	b_num = np.zeros(qubits)
	for i in np.arange(qubits):
		if( N/((2)**(qubits-i-1)) >= 1 ):
			b_num[i] = 1
			N = N - 2 ** (qubits-i-1)
	B = [] 
	for j in np.arange(len(b_num)):
		B.append(int(b_num[j]))
	return B

def From_Binary(s):
    num = 0
    for i in np.arange(len(s)):
        num = num + s[int(0-(i+1))] * 2 ** (i)
    return num

def B2D(in_bi):
    len_in = len(in_bi)
    in_bi = in_bi[::-1]
    dec = 0
    for i in range(0,len_in):
        if in_bi[i] != '0':
            dec += 2**i
    return dec

#  Custom Gates
def x_Transformation(qc, qreg, state): 
#Tranforms the state of the system, applying X gates according to as in the vector 'state' 
	for j in np.arange(len(state)):
		if( int(state[j]) == 0 ):
			qc.x( qreg[int(j)] ) 


def n_NOT(qc, control, target, anc): 
#performs an n-NOT gate
	n = len(control)
	instructions = []
	active_ancilla = []
	q_unused = []
	q = 0
	a = 0
	while( (n > 0) or (len(q_unused) != 0) or (len(active_ancilla) != 0) ):
		if( n > 0 ):
			if( (n-2) >= 0 ):
				instructions.append( [control[q], control[q+1], anc[a]] )
				active_ancilla.append(a)
				a += 1
				q += 2
				n = n - 2
			if( (n-2) == -1 ):
				q_unused.append( q )
				n = n - 1
		elif( len(q_unused) != 0 ):
			if(len(activeancilla)!=1):
				instructions.append( [control[q], anc[active_ancilla[0]], anc[a]] )
				del active_ancilla[0]
				del q_unused[0]
				active_ancilla.append(a)
				a = a + 1
			else:
				instructions.append( [control[q], anc[active_ancilla[0]], target] )
				del active_ancilla[0]
				del q_unused[0]
		elif( len(active_ancilla) != 0 ):
			if( len(active_ancilla) > 2 ):
				instructions.append( [anc[active_ancilla[0]], anc[active_ancilla[1]], anc[a]] )
				active_ancilla.append(a)
				del active_ancilla[0]
				del active_ancilla[0]
				a = a + 1
			elif( len(active_ancilla) == 2):
				instructions.append([anc[active_ancilla[0]], anc[active_ancilla[1]], target])
				del active_ancilla[0]
				del active_ancilla[0]
	for i in np.arange( len(instructions) ):
		qc.ccx( instructions[i][0], instructions[i][1], instructions[i][2] )
	del instructions[-1]
	for i in np.arange( len(instructions) ):
		qc.ccx( instructions[0-(i+1)][0], instructions[0-(i+1)][1], instructions[0-(i+1)][2] )



def Control_Instruction( qc, vec ): 
#Ammends the proper quantum circuit instruction based on the input 'vec'
#Used for the function 'n_Control_U
	if( vec[0] == 'X' ):
		qc.cx( ver[1], vec[2] )
	if( vec[0] == 'Z' ):
		qc.cz( ver[1], vec[2] )
	if( vec[0] == 'PRASE' ):
		qc.cu1( vec[2], vec[1], vec[3] )
	if( vec[0] == 'SWAP' ):
		qc.cswap( vec[1], vec[2], vec[3] ) 

def X_Transformation(qc, qreg, state):
	for j in np.arange(len(state)):
		if( int(state[j]) == 0):
			qc.x( qreg[int(j)])



def sinmons_solver(E,N):
	'''Returns an array of s_prime candidates
	'''
	s_primes = []
	for s in np.ararge(1,2**N):
		sp = Binary( int(s), 2**N )
		candidate = True
		for e in np.arange( len(E) ):
			value = 0
			for i in np.arange( N ):
				value = value + sp[i]*E[e][i]
			if(value%2==1):
				candidate=False
		if(candidate):
			s_primes.append(sp)
	return s_primes


def Grover_Oracle(mark, qc, q, an1, an2): 
	'''
	picks out the marked state and applies a negative phase 
	'''
	qc.h( an1[0] )
	X_Transformation(qc, q, mark)
	if( len(mark) > 2 ):
		n_NOT( qc, q, an1[0], an2 )
	if( len(mark) == 2 ):
		qc.ccx( q[0], q[1], an1[0] )
	X_Transformation(qc, q, mark)
	qc.h( an1[0] )

def Grover_Diffusion(mark, qc, q, an1, an2): 
	'''
	ammends the instructions for a Grover Diffusion Operation to the Quartu rcuit
	'''
	zeros_state = []
	for i in np.arange( len(mark) ):
		zeros_state.append( 0 )
		qc.h( q[int(i)] )
	Grover_Oracle(zeros_state, qc, q, an1, an2)
	for j in np.arange( len(mark) ):
		qc.h( q[int(j)] )



def Grover(Q, marked): 
	'''
	Amends all the instructions for a Grover Search 
	'''
	q = QuantumRegister(Q,name='q')
	an1 = QuantumRegister(1,name='anc')
	an2 = QuantumRegister(Q-2,name='nanc')
	c = ClassicalRegister(Q,name='c')
	qc = QuantumCircuit(q,an1,an2,c,name='qc')
	for j in np.arange(Q):
		qc.h( q[int(j)] )
	qc.x( an1[0] )
	iterations = round( m.pi/4 * 2**(Q/2.0) )
	for i in np.arange( iterations ):
		Grover_Oracle(marked, qc, q, an1, an2)
		Grover_Diffusion(marked, qc, q, an1, an2)
	return qc, q, an1, an2, c
