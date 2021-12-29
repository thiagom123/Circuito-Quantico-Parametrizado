import numpy as np
import qiskit


def Rz(theta):
    return np.array([[np.e**(-1j*theta/2), 0],[0, np.e**(1j*theta/2)]])

def Ry(theta):
    return np.array([[np.cos(theta/2), -np.sin(theta/2)],[np.sin(theta/2), np.cos(theta/2)]])

def Z():
    return np.array([[1,0] , [0, -1]])

H = 1 / np.sqrt(2) * np.array([[1, 1], [1, -1]])

def rzryrz(U):
    '''
    Lab2 - questão 1
    :param U: matriz unitária 2 x 2
    :return: [alpha, beta, gamma e delta]
            U = e^(1j * alpha) * Rz(beta) * Ry(gamma) * Rz(delta)
    '''

    # -----------------
    # Seu código aqui

    x00 = np.angle(U[0][0])
    x01= np.angle(U[0][1])+np.pi
    x10 = np.angle(U[1][0])
    alpha = (x01+x10)/2
    beta = x10-x00
    gamma = 2*np.arccos(np.abs(U[0][0]))
    delta = x01-x00
    # -----------------

    return [alpha, beta, gamma, delta]



def operador_controlado(V):
    '''
    Lab2 - questão 2
    :param V: matriz unitária 2 x 2
    :return: circuito quântico com dois qubits aplicando o
             o operador V controlado.
    '''

    circuito = qiskit.QuantumCircuit(2)

    #-----------------
    # Seu código aqui
    ParamDecomposicao = rzryrz(V)
    alpha = ParamDecomposicao[0]
    beta = ParamDecomposicao[1]
    gamma = ParamDecomposicao[2]
    delta = ParamDecomposicao[3]
    # C = Rz((ParamDecomposicao[3]-ParamDecomposicao[1])/2)
    circuito.rz((delta - beta)/2, 1)
    circuito.cx(0, 1)
    # B = Ry((-1)*ParamDecomposicao[2]/2)@Rz((-1)*(ParamDecomposicao[3]+ParamDecomposicao[1])/2)
    circuito.rz((-1) * (delta+beta)/2, 1)
    circuito.ry((-1) * gamma/2, 1)
    circuito.cx(0, 1)
    # A = Rz(ParamDecomposicao[1])@Ry(ParamDecomposicao[2])
    circuito.ry(gamma/2, 1)
    circuito.rz(beta, 1)

    circuito.p(alpha,0)
    circuito.draw()

    # -----------------

    return circuito