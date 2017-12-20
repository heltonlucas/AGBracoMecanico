# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 16:33:29 2017

@author: helto
"""
import numpy as np


erro = 0.1              #AG evolui enquanto o erro de posição for maior que 0,1 cm
tam_populacao = 80
taxa_mutacao = 0.03         #erroaprox = 0.0000001 #popp_tam = 200 #mutacao = 0.05 #geracoes = 25
geracoes = 200
tam_juntas = 10

''' Fatores de ponderação para erro angular '''
W1 = 0.5 
W2 = 0.5

'''Coordenadas para movimento em X e Y (Melhor solucao do Artigo) '''
x = -10   
y = 15


class Coordenada:
    def __init__(self, x, y):

        self.x = x
        self.y = y

class Angulos_do_Braco:
    def __init__(self, A1, A2, A3):
        self.A1 = A1
        self.A2 = A2
        self.A3 = A3


angulo_inicial = Angulos_do_Braco(A1=0.0307, A2=1.8449, A3=1.5691)

''' Com os anguloss (A1, A2, A3)  encontrasse X, Y '''
''' posicao da garra '''
def localBraco(angulos):

    x = tam_juntas * np.cos(angulos.A1) + tam_juntas * np.cos(angulos.A1 + angulos.A2) + tam_juntas * np.cos(angulos.A1 + angulos.A2 + angulos.A3)

    y = tam_juntas * np.sin(angulos.A1) + tam_juntas * np.sin(angulos.A1 + angulos.A2) + tam_juntas * np.sin(angulos.A1 + angulos.A2 + angulos.A3)

    return x, y


def angulo(pontoInicio, pontoFinal):

    ang = np.arctanh(
        (pontoFinal.y - pontoInicio.y/pontoFinal.x - pontoInicio.x)*(180/np.pi))

    return ang

''' Calcula o erro de posicionamento por meio da distância Euclidiana '''
''' entre as coordenadas atuais e finais do manipulador '''
def erro_posicionamento(point_current, ponto_alvo):

    erro = np.sqrt((point_current.x - ponto_alvo.x)**2
                    + (point_current.y - ponto_alvo.y)**2)

    return erro

'''Evitar que o AG atinja o ponto final com grandes deslocamentos angulares'''
def erro_desloc_angular(AngIni, AngFin):

    erro = np.sqrt((AngFin.A1 - AngIni.A1)**2 +
                    (AngFin.A2 - AngIni.A2)**2 +
                    (AngFin.A3 - AngIni.A3)**2)

    return erro

''' Avaliar o erro de pos da garra, busca dimuir o Ea e Ep'''
def fitness(point_i, point_f, AngIni, AngFin):

    value = 1/((W1*erro_posicionamento(point_i, point_f))
               + (W2*erro_desloc_angular(AngIni, AngFin)))

    return value

'''Valores da pop entre -3,14 a 3,14'''
def gerarValsInteralo(min=(-np.pi), max=np.pi): 

    valorLimite = 0.000000000000001

    return np.random.uniform(min, (max+valorLimite))

'''Metod de selecao torneio'''
def selecao(pop, ponto_alvo, angulos_iniciais, sort=2):

    ganhadores = []
    n = len(pop)

    for _ in np.arange(sort):

        sorted_01 = np.random.randint(n)
        sorted_02 = np.random.randint(n)

        selecionado1 = pop[sorted_01]
        selecionado2 = pop[sorted_02]

        angles_01 = Angulos_do_Braco(selecionado1[0], selecionado1[1], selecionado1[2])
        angles_02 = Angulos_do_Braco(selecionado2[0], selecionado2[1], selecionado2[2])

        x_1, y_1 = localBraco(angles_01)
        x_2, y_2 = localBraco(angles_02)

        point_01 = Coordenada(x_1, y_1)
        point_02 = Coordenada(x_2, y_2)

        fitness_1 = fitness(point_01, ponto_alvo, angulos_iniciais, angles_01)
        fitness_2 = fitness(point_02, ponto_alvo, angulos_iniciais, angles_02)

        if fitness_1 < fitness_2:

            ganhadores.append(selecionado1)
        else:
            ganhadores.append(selecionado2)

    return ganhadores[0], ganhadores[1]

'''Melhores do torneio'''
def melhoresIndividuos(pop, ponto_alvo, angulos_iniciais):

    maisApto = None

    for indiviual in pop:

        angulos = Angulos_do_Braco(indiviual[0], indiviual[1], indiviual[2])

        x, y = localBraco(angulos)

        ponto = Coordenada(x, y)

        fitness_value = fitness(ponto, ponto_alvo, angulos_iniciais, angulos)

        if maisApto is None:
            maisApto = (fitness_value, indiviual)
        else:
            if maisApto[0] > fitness_value:
                maisApto = (fitness_value, indiviual)

    return maisApto

'''Cruzamneto com media aritimetnica'''
def crossover(pai, mae, taxa_mutacao):

    A1 = (pai[0] + mae[0])/2
    A2 = (pai[1] + mae[1])/2
    A3 = (pai[2] + mae[2])/2

    filho = [A1, A2, A3]

    taxaMutRandom = np.random.random()

    if taxaMutRandom <= taxa_mutacao:

        filho = mutacao(filho)

    return filho

'''genes alterados de acordo com a probabilidade pm de mutação'''
def mutacao(cromossome):

    n_g = len(cromossome)

    sorted = np.random.randint(n_g)

    random_value = gerarValsInteralo()

    cromossome[sorted] = random_value

    return cromossome


def gerarPop(n=10, gene=3):

    pop = []

    for _ in np.arange(n):

        cromossomo = []

        for _ in np.arange(gene):

            value = gerarValsInteralo()
            cromossomo.append(value)

        pop.append(cromossomo)
    return pop

'''Pop 100 com 3 genes'''
pop = gerarPop(n = tam_populacao)

x_inicial, y_inicial = localBraco(angulo_inicial)


ponto_alvo = Coordenada(x, y)

for x in np.arange(1,geracoes+1):

    pop_atual = []

    population_size_temp = len(pop_atual)

    while population_size_temp <= tam_populacao:

        pai, mae = selecao(pop, ponto_alvo, angulo_inicial)

        filho = crossover(pai, mae, taxa_mutacao)

        pop_atual.append(filho)

        population_size_temp = len(pop_atual)

    pop = pop_atual.copy()

    ''' Calcular a solucao dos pontos '''
    solution = melhoresIndividuos(pop, ponto_alvo, angulo_inicial)
    
    print('Geração ->', x)
    print('Ang ->', solution[1] , 'Fit ->', solution[0] )
    
