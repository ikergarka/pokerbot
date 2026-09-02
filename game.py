import treys as tr
from enum import Enum
from utils import player,acciones
import random
import model 
import torch
trainingdata=model.ReplayBuffer(10000)
def getSuit(card):
    palo_map={1:0,2:1,4:2,8:3}
    s=tr.Card().get_suit_int(card)
    suit=palo_map[s]
    return suit
def getEstado(round, player, pot, commoncards):
    cards=player.cards
    estado=[0.0]*109
    position=player.orden
    for c in cards:
        rank=tr.Card().get_rank_int(c)
        position=getSuit(c)*13+rank-1
        estado[position]=1
    for c in commoncards:
        rank=tr.Card().get_rank_int(c)
        position=getSuit(c)*13+rank+51
        estado[position]=1
    score=tr.Evaluator().evaluate(cards,commoncards)
    strength=1-score/7462
    estado[104]=strength
    estado[105]=pot
    estado[106]=player.cash
    estado[107]=player.totalbet
    estado[108]=round
    return estado
def getvalidActions(player, minbet):
    valid=[0]*len(acciones)
    valid[acciones.FOLD]=1
    valid[acciones.ALLIN]=1
    if player.cash>=minbet:
        valid[acciones.CALL]=1
    if player.cash>=minbet+minbet*0.1:
        valid[acciones.RAISE_DEC]=1
    if player.cash>=minbet+minbet*0.5:
        valid[acciones.RAISE_HALF]=1
    if player.cash>=minbet*2:
        valid[acciones.RAISE_DOUBLE]=1
    if minbet==0:
        valid[acciones.CHECK]=1
    return valid
    
def headsup(players,commoncards):
    '''
    Compara las manos utilizando Treys
    '''
    puntos=[p.puntuar(commoncards) for p in players]
    puntuacion=dict(zip(puntos,players))
    ganador=puntuacion[max(puntos)]
    return ganador

def grabar_paso(trayectorias,p,estado,accion):
    '''
    Graba las decisiones tomadas y el estado en trayectorias de forma que va añadiendo diccionarios a la lista de acciones del jugador
    '''
    trayectorias.setdefault(p.orden,[]).append({
        "estado":estado,
        "accion":accion,
    })
def cerrar_buffer(trayectorias,players,winner,pot, buy_in=500):
    '''
    Saca la informacion de las listas de diccionarios de trayectorias y graba la informacion en la deque de uno de los dos jugadores
    '''
    n=random.randint(0,1)
    p=players[n]
    pasos=trayectorias.get(p.orden,[])

    if p is winner:
        delta=pot-p.totalbet
    else:
        delta=-p.totalbet
    recompensa=delta/buy_in
    for i, paso in enumerate(pasos):
        done = (i == len(pasos) - 1)
        nuevo_estado = pasos[i + 1]["estado"] if not done else None
        r = recompensa if done else 0.0

        trainingdata.guardar((
            paso["estado"],
            paso["accion"],
            r,
            nuevo_estado,
            done)
        )
    return p


        
def Partida(agente=None):
    """
    Dos jugadores juegan una partida de poker. Se reparten las cartas y se determina el ganador.
    Graba las acciones que va tomando en un diccionario llamado trayectorias, que luego hace append a la variable global trainingdata, que es un deque detamaño 10000
    """
    player1=player(1,500)
    player2=player(2,500)
    players=[player1,player2]
    for p in players:
        p.turnAI(True,agente)
    deck=tr.Deck()

    round,pot,minbet=1,0, 10
    commoncards=[]
    juego_terminado=False
    trayectorias={}
    #flop
    for p in players:
        cards=deck.draw(2)
        p.deal(cards)
    commoncards.extend(deck.draw(3))
    while not juego_terminado and round<3:
        ronda_terminada=False
        while not ronda_terminada:
            for p in players:
                if not p.is_active:
                    ronda_terminada=True
                    continue
                estado=getEstado(round,p,pot,commoncards)
                accion=getvalidActions(p,minbet)
                decision=p.makedecision(estado,accion)
                if p.IsAI==True:
                    grabar_paso(trayectorias,p,estado,decision)
                if decision==0:#CALL
                    minbet=p.bet(minbet)
                    pot+=minbet
                    ronda_terminada=True
                elif decision==1:#RAISE_DEC
                    pot+=p.bet(minbet+minbet*0.1)
                    minbet+=minbet*0.1
                elif decision==2:#RAISE_HALF
                    pot+=p.bet(minbet+minbet*0.5)
                    minbet+=minbet*0.5
                elif decision==3:#RAISE_DOUBLE
                    pot+=p.bet(minbet*2)
                    minbet+=minbet
                elif decision==4:#FOLD
                    juego_terminado=True
                    ronda_terminada=True
                elif decision==5:#ALLIN
                    p.is_active=False
                    pot+=p.bet(p.cash)
                elif decision==6:
                    ronda_terminada=True

        if not juego_terminado:
            commoncards.extend(deck.draw())
            round+=1
        if juego_terminado and len(commoncards)<3:
            commoncards.extend(deck.draw(5-len(commoncards)))

    winner=headsup(players,commoncards)
    AIplayer=cerrar_buffer(trayectorias,players,winner,pot)
    return winner,AIplayer



        
