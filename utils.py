from enum import IntEnum
from random import randint
import treys as tr
import model
class acciones(IntEnum):
    CALL=0
    RAISE_DEC=1
    RAISE_HALF=2
    RAISE_DOUBLE=3
    FOLD=4
    ALLIN=5
    CHECK=6

class player():
    '''
    Clase del Jugador
    '''
    def __init__(self,orden,cash,agente):
        self.agente=agente
        self.cards=[]
        self.IsAI=True
        self.cash=cash
        self.totalbet=0
        self.is_active=True
        self.orden=orden
    def bet(self, amount):
        self.cash-=amount
        self.totalbet+=amount
        return amount
    def makedecision(self, estado,accvalidas:list):
        if self.IsAI==False:
            valid=False
            while not valid:
                accion=randint(1,7)-1
                if accvalidas[accion]!=1:
                    continue
                else:
                    valid=True
        else:
            accion=self.agente.elegir(estado, accvalidas)
        return accion
    def puntuar(self,commoncards):
        puntos=tr.Evaluator.evaluate(self.cards,commoncards)
        return puntos
    def deal(self,cartas):
        self.cards.extend(cartas)