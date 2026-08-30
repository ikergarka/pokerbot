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
    def __init__(self,orden,cash):
        self.agente=None
        self.cards=[]
        self.IsAI=False
        self.cash=cash
        self.totalbet=0
        self.puntos=0
        self.is_active=True
        self.orden=orden
    def turnAI(self, bool,agente):
        self.IsAI=bool
        self.agente=agente
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
        self.puntos=tr.Evaluator().evaluate(self.cards,commoncards)
        return self.puntos
    def deal(self,cartas):
        self.cards.extend(cartas)