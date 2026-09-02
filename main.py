
import treys as tr
from utils import player,acciones
import model 
import torch
import game
import matplotlib.pyplot as plt
import numpy as np
from collections import deque

modelo = model.PokerTron()
agente_objetivo=model.Agente(modelo)
agente = model.Agente(modelo)
optimizer = torch.optim.Adam(modelo.parameters(), lr=1e-3)

game.trainingdata.load("/home/garka/proyectos/pokerbot/Pesos/buffer_entrenamiento.pth")
agente.cargar_pesos("/home/garka/proyectos/pokerbot/Pesos/pesos_pokertron.pth")

historial_episodios = []
historial_loss = []
historial_winrate = []
historial_epsilon = []

# Usamos un deque para calcular el winrate de las últimas 100 partidas
victorias_recientes = deque(maxlen=100)
loss_recientes=deque(maxlen=100)

for episodio in range(500):
    winner,AIPlayer=game.Partida(agente)  # rellena game.trainingdata (tu ReplayBuffer global)
    if winner==AIPlayer:
        victorias_recientes.append(1)
    else:
        victorias_recientes.append(0)
    print(winner.orden)
    loss = model.entrenar(agente, game.trainingdata, batch_size=64, optimizer=optimizer)
    agente.actualizarepsilon()
    loss_recientes.append(loss) if loss is not None else 0.0
    if loss is not None:
        winrate = sum(victorias_recientes) / len(victorias_recientes) if len(victorias_recientes) > 0 else 0
        current_loss = sum(loss_recientes)/len(loss_recientes) if len(loss_recientes)>0 else 0.0
        historial_episodios.append(episodio)
        historial_loss.append(current_loss)
        historial_winrate.append(winrate)
        historial_epsilon.append(agente.epsilon)
    if episodio > 0 and episodio % 499 == 0:
        game.trainingdata.save("/home/garka/proyectos/pokerbot/Pesos/buffer_entrenamiento.pth")
        agente.guardar_pesos("/home/garka/proyectos/pokerbot/Pesos/pesos_pokertron.pth")
        plt.figure(figsize=(12, 8))
        
        # Subplot 1: Tasa de Victorias
        plt.subplot(3, 1, 1)
        plt.plot(historial_episodios, historial_winrate, label='Win Rate', color='green')
        plt.title('Tasa de Victorias (Win Rate) de las últimas 100 partidas')
        plt.ylabel('Win Rate')
        plt.legend()
        plt.grid(True)
        
        # Subplot 2: Loss
        plt.subplot(3, 1, 2)
        plt.plot(historial_episodios, historial_loss, label='Loss', color='red')
        plt.title('Evolución del Error (Loss)')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Subplot 3: Epsilon
        plt.subplot(3, 1, 3)
        plt.plot(historial_episodios, historial_epsilon, label='Epsilon', color='blue')
        plt.title('Decaimiento de la Exploración (Epsilon)')
        plt.xlabel('Episodios')
        plt.ylabel('Epsilon')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('/home/garka/proyectos/pokerbot/Capturas/grafica_rendimiento.png') # Guarda la imagen en tu carpeta
        plt.close() # Cerramos la figura para liberar memoria
        print("--> Gráfica actualizada y guardada como 'grafica_rendimiento.png'")
game.trainingdata.save("/home/garka/proyectos/pokerbot/Pesos/buffer_entrenamiento.pth")
agente.guardar_pesos("/home/garka/proyectos/pokerbot/Pesos/pesos_pokertron.pth")