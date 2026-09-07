
import treys as tr
from utils import player,acciones
import model 
import torch
import game
import matplotlib.pyplot as plt
import numpy as np
from collections import deque

modelo_rapido = model.PokerTron()
agente_rapido=model.Agente(modelo_rapido)
optimizer_rapido = torch.optim.Adam(modelo_rapido.parameters(), lr=1e-3)

modelo_lento = model.PokerTron()
agente_lento = model.Agente(modelo_lento)
optimizer_lento = torch.optim.Adam(modelo_lento.parameters(),lr=1e-3/100)

game.trainingdata_rapido.load("/home/garka/proyectos/pokerbot/Pesos/buffer_entrenamiento_rapido.pth")
agente_rapido.cargar_pesos("/home/garka/proyectos/pokerbot/Pesos/pesos_pokertron_rapido.pth")

game.trainingdata_lento.load("/home/garka/proyectos/pokerbot/Pesos/buffer_entrenamiento_lento.pth")
agente_lento.cargar_pesos("/home/garka/proyectos/pokerbot/Pesos/pesos_pokertron_lento.pth")

historial_episodios = []
historial_loss_lento = []
historial_winrate_lento = []
historial_epsilon_lento = []

historial_loss_rapido = []
historial_winrate_rapido = []
historial_epsilon_rapido = []

# Usamos un deque para calcular el winrate de las últimas 100 partidas de cada agente
victorias_recientes_rapido = deque(maxlen=100)
loss_recientes_rapido=deque(maxlen=100)

victorias_recientes_lento = deque(maxlen=100)
loss_recientes_lento=deque(maxlen=100)

for episodio in range(20000):
    winner=game.Partida(agente_rapido,agente_lento)  # rellena game.trainingdata (tu ReplayBuffer global)
    if winner.agente==agente_rapido:
        victorias_recientes_rapido.append(1)
        victorias_recientes_lento.append(0)
    elif winner.agente==agente_lento:
        victorias_recientes_lento.append(1)
        victorias_recientes_rapido.append(0)

    print(winner.orden)
    loss_rapido = model.entrenar(agente_rapido, game.trainingdata, batch_size=64, optimizer=optimizer_rapido)
    loss_lento = model.entrenar(agente_lento,game.trainingdata, batch_size=64,optimizer=optimizer_lento)
    agente_rapido.actualizarepsilon()
    agente_lento.actualizarepsilon()
    loss_recientes_rapido.append(loss_rapido) if loss_rapido is not None else 0.0
    loss_recientes_lento.append(loss_lento) if loss_lento is not None else 0.0
    if loss_rapido is not None and loss_lento is not None:
        #Definimos las estadísticas para visualizar los datos de los dos modelos enfrentados
        winrate_lento = sum(victorias_recientes_lento) / len(victorias_recientes_lento) if len(victorias_recientes_lento) > 0 else 0
        winrate_rapido = sum(victorias_recientes_rapido) / len(victorias_recientes_rapido) if len(victorias_recientes_rapido) > 0 else 0

        current_loss_lento = sum(loss_recientes_lento)/len(loss_recientes_lento) if len(loss_recientes_lento)>0 else 0.0
        current_loss_rapido = sum(loss_recientes_rapido)/len(loss_recientes_rapido) if len(loss_recientes_rapido)>0 else 0.0

        historial_episodios.append(episodio)

        historial_loss_lento.append(current_loss_lento)
        historial_winrate_lento.append(winrate_lento)
        historial_epsilon_lento.append(agente_lento.epsilon)

        historial_loss_rapido.append(current_loss_rapido)
        historial_winrate_rapido.append(winrate_rapido)
        historial_epsilon_rapido.append(agente_rapido.epsilon)        

    if episodio > 0 and episodio % 499 == 0:
        game.trainingdata_lento.save("/home/garka/proyectos/pokerbot/Pesos/buffer_entrenamiento_lento.pth")
        agente_lento.guardar_pesos("/home/garka/proyectos/pokerbot/Pesos/pesos_pokertron_lento.pth")

        game.trainingdata_rapido.save("/home/garka/proyectos/pokerbot/Pesos/buffer_entrenamiento_rapido.pth")
        agente_rapido.guardar_pesos("/home/garka/proyectos/pokerbot/Pesos/pesos_pokertron_rapido.pth")
        plt.figure(figsize=(12, 8))
        
        # Subplot 1: Tasa de Victorias Rapido
        plt.subplot(3, 1, 1)
        plt.plot(historial_episodios, historial_winrate_rapido, label='Win Rate', color='green')
        plt.title('Tasa de Victorias (Win Rate) de las últimas 100 partidas')
        plt.ylabel('Win Rate')
        plt.legend()
        plt.grid(True)
        
        # Subplot 2: Loss rapido
        plt.subplot(3, 1, 2)
        plt.plot(historial_episodios, historial_loss_rapido, label='Loss', color='red')
        plt.title('Evolución del Error (Loss)')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Subplot 3: Epsilon Rapido
        plt.subplot(3, 1, 3)
        plt.plot(historial_episodios, historial_epsilon_rapido, label='Epsilon', color='blue')
        plt.title('Decaimiento de la Exploración (Epsilon)')
        plt.xlabel('Episodios')
        plt.ylabel('Epsilon')
        plt.legend()
        plt.grid(True)

        # Subplot 4: Tasas de Victorias Lento
        plt.subplot(3, 2, 1)
        plt.plot(historial_episodios, historial_winrate_lento, label='Win Rate', color='green')
        plt.title('Tasa de Victorias Lento')
        plt.ylabel('Win Rate')
        plt.legend()
        plt.grid(True)
        
        # Subplot 2: Loss Lento
        plt.subplot(3, 2, 2)
        plt.plot(historial_episodios, historial_loss_lento, label='Loss', color='red')
        plt.title('Evolución del Error Lento')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Subplot 3: Epsilon
        plt.subplot(3, 2, 3)
        plt.plot(historial_episodios, historial_epsilon_lento, label='Epsilon', color='blue')
        plt.title('Decaimiento de la Exploración Lento')
        plt.xlabel('Episodios')
        plt.ylabel('Epsilon')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('/home/garka/proyectos/pokerbot/Capturas/grafica_rendimiento.png') # Guarda la imagen en tu carpeta
        plt.close() # Cerramos la figura para liberar memoria
        print("--> Gráfica actualizada y guardada como 'grafica_rendimiento.png'")
game.trainingdata_lento.save("/home/garka/proyectos/pokerbot/Pesos/buffer_entrenamiento_lento.pth")
game.trainingdata_rapido.save("/home/garka/proyectos/pokerbot/Pesos/buffer_entrenamiento_rapido.pth")
agente_lento.guardar_pesos("/home/garka/proyectos/pokerbot/Pesos/pesos_pokertron_lento.pth")
agente_rapido.guardar_pesos("/home/garka/proyectos/pokerbot/Pesos/pesos_pokertron_rapido.pth")