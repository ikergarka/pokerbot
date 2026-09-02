import torch
import random
import torch.nn as nn
from collections import deque

class PokerTron(nn.Module):
    '''
    Modelo de DQN con ReLU para romper la linealidad
    '''
    def __init__(self, input_dim=109, output_dim=7):
        super(PokerTron,self).__init__()
        self.red=nn.Sequential(
            nn.Linear(input_dim,128),
            nn.ReLU(),
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Linear(128,output_dim)
        )
    def forward(self,x):
        return self.red(x)
class Agente():
    '''
    Clase que opera el modelo, permite accionarlo
    '''
    def __init__(self,modelo):
        self.modelo=modelo
        self.epsilon=1.0
        self.epsilon_min=0.05
        self.epsilon_decay=0.999
    def elegir(self,estado, mascara_validas):
        e=random.random()
        if e<self.epsilon:
            accioneslegales=[i for i, es_legal in enumerate(mascara_validas) if es_legal == 1]
            eleccion = random.choice(accioneslegales)
        else:
            tensorestado=torch.tensor(estado, dtype=torch.float32)
            with torch.no_grad():
                q_values= self.modelo(tensorestado)
                mask_tensor=torch.tensor([0.0 if v==1 else -1e9 for v in mascara_validas], dtype=torch.float32)
                q_values_filtrados=q_values+mask_tensor
                eleccion=torch.argmax(q_values_filtrados).item()
        return eleccion
    def actualizarepsilon(self):
        if self.epsilon>=self.epsilon_min:
            self.epsilon*=self.epsilon_decay
        else:
            self.epsilon=self.epsilon_min
    def guardar_pesos(self, filepath="/home/garka/proyectos/pokerbot/Pesos/pesos_pokertron.pth"):
        torch.save(self.modelo.state_dict(), filepath)
        print(f"Pesos del modelo guardados en {filepath}")

    def cargar_pesos(self, filepath="/home/garka/proyectos/pokerbot/Pesos/pesos_pokertron.pth"):
        import os
        if os.path.exists(filepath):
            self.modelo.load_state_dict(torch.load(filepath))
            print(f"Pesos del agente cargados desde {filepath}")
        else:
            print(f"No se encontró {filepath}. El agente empezará a aprender desde cero.")
class ReplayBuffer:
    '''
    Clase para gestionar la deque con más eficiencia
    '''
    def __init__(self, maxcapacity=10000):
        self.memoria=deque(maxlen=maxcapacity)

    def guardar(self, trainingdata):
        self.memoria.append(trainingdata)
    def muestrear(self,tamaño_lote):
        lote=random.sample(self.memoria,tamaño_lote)
        estados, acciones, recompensas, siguienteestado, done=zip(*lote)
        return estados, acciones, recompensas, siguienteestado, done
    def __len__(self):
        return len(self.memoria)
    def save(self, filepath="buffer_entrenamiento.pth"):
        """Guarda la memoria en un archivo físico."""
        # Convertimos el deque a una lista para serializarlo sin problemas
        torch.save(list(self.memoria), filepath)
        print(f"Buffer guardado exitosamente en {filepath}")

    def load(self, filepath="buffer_entrenamiento.pth"):
        """Carga la memoria desde un archivo físico."""
        import os
        if os.path.exists(filepath):
            datos = torch.load(filepath)
            self.memoria.clear()
            self.memoria.extend(datos)
            print(f"Buffer cargado. Tamaño actual: {len(self.memoria)}")
        else:
            print(f"No se encontró el archivo {filepath}. Se empezará desde cero.")

    
def entrenar(agente: Agente, buffer:ReplayBuffer, batch_size=64, gamma=0.99, optimizer=None):
    '''
    Recibe un buffer de s,a,r,s',done
    '''
    if len(buffer) < batch_size:
        return None 

    estados, acciones_tomadas, recompensas, siguientes_estados, dones = buffer.muestrear(batch_size)

    estados_t = torch.tensor(estados, dtype=torch.float32)
    acciones_t = torch.tensor(acciones_tomadas, dtype=torch.long)
    recompensas_t = torch.tensor(recompensas, dtype=torch.float32)
    dones_t = torch.tensor(dones, dtype=torch.float32)

    # los estados terminales guardan None como siguiente_estado -> los sustituimos por ceros,
    # igualmente se anulan multiplicando por (1 - done)
    dim = len(estados[0])
    siguientes_fix = [s if s is not None else [0.0] * dim for s in siguientes_estados]
    siguientes_t = torch.tensor(siguientes_fix, dtype=torch.float32)

    # Q(s,a) para las acciones que realmente se tomaron, base del DQN
    q_valores = agente.modelo(estados_t)#Aquí podríamos añadir un valor que fuesen los Q(s,a) de la red objetivo
    q_actual = q_valores.gather(1, acciones_t.unsqueeze(1)).squeeze(1)

    # aplicamos simplemente la ecuacion de Bellman
    with torch.no_grad():
        q_siguiente = agente.modelo(siguientes_t)
        max_q_siguiente, _ = q_siguiente.max(dim=1)
        target = recompensas_t + gamma * max_q_siguiente * (1 - dones_t)

    loss_fn = nn.MSELoss()
    loss = loss_fn(q_actual, target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()



    


            




