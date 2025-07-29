import sys
import gym
import random
import numpy as np
from time import sleep
from collections import deque
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential
from multiprocessing import Process, Queue


#DEFINICIÓN DE AGENTE
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.render = False

        #Estados y acciones
        self.state_size = state_size
        self.action_size = action_size

        #Constantes
        self.discount_factor = 0.99
        self.learning_rate = 0.001
        self.epsilon = 1.0
        self.epsilon_decay = 0.99
        self.batch_size = 64
        self.tau = 0.01

        #Creamos los modelos
        self.model = self.build_model()
        self.target_model = self.build_model()

        #Update del target (no es necesario)
        self.update_target_model()

    #Definimos el modelo particular
    def build_model(self):
        model = Sequential()
        model.add(Dense(64, input_dim=self.state_size, activation='relu',
                        kernel_initializer='he_uniform'))
        model.add(Dense(64, activation='relu',
                        kernel_initializer='he_uniform'))
        model.add(Dense(self.action_size, activation='linear',
                        kernel_initializer='he_uniform'))
        model.summary()
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    #Soft-update del target model
    def update_target_model(self):
        weights = self.model.get_weights() #pesos del modelo original
        target_weights = self.target_model.get_weights() #pesos del modelo auxiliar

        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau) #actualizo SOFT los pesos del modelo aux
        
        self.target_model.set_weights(target_weights)


    #Actuar según epsilon
    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            q_value = self.model.predict(state, verbose = 0)
            return np.argmax(q_value[0])

###########ACA EMPIEZA LA FORMULACIÓN PARA USAR 2 PROCESADORES
##Definimos un actor, que se encarga de actuar en el ambiente según un modelo particular y guardar experiencias en q1
##Y también definimos learner, que toma las experiencias del actor, las memoriza y aprende a partir de ellas, deja un modelo fiteado en q2.
##La idea es tratar de "solapar" lo más que se pueda estos procesos.

#Proceso de acción
def actor(q1, q2, q3):
    print('start process 1')
    env = gym.make('LunarLander-v2')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    #Agente particular
    agent = DQNAgent(state_size, action_size)
    scores = []

    #Número máximo de episodios a entrenar
    for e in range(10000):
        done = False
        score = 0
        #Reseteamos el ambiente con la misma semilla (siempre es el mismo ambiente)
        state = env.reset(seed=42)
        state = np.reshape(state, [1, state_size])

        #Tiempo dentro de un episodio
        for _ in range(1000):

            #Agarramos el último modelo ajustado para actuar.
            if q2.qsize() > 0:
                while q2.qsize() > 1:
                    q2.get()
                model = q2.get()
                agent.model.set_weights(model)

            #Actua
            action = agent.get_action(state)
            #Ambiente después de la acción
            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            
            #Ponemos en la memoria s, a, r, s'
            q1.put([state, action, reward, next_state, done])

            score += reward
            state = next_state

            #Si terminó salir del episodio (duerme 0.05 para asegurar que no se superponga)    
            if done:
                sleep(0.05)
                print("episode:", e, "  score:", score, "  epsilon:", agent.epsilon, "Factor Discount:",agent.discount_factor)
                scores.append(score)
                break

                #Si el promedio de los últimos 10 episodios supera los 200
        if np.mean(scores[-min(10, len(scores)):]) > 200:
            agent.model.save_weights("DQN_MP.hdf5") #Guardo los pesos del modelo
            np.save('scores', scores) #Guardo los scores del modelo
            q3.put(True)
            sys.exit()
        
        #Aplico el epsilon decay después de cada episodio
        if agent.epsilon > 0.01:
            agent.epsilon *= agent.epsilon_decay

#Proceso de aprendizaje
def learner(q1, q2, q3):
    print('start process 2')
    replay_memory = deque(maxlen=100000) #Memoriza los últimos 100000 frames (como min 100 episodios)
    agent = DQNAgent(8, 4)
    count = 0 #Contador para controlar el update del target

    while True:
        count += 1
        #Mientras tenga alguna experiencia, la sumo a la memoria
        while q1.qsize() > 0:
            sample = q1.get()
            replay_memory.append(sample)

        if q3.qsize() > 0:
            sys.exit()

        #Proceso de aprendizaje       
        if len(replay_memory) > agent.batch_size:
            mini_batch = random.sample(replay_memory, agent.batch_size)

            states = np.zeros((agent.batch_size, agent.state_size))
            next_states = np.zeros((agent.batch_size, agent.state_size))
            actions, rewards, dones = [], [], []

            #Selecciono los s, a, r, s' que van en mi mini_batch
            for i in range(agent.batch_size):
                states[i] = mini_batch[i][0]
                actions.append(mini_batch[i][1])
                rewards.append(mini_batch[i][2])
                next_states[i] = mini_batch[i][3]
                dones.append(mini_batch[i][4])

            #Predicciones del modelo
            target = agent.model.predict(states, verbose = 0)
            target_val = agent.target_model.predict(next_states, verbose = 0)

            #Fiteo el modelo y lo guardo en q2
            for i in range(agent.batch_size):
                if dones[i]:
                    target[i][actions[i]] = rewards[i]
                else:
                    target[i][actions[i]] = rewards[i] + agent.discount_factor * (
                        np.amax(target_val[i]))

            agent.model.fit(states, target, batch_size=agent.batch_size,
                            epochs=1, verbose=0)
            model = agent.model.get_weights()
            q2.put(model)

            #Cada 10 frames soft-update del modelo target
            if (count % 10) == 0: #Cada cuánto update
                print('update target model')
                agent.target_model.set_weights(agent.model.get_weights())

######### ESTO ES LO NECESARIO PARA ARRANCAR LAS COLAS q1, q2, y q3 ###############
if __name__ == '__main__':
    memory = Queue()
    model = Queue()
    end = Queue()
    process_one = Process(target=actor, args=(memory, model, end))
    process_two = Process(target=learner, args=(memory, model, end))
    process_one.start()
    process_two.start()

    memory.close()
    model.close()
    end.close()
    memory.join_thread()
    model.join_thread()
    end.close()

    process_one.join()
    process_two.join()