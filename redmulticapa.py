import matplotlib.pyplot as plt
import numpy as np
from sklearn.neural_network import MLPClassifier
from matplotlib.widgets import Button

# Datos y etiquetas iniciales
X = np.array([]).reshape(0, 2)
y = np.array([])
training = False  # Bandera para controlar cuando comenzar a entrenar

# Configuración de la red neuronal
clf = MLPClassifier(
    hidden_layer_sizes=(10,),
    max_iter=5000,  
    random_state=0,
    learning_rate='adaptive',  
    tol=1e-4  # Añadido criterio de tolerancia
)

# Configuración inicial del gráfico
fig, ax = plt.subplots(figsize=(8, 6))
plt.subplots_adjust(bottom=0.2)
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_title("Red Neuronal Multicapa")

# Crear botón
ax_button = plt.axes([0.4, 0.05, 0.2, 0.075])
btn_train = Button(ax_button, 'Comenzar')

# Función para graficar el plano de clasificación
def plot_decision_boundary():
    if len(X) >= 2:  # Necesitamos al menos 2 puntos
        # Crear una cuadrícula de puntos
        x_min, x_max = -1.5, 1.5
        y_min, y_max = -1.5, 1.5
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), 
                           np.linspace(y_min, y_max, 100))
        
        # Predicción
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        # Dibujar regiones con colores invertidos (rojo para clase 0, azul para clase 1)
        ax.contourf(xx, yy, Z, alpha=0.2, cmap=plt.cm.RdBu)
        fig.canvas.draw()

# Función para manejar clics en el gráfico
def onclick(event):
    if event.inaxes != ax or training:
        return
    
    if event.button == 1:  # Botón izquierdo - Rojo
        label = 0
        color = 'red'
    elif event.button == 3:  # Botón derecho - Azul
        label = 1
        color = 'blue'
    else:
        return
    
    # Añadir punto
    global X, y
    new_point = np.array([[event.xdata, event.ydata]])
    X = np.vstack([X, new_point]) if len(X) > 0 else new_point
    y = np.append(y, label)
    
    # Dibujar punto
    ax.scatter(event.xdata, event.ydata, color=color, s=50)
    fig.canvas.draw()

