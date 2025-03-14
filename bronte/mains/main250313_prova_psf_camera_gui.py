import sys
import numpy as np
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg
import time  # Import necessario per gestire il tempo reale

class RealtimePlotter1D(QtWidgets.QMainWindow):
    """Finestra per il grafico 1D in tempo reale."""
    
    def __init__(self, title="Plot 1D in Tempo Reale", update_interval=100):
        super().__init__()
        self.setWindowTitle(title)
        
        # Layout principale
        self.central_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QtWidgets.QVBoxLayout(self.central_widget)

        # Creazione del plot
        self.plot_widget = pg.PlotWidget(title="Segnale in tempo reale")
        self.layout.addWidget(self.plot_widget)
        self.curve = self.plot_widget.plot(pen="y")  # Linea gialla
        
        # Timer per aggiornare il plot
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(update_interval)

        # Dati iniziali
        self.x = np.linspace(0, 10, 100)
        self.y = np.sin(self.x)

        self.show()

    def update_plot(self):
        """Aggiorna il grafico 1D con nuovi dati."""
        self.y = np.roll(self.y, -1)  # Scorre il segnale
        self.y[-1] = np.sin(np.random.rand() * np.pi * 2)  # Nuovo valore casuale
        self.curve.setData(self.x, self.y)  # Aggiorna il grafico

    def closeEvent(self, event):
        """Ferma il timer quando la finestra viene chiusa."""
        self.timer.stop()
        event.accept()
        


class StrehlRatioPlotter(QtWidgets.QMainWindow):
    """Finestra per il grafico dello Strehl Ratio in tempo reale."""
    
    def __init__(self, title="Strehl Ratio in Tempo Reale", update_interval=100):
        super().__init__()
        self.setWindowTitle(title)

        # Layout principale
        self.central_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QtWidgets.QVBoxLayout(self.central_widget)

        # Creazione del plot
        self.plot_widget = pg.PlotWidget(title="Strehl Ratio")
        self.layout.addWidget(self.plot_widget)
        self.curve = self.plot_widget.plot(pen="b")  # Linea blu

        # Configura asse X come tempo relativo
        self.plot_widget.setLabel("bottom", "Tempo", units="s")
        self.plot_widget.setLabel("left", "Strehl Ratio")

        # Timer per aggiornare il plot
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(update_interval)

        # Dati iniziali
        self.start_time = time.time()  # Tempo iniziale
        self.times = [0]  # Lista dei tempi relativi
        self.values = [self.get_strehl_ratio()]  # Lista dei valori Strehl Ratio

        self.show()

    def get_strehl_ratio(self):
        """Restituisce un valore scalare simulato di Strehl Ratio."""
        return np.random.rand()  # Genera un valore casuale tra 0 e 1

    def update_plot(self):
        """Aggiorna il grafico con il nuovo valore e aggiorna l'asse temporale."""
        current_time = time.time() - self.start_time  # Tempo relativo
        new_value = self.get_strehl_ratio()

        self.times.append(current_time)
        self.values.append(new_value)

        # Mantiene solo gli ultimi 100 valori per evitare troppi dati
        if len(self.times) > 100:
            self.times.pop(0)
            self.values.pop(0)

        self.curve.setData(self.times, self.values)  # Aggiorna il grafico

    def closeEvent(self, event):
        """Ferma il timer quando la finestra viene chiusa."""
        self.timer.stop()
        event.accept()


class RealtimePlotter(QtWidgets.QMainWindow):
    
    def __init__(self, title="Plot 2D in Tempo Reale", update_interval=100):
        super().__init__()

        # Creazione dell'applicazione Qt
        self.app = QtWidgets.QApplication.instance()
        if self.app is None:
            self.app = QtWidgets.QApplication([])

        # Imposta la finestra principale
        self.setWindowTitle(title)
        self.central_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QtWidgets.QVBoxLayout(self.central_widget)

        # Widget PyQtGraph
        self.graphics_widget = pg.GraphicsLayoutWidget()
        self.layout.addWidget(self.graphics_widget)

        # Creazione del plot
        self.plot_item = self.graphics_widget.addPlot(title="PSF CAMERA")
        self.image_item = pg.ImageItem()
        self.plot_item.addItem(self.image_item)

        # Abilita zoom e pan con il mouse
        self.plot_item.setMouseEnabled(True, True)
        self.plot_item.setAspectLocked(True)

        # Creazione della colorbar
        self.colorbar = pg.HistogramLUTItem()
        self.colorbar.setImageItem(self.image_item)
        self.graphics_widget.addItem(self.colorbar)

        # Lista di colormap disponibili
        self.colormaps = ["viridis", "plasma", "inferno", "magma", "cividis"]
        self.current_colormap_index = 0  # Indice della colormap attuale

        # Imposta la colormap iniziale
        self.set_colormap(self.colormaps[self.current_colormap_index])

        # Timer per aggiornare il plot
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(update_interval)  # Aggiorna ogni 'update_interval' ms
        
        # Bottone per aprire il grafico 1D
        self.button = QtWidgets.QPushButton("Apri Grafico 1D")
        self.button.clicked.connect(self.open_1d_plot)
        self.layout.addWidget(self.button)
        
        # Aggiunto secondo bottone per aprire il grafico dello Strehl Ratio
        self.button_strehl = QtWidgets.QPushButton("Apri Strehl Ratio")
        self.button_strehl.clicked.connect(self.open_strehl_plot)
        self.layout.addWidget(self.button_strehl)

        self.plot_strehl_window = None  # Finestra del grafico Strehl Ratio


        self.plot_1d_window = None  # Finestra del grafico 1D
        
        # Mostra la finestra
        self.show()

    def set_colormap(self, colormap_name):
        """Cambia la colormap del plot"""
        colormap = pg.colormap.get(colormap_name)
        self.colorbar.gradient.setColorMap(colormap)

    def update_plot(self):
        """Aggiorna il plot con un nuovo array 2D"""
        ima = np.random.rand(200, 300)  # Genera un'immagine casuale
        self.image_item.setImage(ima.T, autoLevels=False)
        self.colorbar.setLevels(np.min(ima), np.max(ima))
        QtWidgets.QApplication.processEvents()  # Mantiene la UI reattiva
        
    def open_1d_plot(self):
        """Apre una nuova finestra con il grafico 1D."""
        if self.plot_1d_window is None or not self.plot_1d_window.isVisible():
            self.plot_1d_window = RealtimePlotter1D()
            self.plot_1d_window.show()
    
    def open_strehl_plot(self):
        """Apre una nuova finestra con il grafico dello Strehl Ratio."""
        if self.plot_strehl_window is None or not self.plot_strehl_window.isVisible():
            self.plot_strehl_window = StrehlRatioPlotter()
            self.plot_strehl_window.show()

    def keyPressEvent(self, event):
        """Cambia la colormap premendo il tasto 'C'"""
        if event.key() == QtCore.Qt.Key_C:
            self.current_colormap_index = (self.current_colormap_index + 1) % len(self.colormaps)
            new_colormap = self.colormaps[self.current_colormap_index]
            self.set_colormap(new_colormap)

    def start(self):
        """Avvia l'interfaccia grafica"""
        self.app.exec_()
    
    def closeEvent(self, event):
        """Chiude la finestra e ferma il timer"""
        self.timer.stop()
        event.accept()

# Avvia il plotter
if __name__ == "__main__":
    plotter = RealtimePlotter()
    plotter.start()
