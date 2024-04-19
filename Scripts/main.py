import sys

from PyQt6.QtWidgets import QWidget, QApplication, QVBoxLayout
from PyQt6.QtCore import Qt

from CustomPyQT6UI.main_window_tab_bar import MainWindowTabBar
from CustomPyQT6UI.data_gather_gui import DataGatherApp
from CustomPyQT6UI.model_building_gui import ModelBuildApp
from CustomPyQT6UI.prediction_gui import PredictionApp
from CustomPyQT6UI.main_window_read_me import ReadMeWindow


class MainApp(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.show()

    def init_ui(self):
        self.setWindowTitle("ASL Recognition")
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.WindowStaysOnTopHint)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowType.WindowStaysOnTopHint)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.resize(640, 640)

        # create the needed UI elements
        tabs = MainWindowTabBar(self)
        readme = ReadMeWindow()
        data_gather = DataGatherApp()
        build_model = ModelBuildApp()
        prediction = PredictionApp()

        # create layouts
        main_layout = QVBoxLayout()

        # set main layout
        self.setLayout(main_layout)

        # add widgets to layouts
        main_layout.addWidget(tabs, stretch=1)
        # main_layout.addWidget(training, stretch=1)
        tabs.add_tab(readme, "Home")
        tabs.add_tab(data_gather, "Create Data")
        tabs.add_tab(build_model, "Build Model")
        tabs.add_tab(prediction, "Predict")

        # create layout hierarchy


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_win = MainApp()
    sys.exit(app.exec())
