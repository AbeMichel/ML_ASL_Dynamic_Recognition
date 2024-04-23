import sys
from PyQt6.QtWidgets import QApplication, QWidget, QStackedWidget, QVBoxLayout, QListWidget, QListWidgetItem, QComboBox, QSpinBox, \
    QLabel, QPushButton, QHBoxLayout, QDialog, QDoubleSpinBox
import tensorflow.keras as keras
from tensorflow.keras import layers


ACTIVATIONS = {
    "None": None,
    "ELU": "elu",
    "Exponential": "exponential",
    "GELU": "gelu",
    "Hard Sigmoid": "hard_sigmoid",
    "Linear": "linear",
    "ReLU": "relu",
    "SELU": "selu",
    "Sigmoid": "sigmoid",
    "Softmax": "softmax",
    "Softplus": "softplus",
    "Softsign": "softsign",
    "Swish": "swish",
    "Tanh": "tanh"
}


class LayerWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.main_layout = QHBoxLayout()
        self.setLayout(self.main_layout)

    def get_layer(self):
        return None


class ConvolutionLayerWidget(LayerWidget):
    def __init__(self):
        super().__init__()

        self.filters_spinbox = QSpinBox()
        filters_labels = QLabel("Filters")

        self.kernal_spinbox = QSpinBox()
        kernal_labels = QLabel("Kernal Size")

        self.padding_combobox = QComboBox()
        padding_labels = QLabel("Padding")

        self.activation_box = QComboBox()
        activation_labels = QLabel("Activation Function")

        self.main_layout.addWidget(filters_labels)
        self.main_layout.addWidget(self.filters_spinbox)
        self.main_layout.addWidget(kernal_labels)
        self.main_layout.addWidget(self.kernal_spinbox)
        self.main_layout.addWidget(padding_labels)
        self.main_layout.addWidget(self.padding_combobox)
        self.main_layout.addWidget(activation_labels)
        self.main_layout.addWidget(self.activation_box)

        self.filters_spinbox.setRange(1, 2147483647)
        self.filters_spinbox.setValue(32)

        self.kernal_spinbox.setRange(1, 10)
        self.kernal_spinbox.setValue(3)

        for padding_type in ['valid', 'same', 'causal']:
            self.padding_combobox.addItem(padding_type)

        for activation_function in ACTIVATIONS.keys():
            self.activation_box.addItem(activation_function)

    def get_layer(self):
        filters = self.filters_spinbox.value()
        kernals = self.kernal_spinbox.value()
        padding_type = self.padding_combobox.currentText()
        activation = ACTIVATIONS[self.activation_box.currentText()]
        return layers.Conv1D(filters, kernals, padding=padding_type, activation=activation)


class MaxPoolingLayerWidget(LayerWidget):
    def __init__(self):
        super().__init__()

        self.pool_spinbox = QSpinBox()
        pool_label = QLabel("Pool Size")

        self.padding_combobox = QComboBox()
        padding_labels = QLabel("Padding")

        self.main_layout.addWidget(pool_label)
        self.main_layout.addWidget(self.pool_spinbox)
        self.main_layout.addWidget(padding_labels)
        self.main_layout.addWidget(self.padding_combobox)

        self.pool_spinbox.setRange(1, 2147483647)
        self.pool_spinbox.setValue(2)

        for padding_type in ['valid', 'same', 'causal']:
            self.padding_combobox.addItem(padding_type)

    def get_layer(self):
        pool_size = self.pool_spinbox.value()
        padding_type = self.padding_combobox.currentText()
        return layers.MaxPooling1D(pool_size=pool_size, padding=padding_type)


class FlattenLayerWidget(LayerWidget):
    def __init__(self):
        super().__init__()

    def get_layer(self):
        return layers.Flatten()


class DenseLayerWidget(LayerWidget):
    def __init__(self):
        super().__init__()

        self.num_neurons_spinbox = QSpinBox()
        num_neurons_labels = QLabel("Number of Neurons")

        self.activation_combobox = QComboBox()
        activation_labels = QLabel("Activation Function")

        self.main_layout.addWidget(num_neurons_labels)
        self.main_layout.addWidget(self.num_neurons_spinbox)
        self.main_layout.addWidget(activation_labels)
        self.main_layout.addWidget(self.activation_combobox)

        self.num_neurons_spinbox.setRange(1, 2147483647)
        self.num_neurons_spinbox.setValue(32)

        for activation_function in ACTIVATIONS.keys():
            self.activation_combobox.addItem(activation_function)

    def get_layer(self):
        num_neurons = self.num_neurons_spinbox.value()
        activation_function = ACTIVATIONS.get(self.activation_combobox.currentText())
        return layers.Dense(num_neurons, activation=activation_function)


class DropoutLayerWidget(LayerWidget):
    def __init__(self):
        super().__init__()
        self.rate_spinbox = QDoubleSpinBox()
        self.rate_label = QLabel("Dropout Rate")

        self.rate_spinbox.setRange(0, 1)

        self.main_layout.addWidget(self.rate_label)
        self.main_layout.addWidget(self.rate_spinbox)

    def get_layer(self):
        layer = layers.Dropout(self.rate_spinbox.value())
        return layer


class CustomItemWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.comboBox = QComboBox()
        self.stacked_widget = QStackedWidget()
        self.create_layers()

        self.layout = QHBoxLayout()
        self.layout.addWidget(self.comboBox)
        self.layout.addWidget(self.stacked_widget)

        self.comboBox.currentTextChanged.connect(self.on_layer_type_changed)

        self.setLayout(self.layout)

    def create_layers(self):
        self.comboBox.addItem("")
        self.stacked_widget.addWidget(LayerWidget())
        self.comboBox.addItem("Max Pooling")
        self.stacked_widget.addWidget(MaxPoolingLayerWidget())
        self.comboBox.addItem("Dense")
        self.stacked_widget.addWidget(DenseLayerWidget())
        self.comboBox.addItem("Flatten")
        self.stacked_widget.addWidget(FlattenLayerWidget())
        self.comboBox.addItem("Dropout")
        self.stacked_widget.addWidget(DropoutLayerWidget())
        self.comboBox.addItem("Convolution 1D")
        self.stacked_widget.addWidget(ConvolutionLayerWidget())

    def on_layer_type_changed(self, e):
        index = self.comboBox.currentIndex()
        self.stacked_widget.setCurrentIndex(index)

    def get_layer(self):
        return self.stacked_widget.currentWidget().get_layer()


class ReorderableListWidget(QListWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setDragEnabled(True)
        self.setAcceptDrops(True)
        self.setDropIndicatorShown(True)
        self.setDragDropMode(self.DragDropMode.InternalMove)
        self.setSelectionMode(QListWidget.SelectionMode.SingleSelection)

    def addItem(self):
        item = QListWidgetItem(self)
        custom_widget = CustomItemWidget()
        item.setSizeHint(custom_widget.sizeHint())
        self.setItemWidget(item, custom_widget)

    def delete_selected_item(self):
        selected_index = self.currentIndex().row()
        item = self.takeItem(selected_index)
        if item is not None:
            del item

    def get_layers(self) -> list[layers]:
        layers_ = []
        for i in range(self.count()):
            item = self.item(i)
            custom_widget = self.itemWidget(item)
            curr_layer = custom_widget.get_layer()
            if curr_layer is not None:
                layers_.append(curr_layer)
        return layers_


class CreateLayersDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Build Your Machine Learning Model")
        self.resize(800, 600)

        self.layers = []
        layout = QVBoxLayout()
        first_layer_label = QLabel("First: Reshape((input_shape, 1), input_shape=(input_shape, 1))")
        layout.addWidget(first_layer_label)

        self.listWidget = ReorderableListWidget()
        layout.addWidget(self.listWidget)

        last_layer_label = QLabel("Last: Dense(len(class_names), activation='softmax')")
        layout.addWidget(last_layer_label)

        btn_layout = QHBoxLayout()

        add_button = QPushButton("Add Layer")
        add_button.clicked.connect(self.listWidget.addItem)
        btn_layout.addWidget(add_button)

        del_button = QPushButton("Delete Selected Layer")
        del_button.clicked.connect(self.listWidget.delete_selected_item)
        btn_layout.addWidget(del_button)

        layout.addLayout(btn_layout)

        build_button = QPushButton("Build")
        build_button.clicked.connect(self.build_model)
        layout.addWidget(build_button)

        self.setLayout(layout)

    def build_model(self):
        self.layers = self.listWidget.get_layers()
        self.accept()

    def get_result(self):
        if self.result() == QDialog.DialogCode.Accepted:
            return self.layers
        else:
            return None


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CreateLayersDialog()
    if window.exec() == QDialog.DialogCode.Accepted:
        print("YES")
