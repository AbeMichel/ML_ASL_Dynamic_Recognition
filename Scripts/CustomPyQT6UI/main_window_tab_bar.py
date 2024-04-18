from PyQt6.QtWidgets import QTabBar, QWidget


class MainWindowTabBar(QTabBar):
    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self.parent_widget = parent
        self.widgets: dict[int, QWidget] = {}  # where we store widgets that will be associated with tabs
        self.current_tab: int = 0

        self.init_ui()

    def init_ui(self):
        self.currentChanged.connect(self.on_tab_changed)

    def add_tab(self, widget: QWidget, title: str):
        widget.hide()
        num_existing_tabs = self.count()
        self.widgets[num_existing_tabs] = widget
        self.addTab(title)
        self.parent_widget.layout().addWidget(widget)

    def on_tab_changed(self, index):
        self.widgets[self.current_tab].hide()
        self.widgets[index].show()
        self.current_tab = index
