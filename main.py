import sys
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QIcon
from ui_controller import MainWindow;

def main():
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon("alignment_icon.svg"))
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
