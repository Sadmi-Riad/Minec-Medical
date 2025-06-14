import sys
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QIcon
from version0.main_window import MainApp
import resources_rc

if __name__ == "__main__":
    app = QApplication(sys.argv)
    icon = QIcon(":/minecLogo.ico")
    app.setWindowIcon(icon)
    window = MainApp()
    window.setWindowTitle("Minec")
    window.setWindowIcon(icon)
    window.setUnifiedTitleAndToolBarOnMac(False)
    window.setWindowFilePath("/tmp/fichier-factice.txt")
    window.show()
    sys.exit(app.exec_())