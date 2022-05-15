import sys

from PyQt5.QtWidgets import QMainWindow
from PyQt5.uic import loadUi

from Tool_Model.GlobalVariables import mainWindowUI


class MainWindowClass(QMainWindow):
    def __init__(self):
        try:
            super(MainWindowClass, self).__init__()
            loadUi(mainWindowUI, self)
            self.btnClose.clicked.connect(lambda: self.ExitProgram())

        except Exception as e:
            print(e)

    def ExitProgram(self):
        sys.exit(0)
