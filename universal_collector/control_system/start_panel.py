"""
Iniciar Panel de Control
"""

from .control_interface import ControlPanel


def main():
    # Crear y mostrar panel
    panel = ControlPanel()
    panel.mainloop()


if __name__ == "__main__":
    main()
