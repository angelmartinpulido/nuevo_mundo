from src.core.system import AdaptiveSystem
from src.utils.config import load_config


def main():
    """
    Punto de entrada principal del sistema.
    """
    # Cargar configuración
    config = load_config()

    # Inicializar sistema
    system = AdaptiveSystem(config)

    # Ejecutar sistema
    system.run()


if __name__ == "__main__":
    main()
