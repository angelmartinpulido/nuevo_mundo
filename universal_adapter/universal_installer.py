"""
Instalador Universal para Cualquier Sistema
"""

import os
import sys
import platform
import subprocess
import shutil
import json
import logging
import requests
import zipfile
import tarfile
import tempfile
from typing import Dict, List, Optional, Any
import asyncio
import aiohttp
import aiofiles
import pkg_resources


class UniversalInstaller:
    """Instalador universal para cualquier sistema"""

    def __init__(self, package_name: str = "p2p_module", version: str = "latest"):
        self.package_name = package_name
        self.version = version
        self.system_info = self._get_system_info()
        self.installation_paths = self._get_installation_paths()
        self.dependencies = self._get_core_dependencies()
        self.optional_dependencies = self._get_optional_dependencies()

    def _get_system_info(self) -> Dict[str, str]:
        """Obtener información del sistema"""
        return {
            "os": platform.system().lower(),
            "os_version": platform.version(),
            "architecture": platform.machine(),
            "python_version": platform.python_version(),
        }

    def _get_installation_paths(self) -> Dict[str, str]:
        """Obtener rutas de instalación según sistema"""
        paths = {
            "linux": {
                "system": "/opt/p2p_module",
                "user": os.path.expanduser("~/.local/share/p2p_module"),
                "venv": "/opt/p2p_module/venv",
            },
            "windows": {
                "system": "C:\\Program Files\\p2p_module",
                "user": os.path.expanduser("~\\AppData\\Local\\p2p_module"),
                "venv": "C:\\Program Files\\p2p_module\\venv",
            },
            "darwin": {
                "system": "/Applications/p2p_module",
                "user": os.path.expanduser("~/Library/Application Support/p2p_module"),
                "venv": "/Applications/p2p_module/venv",
            },
            "android": {
                "system": "/data/data/p2p_module",
                "user": "/sdcard/p2p_module",
                "venv": "/data/data/p2p_module/venv",
            },
            "ios": {
                "system": "/private/var/p2p_module",
                "user": os.path.expanduser("~/Documents/p2p_module"),
                "venv": "/private/var/p2p_module/venv",
            },
        }

        return paths.get(self.system_info["os"], paths["linux"])

    def _get_core_dependencies(self) -> List[str]:
        """Obtener dependencias principales"""
        return [
            "torch",
            "numpy",
            "asyncio",
            "psutil",
            "cpuinfo",
            "aiohttp",
            "docker",
            "requests",
            "aiofiles",
        ]

    def _get_optional_dependencies(self) -> Dict[str, List[str]]:
        """Obtener dependencias opcionales por sistema"""
        return {
            "linux": ["nvidia-cuda-toolkit", "rocm-runtime", "opencl-headers"],
            "windows": ["cuda-toolkit", "directx-sdk"],
            "darwin": ["metal-framework", "mps-support"],
            "android": ["termux-api", "android-ndk"],
            "ios": ["metal-performance-shaders"],
        }

    async def download_package(self, download_url: Optional[str] = None) -> str:
        """Descargar paquete de instalación"""
        if not download_url:
            download_url = await self._get_download_url()

        async with aiohttp.ClientSession() as session:
            async with session.get(download_url) as response:
                if response.status != 200:
                    raise Exception(f"Error descargando paquete: {response.status}")

                # Crear directorio temporal
                temp_dir = tempfile.mkdtemp()
                file_path = os.path.join(temp_dir, f"{self.package_name}.zip")

                # Guardar archivo
                async with aiofiles.open(file_path, "wb") as f:
                    await f.write(await response.read())

                return file_path

    async def _get_download_url(self) -> str:
        """Obtener URL de descarga según sistema"""
        base_url = "https://github.com/tu_organizacion/p2p_module/releases"

        # Construir URL específica
        url = f"{base_url}/{self.version}/{self.system_info['os']}-{self.system_info['architecture']}.zip"

        return url

    async def install_dependencies(self):
        """Instalar dependencias"""
        # Dependencias principales
        await self._install_core_dependencies()

        # Dependencias opcionales
        await self._install_optional_dependencies()

    async def _install_core_dependencies(self):
        """Instalar dependencias principales"""
        for dep in self.dependencies:
            try:
                await self._pip_install(dep)
            except Exception as e:
                logging.warning(f"No se pudo instalar {dep}: {e}")

    async def _install_optional_dependencies(self):
        """Instalar dependencias opcionales"""
        optional_deps = self.optional_dependencies.get(self.system_info["os"], [])

        for dep in optional_deps:
            try:
                await self._system_package_install(dep)
            except Exception as e:
                logging.warning(f"No se pudo instalar {dep}: {e}")

    async def _pip_install(self, package: str):
        """Instalar paquete con pip"""
        cmd = [sys.executable, "-m", "pip", "install", "-U", package]
        process = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )

        await process.communicate()

    async def _system_package_install(self, package: str):
        """Instalar paquete con gestor de paquetes del sistema"""
        install_commands = {
            "linux": f"sudo apt-get install -y {package}",
            "windows": f"winget install {package}",
            "darwin": f"brew install {package}",
            "android": f"pkg install {package}",
            "ios": "No hay gestor de paquetes directo",
        }

        cmd = install_commands.get(self.system_info["os"])

        if cmd:
            process = await asyncio.create_subprocess_shell(
                cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )

            await process.communicate()

    async def extract_package(self, package_path: str) -> str:
        """Extraer paquete de instalación"""
        # Crear directorio de instalación
        install_dir = self.installation_paths["system"]
        os.makedirs(install_dir, exist_ok=True)

        # Extraer según tipo de archivo
        if package_path.endswith(".zip"):
            with zipfile.ZipFile(package_path, "r") as zip_ref:
                zip_ref.extractall(install_dir)
        elif package_path.endswith((".tar.gz", ".tgz")):
            with tarfile.open(package_path, "r:gz") as tar_ref:
                tar_ref.extractall(install_dir)

        return install_dir

    async def create_virtual_environment(self) -> str:
        """Crear entorno virtual"""
        venv_path = self.installation_paths["venv"]

        # Crear entorno virtual
        cmd = [sys.executable, "-m", "venv", venv_path]

        process = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )

        await process.communicate()

        return venv_path

    async def configure_system(self):
        """Configurar sistema para el paquete"""
        # Añadir al PATH
        await self._update_path()

        # Configurar permisos
        await self._set_permissions()

        # Crear configuración de usuario
        await self._create_user_config()

    async def _update_path(self):
        """Actualizar PATH del sistema"""
        install_dir = self.installation_paths["system"]

        # Añadir al PATH según sistema
        if self.system_info["os"] == "linux":
            await self._update_linux_path(install_dir)
        elif self.system_info["os"] == "windows":
            await self._update_windows_path(install_dir)
        elif self.system_info["os"] == "darwin":
            await self._update_macos_path(install_dir)

    async def _update_linux_path(self, install_dir: str):
        """Actualizar PATH en Linux"""
        profile_files = [
            os.path.expanduser("~/.bashrc"),
            os.path.expanduser("~/.bash_profile"),
            os.path.expanduser("~/.zshrc"),
        ]

        path_line = f"export PATH=$PATH:{install_dir}"

        for profile in profile_files:
            if os.path.exists(profile):
                async with aiofiles.open(profile, "a") as f:
                    await f.write(f"\n{path_line}\n")

    async def _update_windows_path(self, install_dir: str):
        """Actualizar PATH en Windows"""
        # Usar PowerShell para modificar PATH
        cmd = [
            "powershell",
            "-Command",
            f'[Environment]::SetEnvironmentVariable("PATH", $env:PATH + ";{install_dir}", "User")',
        ]

        process = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )

        await process.communicate()

    async def _update_macos_path(self, install_dir: str):
        """Actualizar PATH en macOS"""
        profile_files = [
            os.path.expanduser("~/.bash_profile"),
            os.path.expanduser("~/.zshrc"),
        ]

        path_line = f"export PATH=$PATH:{install_dir}"

        for profile in profile_files:
            if os.path.exists(profile):
                async with aiofiles.open(profile, "a") as f:
                    await f.write(f"\n{path_line}\n")

    async def _set_permissions(self):
        """Establecer permisos de instalación"""
        install_dir = self.installation_paths["system"]

        # Cambiar permisos según sistema
        if self.system_info["os"] in ["linux", "darwin", "android", "ios"]:
            cmd = ["chmod", "-R", "755", install_dir]

            process = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )

            await process.communicate()

    async def _create_user_config(self):
        """Crear configuración de usuario"""
        config_dir = self.installation_paths["user"]
        os.makedirs(config_dir, exist_ok=True)

        config_path = os.path.join(config_dir, "config.json")

        config = {
            "system_info": self.system_info,
            "installation_path": self.installation_paths["system"],
            "version": self.version,
            "dependencies": self.dependencies,
        }

        async with aiofiles.open(config_path, "w") as f:
            await f.write(json.dumps(config, indent=4))

    async def install(self) -> bool:
        """Proceso completo de instalación"""
        try:
            # Descargar paquete
            package_path = await self.download_package()

            # Instalar dependencias
            await self.install_dependencies()

            # Extraer paquete
            install_dir = await self.extract_package(package_path)

            # Crear entorno virtual
            venv_path = await self.create_virtual_environment()

            # Configurar sistema
            await self.configure_system()

            # Limpiar descarga
            os.remove(package_path)

            return True

        except Exception as e:
            logging.error(f"Error en instalación: {e}")
            return False


# Ejemplo de uso
async def main():
    installer = UniversalInstaller()
    success = await installer.install()

    if success:
        print("Instalación completada con éxito")
    else:
        print("Error en la instalación")


if __name__ == "__main__":
    asyncio.run(main())
