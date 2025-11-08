#!/usr/bin/env python3
"""
🔧 Gestor de Git LFS para modelos ML
Automatiza la configuración y verificación de Git LFS
"""

import os
import sys
import subprocess
from pathlib import Path
from typing import List, Tuple

class Colors:
    """Colores para terminal"""
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    RED = '\033[0;31m'
    BLUE = '\033[0;34m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_header(text: str):
    """Imprime un header destacado"""
    print(f"\n{Colors.BOLD}{'=' * 60}{Colors.END}")
    print(f"{Colors.BOLD}{text}{Colors.END}")
    print(f"{Colors.BOLD}{'=' * 60}{Colors.END}\n")

def print_success(text: str):
    """Imprime mensaje de éxito"""
    print(f"{Colors.GREEN}✅ {text}{Colors.END}")

def print_warning(text: str):
    """Imprime mensaje de advertencia"""
    print(f"{Colors.YELLOW}⚠️  {text}{Colors.END}")

def print_error(text: str):
    """Imprime mensaje de error"""
    print(f"{Colors.RED}❌ {text}{Colors.END}")

def print_info(text: str):
    """Imprime mensaje informativo"""
    print(f"{Colors.BLUE}ℹ️  {text}{Colors.END}")

def run_command(cmd: List[str], check: bool = True) -> Tuple[bool, str]:
    """Ejecuta un comando y retorna resultado"""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=check
        )
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        return False, e.stderr
    except FileNotFoundError:
        return False, f"Comando no encontrado: {cmd[0]}"

def check_git_lfs_installed() -> bool:
    """Verifica si Git LFS está instalado"""
    print_info("Verificando instalación de Git LFS...")
    success, _ = run_command(["git", "lfs", "version"], check=False)
    
    if success:
        print_success("Git LFS está instalado")
        return True
    else:
        print_error("Git LFS no está instalado")
        print("\n📦 Instrucciones de instalación:\n")
        print("🐧 Linux (Ubuntu/Debian):")
        print("   sudo apt-get install git-lfs\n")
        print("🍎 macOS:")
        print("   brew install git-lfs\n")
        print("🪟 Windows:")
        print("   Descarga desde: https://git-lfs.github.com/\n")
        return False

def check_git_repo() -> bool:
    """Verifica si estamos en un repositorio Git"""
    print_info("Verificando repositorio Git...")
    success, _ = run_command(["git", "rev-parse", "--git-dir"], check=False)
    
    if success:
        print_success("Repositorio Git detectado")
        return True
    else:
        print_error("No estás en un repositorio Git")
        print("\n💡 Ejecuta: git init\n")
        return False

def find_model_files() -> List[Path]:
    """Encuentra archivos de modelos en el proyecto"""
    print_info("Buscando archivos de modelos...")
    
    extensions = ['.pkl', '.h5', '.model', '.pth', '.onnx', '.joblib']
    model_files = []
    
    for ext in extensions:
        files = list(Path('.').rglob(f'*{ext}'))
        # Filtrar archivos en .git
        files = [f for f in files if '.git' not in str(f)]
        model_files.extend(files)
    
    if model_files:
        print_success(f"Se encontraron {len(model_files)} archivo(s):")
        for f in model_files:
            size = f.stat().st_size / (1024 * 1024)  # MB
            print(f"   📄 {f} ({size:.2f} MB)")
    else:
        print_warning("No se encontraron archivos de modelos")
    
    return model_files

def check_file_size(file: Path) -> bool:
    """Verifica si el archivo es apropiado para Git LFS"""
    size_mb = file.stat().st_size / (1024 * 1024)
    size_gb = size_mb / 1024
    
    if size_gb > 2:
        print_error(f"{file}: {size_gb:.2f} GB (excede límite de 2GB de GitHub LFS gratuito)")
        return False
    elif size_mb > 100:
        print_warning(f"{file}: {size_mb:.2f} MB (grande, pero ok para LFS)")
        return True
    else:
        print_info(f"{file}: {size_mb:.2f} MB (podría ir sin LFS, pero LFS es mejor práctica)")
        return True

def init_git_lfs():
    """Inicializa Git LFS"""
    print_info("Inicializando Git LFS...")
    success, output = run_command(["git", "lfs", "install"])
    
    if success:
        print_success("Git LFS inicializado")
        return True
    else:
        print_error(f"Error al inicializar Git LFS: {output}")
        return False

def track_files(extensions: List[str]):
    """Configura tracking de archivos"""
    print_info("Configurando tracking de archivos...")
    
    for ext in extensions:
        pattern = f"*{ext}"
        success, output = run_command(["git", "lfs", "track", pattern])
        
        if success:
            print_success(f"Tracking configurado para: {pattern}")
        else:
            print_error(f"Error al configurar {pattern}: {output}")

def show_gitattributes():
    """Muestra el contenido de .gitattributes"""
    gitattributes_path = Path(".gitattributes")
    
    if gitattributes_path.exists():
        print_info("Contenido de .gitattributes:")
        print("\n" + "─" * 60)
        with open(gitattributes_path, 'r') as f:
            print(f.read())
        print("─" * 60 + "\n")
    else:
        print_warning(".gitattributes no encontrado")

def add_files_to_git(files: List[Path]):
    """Agrega archivos a Git"""
    print_info("Agregando archivos a Git...")
    
    # Agregar .gitattributes primero
    success, _ = run_command(["git", "add", ".gitattributes"])
    if success:
        print_success(".gitattributes agregado")
    
    # Agregar archivos de modelos
    for file in files:
        success, _ = run_command(["git", "add", str(file)])
        if success:
            print_success(f"Agregado: {file}")
        else:
            print_error(f"Error al agregar: {file}")

def show_lfs_files():
    """Muestra archivos trackeados por LFS"""
    print_info("Archivos trackeados por Git LFS:")
    success, output = run_command(["git", "lfs", "ls-files"], check=False)
    
    if success and output.strip():
        print("\n" + output)
    else:
        print_warning("No hay archivos trackeados por LFS aún (normal antes del commit)")

def create_commit(message: str = None):
    """Crea un commit"""
    if message is None:
        message = "Add ML models with Git LFS"
    
    print_info("Creando commit...")
    success, output = run_command(["git", "commit", "-m", message])
    
    if success:
        print_success("Commit creado exitosamente")
        return True
    else:
        if "nothing to commit" in output:
            print_warning("No hay cambios para commitear")
        else:
            print_error(f"Error al crear commit: {output}")
        return False

def show_status():
    """Muestra el estado de Git"""
    print_info("Estado actual de Git:")
    success, output = run_command(["git", "status", "--short"])
    if success:
        print("\n" + output)

def interactive_setup():
    """Setup interactivo paso a paso"""
    print_header("🚀 CONFIGURACIÓN DE GIT LFS PARA MODELOS ML")
    
    # 1. Verificar Git LFS
    if not check_git_lfs_installed():
        return False
    
    # 2. Verificar repo Git
    if not check_git_repo():
        return False
    
    # 3. Buscar archivos
    model_files = find_model_files()
    
    if not model_files:
        print("\n❓ ¿Dónde está tu modelo?")
        custom_path = input("Ingresa la ruta (o Enter para salir): ").strip()
        if custom_path:
            model_files = [Path(custom_path)]
        else:
            print_info("Saliendo...")
            return False
    
    # 4. Verificar tamaños
    print_header("📊 VERIFICACIÓN DE TAMAÑOS")
    valid = all(check_file_size(f) for f in model_files)
    
    if not valid:
        proceed = input("\n⚠️  Algunos archivos exceden límites. ¿Continuar? (s/n): ")
        if proceed.lower() != 's':
            return False
    
    # 5. Inicializar Git LFS
    print_header("🔧 CONFIGURACIÓN DE GIT LFS")
    if not init_git_lfs():
        return False
    
    # 6. Configurar tracking
    extensions = list(set([f.suffix for f in model_files]))
    track_files(extensions)
    
    # 7. Mostrar .gitattributes
    show_gitattributes()
    
    # 8. Agregar archivos
    print_header("➕ AGREGANDO ARCHIVOS")
    add_files_to_git(model_files)
    
    # 9. Mostrar status
    show_status()
    
    # 10. Mostrar LFS files
    show_lfs_files()
    
    # 11. Commit
    print_header("💾 CREAR COMMIT")
    commit_msg = input("Mensaje de commit (Enter para usar default): ").strip()
    if not commit_msg:
        commit_msg = "Add ML models with Git LFS"
    
    if not create_commit(commit_msg):
        return False
    
    # 12. Push
    print_header("🚀 PUSH A GITHUB")
    print("\n📋 Antes de hacer push, verifica:")
    print("   ✓ Remote configurado (git remote -v)")
    print("   ✓ Permisos de escritura")
    print("   ✓ Rama correcta\n")
    
    do_push = input("¿Hacer push ahora? (s/n): ")
    
    if do_push.lower() == 's':
        # Obtener rama actual
        _, branch = run_command(["git", "branch", "--show-current"])
        branch = branch.strip()
        print_info(f"Haciendo push a rama: {branch}")
        
        success, output = run_command(["git", "push", "origin", branch], check=False)
        
        if success:
            print_success("Push completado exitosamente!")
            print("\n" + "🎉" * 30)
            print("\n✅ ¡CONFIGURACIÓN COMPLETADA!\n")
            print("📋 Próximos pasos:")
            print("   1. Verifica en GitHub que los archivos .pkl tienen ícono LFS")
            print("   2. Ahora puedes deployar en:")
            print("      • Render.com")
            print("      • Railway.app")
            print("      • Fly.io")
            print("\n" + "🎉" * 30)
        else:
            print_error(f"Error en push: {output}")
            print("\n💡 Intenta manualmente: git push origin " + branch)
    else:
        print_info("Push cancelado")
        print(f"\n💡 Para hacer push más tarde: git push origin {branch}")
    
    return True

def main():
    """Función principal"""
    try:
        success = interactive_setup()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⏸️  Proceso cancelado por el usuario")
        sys.exit(1)
    except Exception as e:
        print_error(f"Error inesperado: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
