"""
Servicio de Ubigeo
Gestiona el mapeo entre Departamento+Provincia y códigos de Ubigeo
"""

import pandas as pd
import os
from typing import Dict, List, Optional


class UbigeoService:
    """Servicio para gestionar mapeos de ubigeo"""

    def __init__(self, ubigeo_file_path: str):
        """
        Initialize the ubigeo service

        Parameters:
        -----------
        ubigeo_file_path : str
            Path to the TB_UBIGEOS.csv file
        """
        self.ubigeo_file_path = ubigeo_file_path
        self.df_ubigeos = None
        self.load_ubigeos()

    def load_ubigeos(self):
        """Cargar el archivo CSV de ubigeos"""
        try:
            # Probar diferentes codificaciones
            encodings = ['iso-8859-1', 'latin-1', 'cp1252', 'utf-8']

            for encoding in encodings:
                try:
                    # Leer el archivo línea por línea para manejar el formato especial
                    # Cada línea viene entre comillas: "campo1\tcampo2\tcampo3"
                    lines = []
                    with open(self.ubigeo_file_path, 'r', encoding=encoding) as f:
                        for line in f:
                            # Eliminar comillas y salto de línea final
                            line = line.strip().strip('"')
                            # Separar por tabulación
                            fields = line.split('\t')
                            # Tomar solo los primeros 4 campos (ubigeo, departamento, provincia, distrito)
                            if len(fields) >= 4:
                                lines.append(fields[:4])

                    # Crear DataFrame
                    if len(lines) > 1:
                        self.df_ubigeos = pd.DataFrame(
                            lines[1:],  # Omitir el encabezado
                            columns=['ubigeo_reniec', 'departamento', 'provincia', 'distrito']
                        )
                        print(f"✅ Ubigeos cargados con codificación: {encoding}")
                        break
                except (UnicodeDecodeError, Exception) as e:
                    continue

            if self.df_ubigeos is None:
                raise Exception("No se pudo leer el archivo con ninguna codificación")

            # Limpiar datos (eliminar espacios extra y convertir a mayúsculas)
            self.df_ubigeos['ubigeo_reniec'] = self.df_ubigeos['ubigeo_reniec'].str.strip()
            self.df_ubigeos['departamento'] = self.df_ubigeos['departamento'].str.strip().str.upper()
            self.df_ubigeos['provincia'] = self.df_ubigeos['provincia'].str.strip().str.upper()
            self.df_ubigeos['distrito'] = self.df_ubigeos['distrito'].str.strip().str.upper()

            # Filtrar filas con ubigeo vacío
            self.df_ubigeos = self.df_ubigeos[self.df_ubigeos['ubigeo_reniec'] != '']

            # Convertir ubigeo a entero
            self.df_ubigeos['ubigeo_reniec'] = self.df_ubigeos['ubigeo_reniec'].astype(int)

            print(f"✅ Ubigeos cargados: {len(self.df_ubigeos)} registros")

        except Exception as e:
            print(f"❌ Error cargando ubigeos: {e}")
            raise

    def get_provincias_by_departamento(self, departamento: str) -> List[str]:
        """
        Get list of provinces for a given department

        Parameters:
        -----------
        departamento : str
            Department name

        Returns:
        --------
        List[str] : List of province names
        """
        departamento = departamento.strip().upper()

        provincias = self.df_ubigeos[
            self.df_ubigeos['departamento'] == departamento
        ]['provincia'].unique().tolist()

        return sorted(provincias)

    def get_ubigeo_by_dept_prov(self, departamento: str, provincia: str) -> Optional[int]:
        """
        Get ubigeo code for a given department and province
        Returns the first district's ubigeo for that province (usually the capital)

        Parameters:
        -----------
        departamento : str
            Department name
        provincia : str
            Province name

        Returns:
        --------
        Optional[int] : Ubigeo code or None if not found
        """
        departamento = departamento.strip().upper()
        provincia = provincia.strip().upper()

        # Filtrar por departamento y provincia
        matches = self.df_ubigeos[
            (self.df_ubigeos['departamento'] == departamento) &
            (self.df_ubigeos['provincia'] == provincia)
        ]

        if len(matches) == 0:
            return None

        # Devolver el primer ubigeo (usualmente el distrito capital)
        # El formato de ubigeo es DDPPDD donde:
        # - DD = código de departamento
        # - PP = código de provincia
        # - DD = código de distrito
        # Buscamos el que termina en '01' (capital) si está disponible

        # Intentar encontrar el distrito capital (termina en 01)
        capital = matches[matches['ubigeo_reniec'] % 100 == 1]
        if len(capital) > 0:
            return int(capital.iloc[0]['ubigeo_reniec'])

        # En caso contrario, devolver el primero
        return int(matches.iloc[0]['ubigeo_reniec'])

    def get_all_departamentos(self) -> List[str]:
        """
        Get list of all departments

        Returns:
        --------
        List[str] : List of department names
        """
        return sorted(self.df_ubigeos['departamento'].unique().tolist())

    def validate_departamento_provincia(self, departamento: str, provincia: str) -> bool:
        """
        Validate if a department-province combination exists

        Parameters:
        -----------
        departamento : str
            Department name
        provincia : str
            Province name

        Returns:
        --------
        bool : True if valid combination, False otherwise
        """
        ubigeo = self.get_ubigeo_by_dept_prov(departamento, provincia)
        return ubigeo is not None

    def get_location_info(self, ubigeo: int) -> Optional[Dict]:
        """
        Get location information for a given ubigeo

        Parameters:
        -----------
        ubigeo : int
            Ubigeo code

        Returns:
        --------
        Optional[Dict] : Location information or None if not found
        """
        match = self.df_ubigeos[self.df_ubigeos['ubigeo_reniec'] == ubigeo]

        if len(match) == 0:
            return None

        row = match.iloc[0]
        return {
            'ubigeo': int(row['ubigeo_reniec']),
            'departamento': row['departamento'],
            'provincia': row['provincia'],
            'distrito': row['distrito']
        }


# Instancia singleton
_ubigeo_service = None


def get_ubigeo_service(ubigeo_file_path: str = None) -> UbigeoService:
    """
    Get or create the ubigeo service singleton

    Parameters:
    -----------
    ubigeo_file_path : str, optional
        Path to the ubigeos file (only needed on first call)

    Returns:
    --------
    UbigeoService : The ubigeo service instance
    """
    global _ubigeo_service

    if _ubigeo_service is None:
        if ubigeo_file_path is None:
            # Ruta por defecto - buscar en múltiples ubicaciones
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

            # Intentar diferentes rutas posibles
            possible_paths = [
                os.path.join(base_dir, 'data', 'TB_UBIGEOS.csv'),  # Desarrollo local
                os.path.join('/app', 'data', 'TB_UBIGEOS.csv'),    # Render/Docker
                os.path.join(os.getcwd(), 'data', 'TB_UBIGEOS.csv'),  # Current working directory
            ]

            ubigeo_file_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    ubigeo_file_path = path
                    print(f"✅ Archivo de ubigeos encontrado en: {path}")
                    break

            if ubigeo_file_path is None:
                raise FileNotFoundError(
                    f"No se encontró el archivo TB_UBIGEOS.csv en ninguna de las rutas: {possible_paths}"
                )

        _ubigeo_service = UbigeoService(ubigeo_file_path)

    return _ubigeo_service
