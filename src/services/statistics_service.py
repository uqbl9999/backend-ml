"""
Servicio de Estadísticas
Gestiona consultas estadísticas sobre tamizajes de salud mental
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import os


class StatisticsService:
    """Servicio para gestionar estadísticas de tamizajes"""

    def __init__(self, data_file_path: str):
        """
        Initialize the statistics service

        Parameters:
        -----------
        data_file_path : str
            Path to the tamizajes.csv file
        """
        self.data_file_path = data_file_path
        self.df = None
        self.load_data()

    def load_data(self):
        """Cargar el archivo CSV de tamizajes"""
        try:
            self.df = pd.read_csv(self.data_file_path, sep=';', encoding='latin1')

            # Convertir 'Casos' a numérico
            if self.df['Casos'].dtype == 'object':
                self.df['Casos'] = self.df['Casos'].astype(str).str.replace(',', '').str.replace(' ', '')
                self.df['Casos'] = pd.to_numeric(self.df['Casos'], errors='coerce')
                self.df['Casos'] = self.df['Casos'].fillna(0)

            # Calcular tasa de positividad
            self._calculate_positivity_rate()

            print(f"✅ Datos de estadísticas cargados: {self.df.shape[0]:,} filas")
        except Exception as e:
            print(f"❌ Error al cargar datos: {e}")
            raise

    def _calculate_positivity_rate(self):
        """Calcular la tasa de positividad para cada registro"""
        # Separar registros de TOTAL y POSITIVOS
        df_total = self.df[self.df['GrupoTamizaje'].str.contains('TOTAL', case=False, na=False)].copy()
        df_positivos = self.df[self.df['GrupoTamizaje'].str.contains('POSITIVOS', case=False, na=False)].copy()

        # Renombrar columnas
        df_total = df_total.rename(columns={'Casos': 'Total'})
        df_positivos = df_positivos.rename(columns={'Casos': 'Positivos'})

        # Eliminar columna GrupoTamizaje
        df_total = df_total.drop(columns=['GrupoTamizaje'])
        df_positivos = df_positivos.drop(columns=['GrupoTamizaje'])

        # Unir datasets
        join_columns = ['Anio', 'NroMes', 'ubigeo', 'Departamento', 'Provincia',
                       'Distrito', 'Sexo', 'Etapa', 'DetalleTamizaje']

        self.df_pivot = df_total.merge(
            df_positivos[join_columns + ['Positivos']],
            on=join_columns,
            how='left'
        )

        # Rellenar NaN y calcular tasa
        self.df_pivot['Positivos'] = self.df_pivot['Positivos'].fillna(0)
        self.df_pivot['Tasa_Positividad'] = np.where(
            self.df_pivot['Total'] > 0,
            (self.df_pivot['Positivos'] / self.df_pivot['Total']) * 100,
            0
        )

        # Limpiar tasas imposibles (> 100%)
        self.df_pivot = self.df_pivot[self.df_pivot['Tasa_Positividad'] <= 100].copy()

    def get_descriptive_statistics(self) -> Dict:
        """
        Obtener estadísticas descriptivas sobre la tasa de positividad

        Returns:
        --------
        Dict con media, mediana, desviación estándar y máximo
        """
        if self.df_pivot is None or len(self.df_pivot) == 0:
            return {
                'media': 0.0,
                'mediana': 0.0,
                'desviacion_estandar': 0.0,
                'maximo': 0.0
            }

        tasa = self.df_pivot['Tasa_Positividad']

        return {
            'media': round(float(tasa.mean()), 2),
            'mediana': round(float(tasa.median()), 2),
            'desviacion_estandar': round(float(tasa.std()), 2),
            'maximo': round(float(tasa.max()), 2)
        }

    def get_distribution_by_groups(self) -> Dict:
        """
        Obtener distribución por grupos de tamizaje

        Returns:
        --------
        Dict con información de distribución de registros y suma de casos
        """
        # Separar por grupos
        df_total = self.df[self.df['GrupoTamizaje'].str.contains('TOTAL', case=False, na=False)]
        df_positivos = self.df[self.df['GrupoTamizaje'].str.contains('POSITIVOS', case=False, na=False)]
        df_violencia = self.df[
            self.df['GrupoTamizaje'].str.contains('VIOLENCIA', case=False, na=False) |
            self.df['DetalleTamizaje'].str.contains('VIOLENCIA', case=False, na=False)
        ]

        return {
            'distribucion_registros': {
                'total_tamizajes': int(len(df_total)),
                'solo_tamizajes_positivos': int(len(df_positivos)),
                'tamizajes_con_violencia_politica': int(len(df_violencia))
            },
            'suma_total_casos': {
                'total_tamizajes': int(df_total['Casos'].sum()),
                'solo_tamizajes_positivos': int(df_positivos['Casos'].sum()),
                'tamizajes_con_violencia_politica': int(df_violencia['Casos'].sum())
            }
        }

    def get_heatmap_by_screening_type(self, grupo: str = None) -> List[Dict]:
        """
        Obtener heatmap por tipo de tamizaje y grupo

        Parameters:
        -----------
        grupo : str, optional
            Filtro por grupo ('TOTAL', 'POSITIVOS', o 'VIOLENCIA')
            Si es None, devuelve todos los grupos

        Returns:
        --------
        List de Dict con casos agregados por grupo de tamizaje y tipo específico
        """
        df_filtered = self.df.copy()

        # Aplicar filtro de grupo si se especifica
        if grupo:
            if grupo.upper() == 'TOTAL':
                df_filtered = df_filtered[df_filtered['GrupoTamizaje'].str.contains('TOTAL', case=False, na=False)]
            elif grupo.upper() == 'POSITIVOS':
                df_filtered = df_filtered[df_filtered['GrupoTamizaje'].str.contains('POSITIVOS', case=False, na=False)]
            elif grupo.upper() == 'VIOLENCIA':
                df_filtered = df_filtered[
                    df_filtered['GrupoTamizaje'].str.contains('VIOLENCIA', case=False, na=False) |
                    df_filtered['DetalleTamizaje'].str.contains('VIOLENCIA', case=False, na=False)
                ]

        # Agrupar por GrupoTamizaje y DetalleTamizaje
        grouped = df_filtered.groupby(['GrupoTamizaje', 'DetalleTamizaje'])['Casos'].sum().reset_index()

        # Pivotar para tener tipos de tamizaje como columnas
        pivot_table = grouped.pivot_table(
            index='GrupoTamizaje',
            columns='DetalleTamizaje',
            values='Casos',
            fill_value=0,
            aggfunc='sum'
        )

        # Convertir a lista de diccionarios
        result = []
        for grupo_tamizaje in pivot_table.index:
            row_data = {'grupo': grupo_tamizaje}
            for detalle_tamizaje in pivot_table.columns:
                # Normalizar nombres de columnas para JSON
                key = detalle_tamizaje.lower().replace(' ', '_').replace('/', '_')
                row_data[key] = int(pivot_table.loc[grupo_tamizaje, detalle_tamizaje])
            result.append(row_data)

        return result

    def get_heatmap_by_department(self, grupo: str = None, top_n: int = None) -> List[Dict]:
        """
        Obtener heatmap por departamento y grupo de tamizaje

        Parameters:
        -----------
        grupo : str, optional
            Filtro por grupo ('TOTAL', 'POSITIVOS', o 'VIOLENCIA')
            Si es None, devuelve todos los grupos
        top_n : int, optional
            Limitar a los top N departamentos con más casos

        Returns:
        --------
        List de Dict con casos agregados por departamento y grupo
        """
        df_filtered = self.df.copy()

        # Aplicar filtro de grupo si se especifica
        if grupo:
            if grupo.upper() == 'TOTAL':
                df_filtered = df_filtered[df_filtered['GrupoTamizaje'].str.contains('TOTAL', case=False, na=False)]
            elif grupo.upper() == 'POSITIVOS':
                df_filtered = df_filtered[df_filtered['GrupoTamizaje'].str.contains('POSITIVOS', case=False, na=False)]
            elif grupo.upper() == 'VIOLENCIA':
                df_filtered = df_filtered[
                    df_filtered['GrupoTamizaje'].str.contains('VIOLENCIA', case=False, na=False) |
                    df_filtered['DetalleTamizaje'].str.contains('VIOLENCIA', case=False, na=False)
                ]

        # Agrupar por Departamento y GrupoTamizaje
        grouped = df_filtered.groupby(['Departamento', 'GrupoTamizaje'])['Casos'].sum().reset_index()

        # Pivotar para tener grupos como columnas
        pivot_table = grouped.pivot_table(
            index='Departamento',
            columns='GrupoTamizaje',
            values='Casos',
            fill_value=0,
            aggfunc='sum'
        )

        # Calcular total por departamento para ordenar
        pivot_table['total'] = pivot_table.sum(axis=1)
        pivot_table = pivot_table.sort_values('total', ascending=False)

        # Limitar a top_n si se especifica
        if top_n:
            pivot_table = pivot_table.head(top_n)

        # Convertir a lista de diccionarios
        result = []
        for departamento in pivot_table.index:
            row_data = {'departamento': departamento}
            for grupo_tamizaje in pivot_table.columns:
                if grupo_tamizaje != 'total':
                    # Normalizar nombres de columnas para JSON
                    key = grupo_tamizaje.lower().replace(' ', '_').replace('/', '_')
                    row_data[key] = int(pivot_table.loc[departamento, grupo_tamizaje])
            row_data['total'] = int(pivot_table.loc[departamento, 'total'])
            result.append(row_data)

        return result

    def get_screening_types_summary(self) -> List[Dict]:
        """
        Obtener resumen de tipos de tamizaje con estadísticas

        Returns:
        --------
        List de Dict con estadísticas por tipo de tamizaje
        """
        if self.df_pivot is None:
            return []

        result = []
        for detalle in self.df_pivot['DetalleTamizaje'].unique():
            df_detalle = self.df_pivot[self.df_pivot['DetalleTamizaje'] == detalle]

            result.append({
                'detalle_tamizaje': detalle,
                'total_registros': int(len(df_detalle)),
                'suma_total_casos': int(df_detalle['Total'].sum()),
                'suma_positivos': int(df_detalle['Positivos'].sum()),
                'tasa_positividad_promedio': round(float(df_detalle['Tasa_Positividad'].mean()), 2),
                'tasa_positividad_mediana': round(float(df_detalle['Tasa_Positividad'].median()), 2),
                'tasa_positividad_max': round(float(df_detalle['Tasa_Positividad'].max()), 2)
            })

        return sorted(result, key=lambda x: x['suma_total_casos'], reverse=True)

    def get_department_summary(self, top_n: int = None) -> List[Dict]:
        """
        Obtener resumen de departamentos con estadísticas

        Parameters:
        -----------
        top_n : int, optional
            Limitar a los top N departamentos

        Returns:
        --------
        List de Dict con estadísticas por departamento
        """
        if self.df_pivot is None:
            return []

        result = []
        for departamento in self.df_pivot['Departamento'].unique():
            df_dept = self.df_pivot[self.df_pivot['Departamento'] == departamento]

            result.append({
                'departamento': departamento,
                'total_registros': int(len(df_dept)),
                'suma_total_casos': int(df_dept['Total'].sum()),
                'suma_positivos': int(df_dept['Positivos'].sum()),
                'tasa_positividad_promedio': round(float(df_dept['Tasa_Positividad'].mean()), 2),
                'tasa_positividad_mediana': round(float(df_dept['Tasa_Positividad'].median()), 2),
                'tasa_positividad_max': round(float(df_dept['Tasa_Positividad'].max()), 2)
            })

        # Ordenar por suma total de casos
        result = sorted(result, key=lambda x: x['suma_total_casos'], reverse=True)

        # Limitar a top_n si se especifica
        if top_n:
            result = result[:top_n]

        return result


# Singleton instance
_statistics_service = None


def get_statistics_service() -> Optional[StatisticsService]:
    """
    Get the singleton instance of StatisticsService

    Returns:
    --------
    StatisticsService instance or None if data file not found
    """
    global _statistics_service

    if _statistics_service is None:
        # Buscar el archivo de datos en múltiples ubicaciones
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

        # Intentar diferentes rutas posibles
        possible_paths = [
            os.path.join(base_dir, 'data', 'tamizajes.csv'),  # Desarrollo local
            os.path.join('/app', 'data', 'tamizajes.csv'),    # Render/Docker
            os.path.join(os.getcwd(), 'data', 'tamizajes.csv'),  # Current working directory
        ]

        data_path = None
        for path in possible_paths:
            if os.path.exists(path):
                data_path = path
                print(f"✅ Archivo de datos encontrado en: {path}")
                break

        if data_path is None:
            print(f"⚠️  Archivo de datos no encontrado en ninguna de las rutas: {possible_paths}")
            return None

        try:
            _statistics_service = StatisticsService(data_path)
        except Exception as e:
            print(f"❌ Error al inicializar servicio de estadísticas: {e}")
            return None

    return _statistics_service
