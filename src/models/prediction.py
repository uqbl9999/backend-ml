"""
Módulo de Predicción
Gestiona predicciones usando el modelo entrenado
"""

import pandas as pd
import numpy as np
import pickle
from typing import Dict, List


class Predictor:
    """Clase para gestionar predicciones con el modelo entrenado"""

    def __init__(self, model_path: str):
        """
        Initialize predictor with a saved model

        Parameters:
        -----------
        model_path : str
            Path to the saved model pickle file
        """
        self.model_path = model_path
        self.model = None
        self.feature_names = None
        self.model_type = None
        self.metrics = None
        self.load_model()

    def load_model(self):
        """Cargar el modelo entrenado desde disco"""
        with open(self.model_path, 'rb') as f:
            model_data = pickle.load(f)

        self.model = model_data['model']
        self.feature_names = model_data['feature_names']
        self.model_type = model_data.get('model_type', 'unknown')
        self.metrics = model_data.get('metrics', {})

        print(f"✅ Model loaded: {self.model_path}")
        print(f"  Model type: {self.model_type}")
        print(f"  Features required: {len(self.feature_names)}")

    def _prepare_features(self, input_data: Dict) -> pd.DataFrame:
        """
        Preparar features a partir del diccionario de entrada

        Parameters:
        -----------
        input_data : dict
            Dictionary with input features

        Returns:
        --------
        pd.DataFrame : Prepared features
        """
        # Crear un DataFrame con todas las features inicializadas en 0
        feature_dict = {feature: 0 for feature in self.feature_names}

        # Actualizar con datos de entrada
        for key, value in input_data.items():
            if key in feature_dict:
                feature_dict[key] = value
            else:
                # Manejar variables categóricas (one-hot encoded)
                # p. ej., "Departamento" : "LIMA" -> "Departamento_LIMA" : 1
                matching_features = [f for f in self.feature_names if f.startswith(f"{key}_")]
                if matching_features:
                    # Restablecer todas las categorías para esta feature
                    for f in matching_features:
                        feature_dict[f] = 0
                    # Establecer la categoría específica
                    specific_feature = f"{key}_{value}"
                    if specific_feature in feature_dict:
                        feature_dict[specific_feature] = 1

        # Crear DataFrame
        df = pd.DataFrame([feature_dict])

        return df[self.feature_names]  # Asegurar orden correcto de columnas

    def predict_single(self, input_data: Dict) -> Dict:
        """
        Hacer una predicción para una sola entrada

        Parameters:
        -----------
        input_data : dict
            Dictionary with input features
            Example:
            {
                'NroMes': 5,
                'ubigeo': 150101,
                'Departamento': 'LIMA',
                'Sexo': 'F',
                'Etapa': '30 - 39',
                'DetalleTamizaje': 'TRASTORNO DEPRESIVO'
            }

        Returns:
        --------
        dict : Prediction result with confidence information
        """
        # Prepare features
        X = self._prepare_features(input_data)

        # Make prediction
        prediction = self.model.predict(X)[0]

        # Clip to valid range [0, 100]
        prediction = np.clip(prediction, 0, 100)

        result = {
            'tasa_positividad_predicha': round(prediction, 2),
            'interpretacion': self._interpret_prediction(prediction),
            'input_data': input_data
        }

        return result

    def predict_batch(self, input_data_list: List[Dict]) -> List[Dict]:
        """
        Hacer predicciones para múltiples entradas

        Parameters:
        -----------
        input_data_list : list of dict
            List of input data dictionaries

        Returns:
        --------
        list of dict : Prediction results
        """
        results = []

        for input_data in input_data_list:
            result = self.predict_single(input_data)
            results.append(result)

        return results

    @staticmethod
    def _interpret_prediction(prediction: float) -> str:
        """
        Interpretar el valor de la predicción

        Parameters:
        -----------
        prediction : float
            Predicted positivity rate

        Returns:
        --------
        str : Interpretation
        """
        if prediction < 2:
            return "Riesgo Muy Bajo - Bajo requerimiento de recursos"
        elif prediction < 5:
            return "Riesgo Bajo - Requerimiento normal de recursos"
        elif prediction < 10:
            return "Riesgo Moderado - Incrementar disponibilidad de personal"
        elif prediction < 20:
            return "Riesgo Alto - Priorizar asignación de especialistas"
        else:
            return "Riesgo Muy Alto - Intervención urgente requerida"

    def get_feature_importance(self, top_n: int = 10) -> List[Dict]:
        """
        Obtener las N características más importantes

        Parameters:
        -----------
        top_n : int
            Number of top features to return

        Returns:
        --------
        list of dict : Feature importance information
        """
        importances = self.model.feature_importances_

        feature_importance = [
            {'feature': feature, 'importance': float(importance)}
            for feature, importance in zip(self.feature_names, importances)
        ]

        # Ordenar por importancia
        feature_importance.sort(key=lambda x: x['importance'], reverse=True)

        return feature_importance[:top_n]

    def get_model_info(self) -> Dict:
        """
        Obtener información sobre el modelo cargado

        Returns:
        --------
        dict : Model information
        """
        info = {
            'model_type': self.model_type,
            'n_features': len(self.feature_names),
            'metrics': self.metrics
        }

        return info

    def validate_input(self, input_data: Dict) -> Dict:
        """
        Validar datos de entrada

        Parameters:
        -----------
        input_data : dict
            Input data to validate

        Returns:
        --------
        dict : Validation result
        """
        errors = []

        # Campos requeridos (ejemplos - ajuste según sus datos)
        required_fields = ['NroMes', 'Departamento', 'Sexo', 'Etapa', 'DetalleTamizaje']

        for field in required_fields:
            if field not in input_data:
                errors.append(f"Missing required field: {field}")

        # Validar NroMes (1-12)
        if 'NroMes' in input_data:
            if not isinstance(input_data['NroMes'], int) or input_data['NroMes'] < 1 or input_data['NroMes'] > 12:
                errors.append("NroMes must be an integer between 1 and 12")

        # Validar ubigeo si está presente
        if 'ubigeo' in input_data:
            if not isinstance(input_data['ubigeo'], int):
                errors.append("ubigeo must be an integer")

        return {
            'is_valid': len(errors) == 0,
            'errors': errors
        }
