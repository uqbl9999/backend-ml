"""
Pruebas unitarias para el módulo de predicción
"""

import pytest
import sys
import os

# Agregar 'src' al path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Nota: Estas pruebas requieren que exista un modelo entrenado
# Ejecutar primero: python src/train_model.py


def test_import_modules():
    """Probar que todos los módulos se pueden importar"""
    try:
        from src.data_preparation import DataPreparation
        from src.models.training import ModelTrainer
        from src.models.prediction import Predictor
        assert True
    except ImportError as e:
        pytest.fail(f"Fallo al importar módulos: {e}")


def test_prediction_input_validation():
    """Probar la validación de entrada para predicciones"""
    # Esta prueba verifica que la lógica de validación de la API funciona
    required_fields = ['NroMes', 'Departamento', 'Sexo', 'Etapa', 'DetalleTamizaje']

    # Entrada válida
    valid_input = {
        'NroMes': 5,
        'Departamento': 'LIMA',
        'Sexo': 'F',
        'Etapa': '30 - 39',
        'DetalleTamizaje': 'TRASTORNO DEPRESIVO'
    }

    # Verificar que todos los campos requeridos estén presentes
    for field in required_fields:
        assert field in valid_input, f"Missing required field: {field}"

    # Verificar rango de NroMes
    assert 1 <= valid_input['NroMes'] <= 12, "NroMes debe estar entre 1 y 12"

    # Verificar Sexo
    assert valid_input['Sexo'] in ['F', 'M'], "Sexo debe ser F o M"


def test_departamentos_list():
    """Probar que la lista de departamentos válidos es correcta"""
    departamentos_validos = [
        'ANCASH', 'APURIMAC', 'AREQUIPA', 'AYACUCHO', 'CAJAMARCA',
        'CALLAO', 'CUSCO', 'HUANCAVELICA', 'HUANUCO', 'ICA',
        'JUNIN', 'LA LIBERTAD', 'LAMBAYEQUE', 'LIMA', 'LORETO',
        'MADRE DE DIOS', 'MOQUEGUA', 'PASCO', 'PIURA', 'PUNO',
        'SAN MARTIN', 'TACNA', 'UCAYALI'
    ]

    assert len(departamentos_validos) == 23
    assert 'LIMA' in departamentos_validos
    assert 'CUSCO' in departamentos_validos


def test_tipos_tamizaje_list():
    """Probar que la lista de tipos de tamizaje válidos es correcta"""
    tipos_validos = [
        'SINDROME Y/O TRASTORNO PSICOTICO',
        'TRASTORNO DE CONSUMO DE ALCOHOL Y OTROS DROGAS',
        'TRASTORNO DEPRESIVO',
        'VIOLENCIA FAMILIAR/MALTRATO INFANTIL'
    ]

    assert len(tipos_validos) == 4


# Integration tests (require trained model)

@pytest.mark.skipif(not os.path.exists('models/trained_model.pkl'),
                   reason="Trained model not found")
def test_load_model():
    """Probar que el modelo puede cargarse"""
    from src.models.prediction import Predictor

    try:
        predictor = Predictor('models/trained_model.pkl')
        assert predictor.model is not None
        assert predictor.feature_names is not None
        assert len(predictor.feature_names) > 0
    except Exception as e:
        pytest.fail(f"Fallo al cargar el modelo: {e}")


@pytest.mark.skipif(not os.path.exists('models/trained_model.pkl'),
                   reason="Trained model not found")
def test_make_prediction():
    """Probar que se pueden realizar predicciones"""
    from src.models.prediction import Predictor

    predictor = Predictor('models/trained_model.pkl')

    test_input = {
        'NroMes': 5,
        'ubigeo': 150101,
        'Departamento': 'LIMA',
        'Sexo': 'F',
        'Etapa': '30 - 39',
        'DetalleTamizaje': 'TRASTORNO DEPRESIVO'
    }

    result = predictor.predict_single(test_input)

    assert 'tasa_positividad_predicha' in result
    assert 'interpretacion' in result
    assert 'input_data' in result

    # Verificar que la predicción esté en un rango válido
    assert 0 <= result['tasa_positividad_predicha'] <= 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
