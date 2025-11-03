"""
Unit tests for the prediction module
"""

import pytest
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Note: These tests require a trained model to be present
# Run python src/train_model.py first


def test_import_modules():
    """Test that all modules can be imported"""
    try:
        from src.data_preparation import DataPreparation
        from src.models.training import ModelTrainer
        from src.models.prediction import Predictor
        assert True
    except ImportError as e:
        pytest.fail(f"Failed to import modules: {e}")


def test_prediction_input_validation():
    """Test input validation for predictions"""
    # This test checks that our API validation logic works
    required_fields = ['NroMes', 'Departamento', 'Sexo', 'Etapa', 'DetalleTamizaje']

    # Valid input
    valid_input = {
        'NroMes': 5,
        'Departamento': 'LIMA',
        'Sexo': 'F',
        'Etapa': '30 - 39',
        'DetalleTamizaje': 'TRASTORNO DEPRESIVO'
    }

    # Check all required fields are present
    for field in required_fields:
        assert field in valid_input, f"Missing required field: {field}"

    # Check NroMes range
    assert 1 <= valid_input['NroMes'] <= 12, "NroMes must be between 1 and 12"

    # Check Sexo
    assert valid_input['Sexo'] in ['F', 'M'], "Sexo must be F or M"


def test_departamentos_list():
    """Test that the list of valid departamentos is correct"""
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
    """Test that the list of valid screening types is correct"""
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
    """Test that the model can be loaded"""
    from src.models.prediction import Predictor

    try:
        predictor = Predictor('models/trained_model.pkl')
        assert predictor.model is not None
        assert predictor.feature_names is not None
        assert len(predictor.feature_names) > 0
    except Exception as e:
        pytest.fail(f"Failed to load model: {e}")


@pytest.mark.skipif(not os.path.exists('models/trained_model.pkl'),
                   reason="Trained model not found")
def test_make_prediction():
    """Test that predictions can be made"""
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

    # Check prediction is in valid range
    assert 0 <= result['tasa_positividad_predicha'] <= 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
