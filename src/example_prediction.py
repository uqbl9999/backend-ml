"""
Script de Ejemplo - Cómo hacer predicciones
Ejecuta esto después de entrenar un modelo para ver predicciones de ejemplo

NOTA: Estos ejemplos usan el predictor interno del modelo directamente.
      Para ejemplos de uso de la API, consulta la documentación o usa el endpoint /docs.
"""

import sys
import os

# Agregar directorio actual al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.prediction import Predictor
from src.services.ubigeo_service import get_ubigeo_service


def main():
    """
    Ejemplo de cómo usar la clase Predictor
    """
    print("="*80)
    print("EXAMPLE PREDICTIONS")
    print("="*80)

    # Cargar el modelo entrenado
    model_path = 'models/trained_model.pkl'

    if not os.path.exists(model_path):
        print(f"\n❌ Error: Model not found at {model_path}")
        print("   Please train a model first by running:")
        print("   python src/train_model.py")
        return

    print(f"\nLoading model from: {model_path}")
    predictor = Predictor(model_path)

    # Cargar servicio de ubigeo
    print("Loading ubigeo service...")
    ubigeo_service = get_ubigeo_service()

    # =========================================================================
    # EJEMPLO 1: Predicción única con mapeo automático de Ubigeo
    # =========================================================================
    print("\n" + "="*80)
    print("EXAMPLE 1: Single Prediction with Automatic Ubigeo Mapping")
    print("="*80)

    # Obtener ubigeo desde departamento y provincia
    departamento = 'LIMA'
    provincia = 'LIMA'
    ubigeo = ubigeo_service.get_ubigeo_by_dept_prov(departamento, provincia)

    print(f"\n🗺️  Location Mapping:")
    print(f"   Departamento: {departamento}")
    print(f"   Provincia: {provincia}")
    print(f"   → Ubigeo: {ubigeo}")

    example_1 = {
        'NroMes': 11,
        'ubigeo': ubigeo,
        'Departamento': departamento,
        'Sexo': 'M',
        'Etapa': '5 - 9',
        'DetalleTamizaje': 'VIOLENCIA FAMILIAR/MALTRATO INFANTIL'
    }

    print("\n📝 Input:")
    for key, value in example_1.items():
        print(f"   {key}: {value}")

    result_1 = predictor.predict_single(example_1)

    print("\n📊 Prediction:")
    print(f"   Tasa de Positividad: {result_1['tasa_positividad_predicha']:.2f}%")
    print(f"   Interpretación: {result_1['interpretacion']}")

    # =========================================================================
    # EJEMPLO 2: Predicciones múltiples
    # =========================================================================
    print("\n" + "="*80)
    print("EXAMPLE 2: Batch Predictions")
    print("="*80)

    # Preparar lote con mapeo de ubigeo
    batch_locations = [
        ('CUSCO', 'CUSCO'),
        ('AREQUIPA', 'AREQUIPA'),
        ('PIURA', 'PIURA')
    ]

    examples_batch = [
        {
            'NroMes': 7,
            'ubigeo': ubigeo_service.get_ubigeo_by_dept_prov('CUSCO', 'CUSCO'),
            'Departamento': 'CUSCO',
            'Sexo': 'M',
            'Etapa': '18 - 24',
            'DetalleTamizaje': 'VIOLENCIA FAMILIAR/MALTRATO INFANTIL'
        },
        {
            'NroMes': 3,
            'ubigeo': ubigeo_service.get_ubigeo_by_dept_prov('AREQUIPA', 'AREQUIPA'),
            'Departamento': 'AREQUIPA',
            'Sexo': 'F',
            'Etapa': '40 - 59',
            'DetalleTamizaje': 'SINDROME Y/O TRASTORNO PSICOTICO'
        },
        {
            'NroMes': 11,
            'ubigeo': ubigeo_service.get_ubigeo_by_dept_prov('PIURA', 'PIURA'),
            'Departamento': 'PIURA',
            'Sexo': 'M',
            'Etapa': '25 - 29',
            'DetalleTamizaje': 'TRASTORNO DE CONSUMO DE ALCOHOL Y OTROS DROGAS'
        }
    ]

    results_batch = predictor.predict_batch(examples_batch)

    print(f"\n📊 {len(results_batch)} Predictions:")
    print("-"*80)

    for i, result in enumerate(results_batch, 1):
        input_data = result['input_data']
        print(f"\nPrediction {i}:")
        print(f"   {input_data['Departamento']} - {input_data['DetalleTamizaje'][:40]}")
        print(f"   Mes: {input_data['NroMes']}, Sexo: {input_data['Sexo']}, Edad: {input_data['Etapa']}")
        print(f"   → Tasa: {result['tasa_positividad_predicha']:.2f}%")
        print(f"   → {result['interpretacion']}")

    # =========================================================================
    # EJEMPLO 3: Importancia de características
    # =========================================================================
    print("\n" + "="*80)
    print("EXAMPLE 3: Most Important Features")
    print("="*80)

    top_features = predictor.get_feature_importance(top_n=10)

    print("\n📈 Top 10 Most Important Features:")
    print("-"*80)

    for i, feature_info in enumerate(top_features, 1):
        feature_name = feature_info['feature']
        importance = feature_info['importance']
        print(f"{i:2}. {feature_name:50} {importance:.4f}")

    # =========================================================================
    # EJEMPLO 4: Información del modelo
    # =========================================================================
    print("\n" + "="*80)
    print("EXAMPLE 4: Model Information")
    print("="*80)

    model_info = predictor.get_model_info()

    print(f"\n📊 Model Type: {model_info['model_type']}")
    print(f"📊 Number of Features: {model_info['n_features']}")

    if 'optimized_test' in model_info['metrics']:
        test_metrics = model_info['metrics']['optimized_test']
        print(f"\n📈 Test Set Performance:")
        print(f"   R² Score: {test_metrics['R2']:.4f}")
        print(f"   MAE: {test_metrics['MAE']:.4f}%")
        print(f"   RMSE: {test_metrics['RMSE']:.4f}%")

    # =========================================================================
    # EJEMPLO 5: Comparando diferentes escenarios
    # =========================================================================
    print("\n" + "="*80)
    print("EXAMPLE 5: Comparing Different Age Groups")
    print("="*80)

    age_groups = ['18 - 24', '30 - 39', '40 - 59', '60 - 79']

    print("\nComparing depression screening by age group in Lima (Female):")
    print("-"*80)

    lima_ubigeo = ubigeo_service.get_ubigeo_by_dept_prov('LIMA', 'LIMA')

    for age in age_groups:
        test_input = {
            'NroMes': 6,
            'ubigeo': lima_ubigeo,
            'Departamento': 'LIMA',
            'Sexo': 'F',
            'Etapa': age,
            'DetalleTamizaje': 'TRASTORNO DEPRESIVO'
        }

        result = predictor.predict_single(test_input)
        print(f"   Age {age:12} → Tasa: {result['tasa_positividad_predicha']:5.2f}%")

    print("\n" + "="*80)
    print("✅ Examples completed!")
    print("="*80)


if __name__ == "__main__":
    main()
