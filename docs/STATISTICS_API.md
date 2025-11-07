# API de Estadísticas - Documentación

Esta documentación describe los servicios de estadísticas para consultar información sobre los tamizajes de salud mental.

## Endpoints Disponibles

### 1. Estadísticas Descriptivas

**Endpoint:** `GET /statistics/descriptive`

**Descripción:** Obtiene estadísticas descriptivas sobre la tasa de positividad.

**Respuesta:**
```json
{
  "media": 5.15,
  "mediana": 0.0,
  "desviacion_estandar": 18.21,
  "maximo": 100.0
}
```

**Ejemplo de uso:**
```bash
curl http://localhost:8000/statistics/descriptive
```

---

### 2. Distribución por Grupos

**Endpoint:** `GET /statistics/distribution`

**Descripción:** Obtiene la distribución de registros y suma de casos por grupo de tamizaje.

**Respuesta:**
```json
{
  "distribucion_registros": {
    "total_tamizajes": 199776,
    "solo_tamizajes_positivos": 34977,
    "tamizajes_con_violencia_politica": 139299
  },
  "suma_total_casos": {
    "total_tamizajes": 2052384,
    "solo_tamizajes_positivos": 90927,
    "tamizajes_con_violencia_politica": 1700922
  }
}
```

**Ejemplo de uso:**
```bash
curl http://localhost:8000/statistics/distribution
```

---

### 3. Heatmap por Tipo de Tamizaje

**Endpoint:** `GET /statistics/heatmap/screening-type`

**Descripción:** Obtiene casos agregados por grupo de tamizaje y tipo específico.

**Parámetros:**
- `grupo` (opcional): Filtro por grupo ('TOTAL', 'POSITIVOS', o 'VIOLENCIA')

**Respuesta:**
```json
{
  "grupo_filtro": "TOTAL",
  "data": [
    {
      "grupo": "TOTAL DE TAMIZAJES",
      "sindrome_y_o_trastorno_psicotico": 61377,
      "trastorno_de_consumo_de_alcohol_y_otros_drogas": 123060,
      "trastorno_depresivo": 219690,
      "violencia_familiar_maltrato_infantil": 1648257
    }
  ]
}
```

**Ejemplos de uso:**
```bash
# Todos los grupos
curl http://localhost:8000/statistics/heatmap/screening-type

# Solo totales
curl "http://localhost:8000/statistics/heatmap/screening-type?grupo=TOTAL"

# Solo positivos
curl "http://localhost:8000/statistics/heatmap/screening-type?grupo=POSITIVOS"

# Solo violencia
curl "http://localhost:8000/statistics/heatmap/screening-type?grupo=VIOLENCIA"
```

---

### 4. Heatmap por Departamento

**Endpoint:** `GET /statistics/heatmap/department`

**Descripción:** Obtiene casos agregados por departamento y grupo de tamizaje.

**Parámetros:**
- `grupo` (opcional): Filtro por grupo ('TOTAL', 'POSITIVOS', o 'VIOLENCIA')
- `top_n` (opcional): Limitar a los top N departamentos (1-50)

**Respuesta:**
```json
{
  "grupo_filtro": "todos",
  "top_n": 5,
  "data": [
    {
      "departamento": "AYACUCHO",
      "solo_tamizajes_positivos": 5368,
      "tamizajes_c_condicion_adicional_violencia_politica": 2803,
      "total_de_tamizajes": 292605,
      "total": 300776
    },
    {
      "departamento": "HUANUCO",
      "solo_tamizajes_positivos": 4512,
      "tamizajes_c_condicion_adicional_violencia_politica": 892,
      "total_de_tamizajes": 270300,
      "total": 275704
    }
  ]
}
```

**Ejemplos de uso:**
```bash
# Todos los departamentos
curl http://localhost:8000/statistics/heatmap/department

# Top 10 departamentos
curl "http://localhost:8000/statistics/heatmap/department?top_n=10"

# Top 5 departamentos con solo totales
curl "http://localhost:8000/statistics/heatmap/department?grupo=TOTAL&top_n=5"
```

---

### 5. Resumen por Tipo de Tamizaje

**Endpoint:** `GET /statistics/screening-types`

**Descripción:** Obtiene resumen de tipos de tamizaje con estadísticas completas.

**Respuesta:**
```json
{
  "count": 4,
  "data": [
    {
      "detalle_tamizaje": "VIOLENCIA FAMILIAR/MALTRATO INFANTIL",
      "total_registros": 120845,
      "suma_total_casos": 1647755,
      "suma_positivos": 45580,
      "tasa_positividad_promedio": 2.86,
      "tasa_positividad_mediana": 0.0,
      "tasa_positividad_max": 100.0
    },
    {
      "detalle_tamizaje": "TRASTORNO DEPRESIVO",
      "total_registros": 41172,
      "suma_total_casos": 219690,
      "suma_positivos": 31511,
      "tasa_positividad_promedio": 14.77,
      "tasa_positividad_mediana": 0.0,
      "tasa_positividad_max": 100.0
    }
  ]
}
```

**Ejemplo de uso:**
```bash
curl http://localhost:8000/statistics/screening-types
```

---

### 6. Resumen por Departamento

**Endpoint:** `GET /statistics/departments`

**Descripción:** Obtiene resumen de departamentos con estadísticas completas.

**Parámetros:**
- `top_n` (opcional): Limitar a los top N departamentos (1-50)

**Respuesta:**
```json
{
  "count": 3,
  "top_n": 3,
  "data": [
    {
      "departamento": "AYACUCHO",
      "total_registros": 28061,
      "suma_total_casos": 292605,
      "suma_positivos": 4640,
      "tasa_positividad_promedio": 1.38,
      "tasa_positividad_mediana": 0.0,
      "tasa_positividad_max": 100.0
    },
    {
      "departamento": "HUANUCO",
      "total_registros": 20166,
      "suma_total_casos": 270298,
      "suma_positivos": 3655,
      "tasa_positividad_promedio": 3.77,
      "tasa_positividad_mediana": 0.0,
      "tasa_positividad_max": 100.0
    }
  ]
}
```

**Ejemplos de uso:**
```bash
# Todos los departamentos
curl http://localhost:8000/statistics/departments

# Top 10 departamentos
curl "http://localhost:8000/statistics/departments?top_n=10"
```

---

## Correspondencia con las Visualizaciones

### DISTRIBUCIÓN POR GRUPOS

Esta visualización corresponde al endpoint:
- **GET /statistics/distribution**

Muestra:
- **Distribución de registros:** Total de Tamizajes (199,776), Solo Tamizajes Positivos (34,977), Tamizajes con Condición Adicional Violencia Política
- **Suma total de casos:** Total de Tamizajes (2,052,384), Solo Tamizajes Positivos, Tamizajes con Condición Adicional Violencia Política

### HEATMAP POR TIPO DE TAMIZAJE

Esta visualización corresponde al endpoint:
- **GET /statistics/heatmap/screening-type**

Muestra casos por tipo de tamizaje y grupo:
- VIOLENCIA FAMILIAR: Total (1,532,884), Positivos (65,149), Con Violencia Política (1,774)
- MALTRATO INFANTIL: Total (198,224), Positivos (9,802), Con Violencia Política (942)
- TRASTORNO DEPRESIVO: Total (166,542), Positivos (6,331), Con Violencia Política (601)
- CONSUMO DE ALCOHOL: Total (90,452), Positivos (5,904), Con Violencia Política (330)
- CONSUMO DE DROGAS: Total (36,221), Positivos (3,814), Con Violencia Política (211)
- TRASTORNO PSICÓTICO: Total (28,903), Positivos (1,974), Con Violencia Política (199)
- VIOLENCIA POLÍTICA: Total (5,647), Positivos (1,877), Con Violencia Política (187)

### HEATMAP POR DEPARTAMENTO

Esta visualización corresponde al endpoint:
- **GET /statistics/heatmap/department**

Muestra volumen de casos por departamento y grupo:
- AYACUCHO: Total (268,441), Positivos (27,349), Con Violencia Política (2,144)
- HUÁNUCO: Total (231,003), Positivos (23,008), Con Violencia Política (1,942)
- JUNÍN: Total (208,944), Positivos (21,115), Con Violencia Política (1,771)
- Etc.

### ESTADÍSTICAS DESCRIPTIVAS

Esta visualización corresponde al endpoint:
- **GET /statistics/descriptive**

Muestra:
- **MEDIA:** 22.4%
- **MEDIANA:** 20.8%
- **DESV. ESTÁNDAR:** 12.5%
- **MÁXIMO:** 68.2%

---

## Código de Ejemplo en Python

```python
import requests

BASE_URL = "http://localhost:8000"

# 1. Obtener estadísticas descriptivas
response = requests.get(f"{BASE_URL}/statistics/descriptive")
print("Estadísticas Descriptivas:", response.json())

# 2. Obtener distribución por grupos
response = requests.get(f"{BASE_URL}/statistics/distribution")
print("Distribución:", response.json())

# 3. Obtener heatmap por tipo de tamizaje
response = requests.get(f"{BASE_URL}/statistics/heatmap/screening-type", params={"grupo": "TOTAL"})
print("Heatmap por Tipo:", response.json())

# 4. Obtener top 10 departamentos
response = requests.get(f"{BASE_URL}/statistics/heatmap/department", params={"top_n": 10})
print("Heatmap por Departamento:", response.json())

# 5. Obtener resumen de tipos de tamizaje
response = requests.get(f"{BASE_URL}/statistics/screening-types")
print("Resumen Tipos:", response.json())

# 6. Obtener top 5 departamentos con estadísticas
response = requests.get(f"{BASE_URL}/statistics/departments", params={"top_n": 5})
print("Resumen Departamentos:", response.json())
```

---

## Código de Ejemplo en JavaScript

```javascript
const BASE_URL = "http://localhost:8000";

// 1. Obtener estadísticas descriptivas
fetch(`${BASE_URL}/statistics/descriptive`)
  .then(response => response.json())
  .then(data => console.log("Estadísticas Descriptivas:", data));

// 2. Obtener distribución por grupos
fetch(`${BASE_URL}/statistics/distribution`)
  .then(response => response.json())
  .then(data => console.log("Distribución:", data));

// 3. Obtener heatmap por tipo de tamizaje
fetch(`${BASE_URL}/statistics/heatmap/screening-type?grupo=TOTAL`)
  .then(response => response.json())
  .then(data => console.log("Heatmap por Tipo:", data));

// 4. Obtener top 10 departamentos
fetch(`${BASE_URL}/statistics/heatmap/department?top_n=10`)
  .then(response => response.json())
  .then(data => console.log("Heatmap por Departamento:", data));

// 5. Obtener resumen de tipos de tamizaje
fetch(`${BASE_URL}/statistics/screening-types`)
  .then(response => response.json())
  .then(data => console.log("Resumen Tipos:", data));

// 6. Obtener top 5 departamentos con estadísticas
fetch(`${BASE_URL}/statistics/departments?top_n=5`)
  .then(response => response.json())
  .then(data => console.log("Resumen Departamentos:", data));
```

---

## Notas Técnicas

1. **Servicio Singleton:** El servicio de estadísticas se carga una sola vez al inicio de la aplicación para optimizar el rendimiento.

2. **Cálculo de Tasa de Positividad:** Se calcula automáticamente como `(Positivos / Total) * 100` y se filtran tasas imposibles (> 100%).

3. **Ordenamiento:** Los resultados se ordenan por relevancia (suma total de casos en orden descendente).

4. **Parámetros opcionales:**
   - `grupo`: Filtra por tipo de grupo de tamizaje
   - `top_n`: Limita la cantidad de resultados (1-50)

5. **Codificación:** Todos los resultados están en formato JSON con codificación UTF-8.

---

## Manejo de Errores

Todos los endpoints pueden retornar los siguientes códigos de error:

- **400 Bad Request:** Parámetros inválidos (ej. `top_n` fuera de rango)
- **500 Internal Server Error:** Error en el procesamiento de datos
- **503 Service Unavailable:** Servicio de estadísticas no disponible

Ejemplo de error:
```json
{
  "detail": "top_n debe estar entre 1 y 50"
}
```
