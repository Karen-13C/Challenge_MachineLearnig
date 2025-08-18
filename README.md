# Challenge_MachineLearnig: Predicción de abandono
## Descripción del proyecto
Este proyecto tiene como objetivo predecir el churn (abandono) de clientes en base a variables relevantes en **Telecom X**. Buscamos identificar momentos críticos en la relación con el cliente y proponer estrategias para mejorar la retención y optimizar la oferta de servicios.

## Datos utilizados
El análisis se basó en una base de datos de clientes que incluye, las siguientes métricas clave:
- `customerID`: número de identificación único de cada cliente
- `Churn`: si el cliente dejó o no la empresa
- `gender`: género (masculino y femenino)
- `SeniorCitizen`: información sobre si un cliente tiene o no una edad igual o mayor a 65 años
- `Partner`: si el cliente tiene o no una pareja
- `Dependents`: si el cliente tiene o no dependientes
- `tenure`: meses de contrato del cliente
- `PhoneService`: suscripción al servicio telefónico
- `MultipleLines`: suscripción a más de una línea telefónica
- `InternetService`: suscripción a un proveedor de internet
- `OnlineSecurity`: suscripción adicional de seguridad en línea
- `OnlineBackup`: suscripción adicional de respaldo en línea
- `DeviceProtection`: suscripción adicional de protección del dispositivo
- `TechSupport`: suscripción adicional de soporte técnico, menor tiempo de espera
- `StreamingTV`: suscripción de televisión por cable
- `StreamingMovies`: suscripción de streaming de películas
- `Contract`: tipo de contrato
- `PaperlessBilling`: si el cliente prefiere recibir la factura en línea
- `PaymentMethod`: forma de pago
- `Charges.Monthly`: total de todos los servicios del cliente por mes
- `Charges.Total`: total gastado por el cliente
- `Cuentas_diarias`: Utiliza la Charges.Monthly para calcular el valor diario (tomando meses de 30 días)

## Estructura del repositorio
. \
├── TelecomX_2.ipynb    <- Cuaderno Jupyter con el análisis principal  
│   ├── 1️⃣ Objetivo del desafío  
│   ├── 2️⃣ Preparación de los datos  
│   │   ├── 2.1 Eliminación de Columnas Irrelevantes  
│   │   ├── 2.2 Encoding  
│   │   ├── 2.3 Verificación de la Proporción de Cancelación (Churn)  
│   │   ├── 2.4 Eliminación columnas repetidas  
│   │   ├── 2.5 Balanceo de Clases  
│   │   │   ├── Función para determinar `sampling_strategy`  
│   │   │   └── Balanceo  
│   │   └── 2.6 Normalización o Estandarización  
│   │       ├── Calcular sesgo de columnas numéricas  
│   │       ├── Normalizar con raíz cuadrada  
│   │       └── Estandarizar variables al rango [0,1]  
│   ├── 3️⃣ Correlación y Selección de Variables  
│   │   ├── 3.1 Análisis Dirigido  
│   │   │   ├── Tiempo de contrato vs Cancelación  
│   │   │   └── Gasto total vs Cancelación  
│   │   └── 3.2 Análisis de Correlación  
│   │       ├── Observaciones  
│   │       │   ├── `tenure` y `Total_sqrt`  
│   │       │   ├── `Monthly` y `InternetService_Fiber optic`  
│   │       │   └── `Monthly` y `InternetService_No`  
│   │       └── Conclusión  
│   ├── 4️⃣ Modelado Predictivo  
│   │   ├── 4.1 Separación de Datos  
│   │   ├── 4.2 Creación de Modelos  
│   │   │   ├── Modelo 1: KNN  
│   │   │   │   ├── Reporte inicial  
│   │   │   │   ├── Pruebas de hiperparámetros  
│   │   │   │   └── Mejor modelo KNN  
│   │   │   └── Modelo 2: Random Forest  
│   │   │       ├── Pruebas de hiperparámetros  
│   │   │       └── Mejor modelo Random Forest  
│   │   └── 4.3 Evaluación de los Modelos  
│   │       ├── Comparación de desempeño  
│   │       │   ├── Exactitud global  
│   │       │   ├── Desempeño por clase  
│   │       │   ├── Promedios  
│   │       │   ├── Overfitting / Underfitting  
│   │       │   └── Mejor desempeño  
│   │       └── Conclusión  
│   └── 5️⃣ Interpretación y Conclusiones  
│       ├── 5.1 Análisis de la Importancia de las Variables  
│       └── 5.2 Conclusión y estrategias de retención  
├── README.md                  <- Este archivo  
├── TelecomX_tratados.csv         <- Archivo de datos en formato CSV  
├── TelecomX_diccionario.md    <- Diccionario de datos / metadatos de columnas
├──Mejor_modelo_KNN.JPG         <- Imagen del reporte de clasificación KNN
└──Mejor_modelo_RandomForest.JPG    <- Imagen del reporte de clasificación RF

## Preparación de los datos

Para preparar los datos y asegurar un buen rendimiento de los modelos, seguimos varias etapas clave:

1. Clasificación de variables:
    * Las variables categóricas incluyeron información como género, tipo de contrato, servicios contratados y método de pago.

    * Las variables numéricas fueron: tenure (antigüedad del cliente), Monthly (cargo mensual) y Total (cargo total).

    * Todas las variables categóricas fueron transformadas en binarias mediante OneHotEncoder para que los modelos pudieran procesarlas de manera adecuada.

2. Transformaciones y ajustes:

    * Se aplicó SMOTE para balancear la variable objetivo. Dado que la proporción original era 71% clientes que no cancelaban y 29% que sí, se ajustó a un 60/40, evitando tanto el sesgo hacia la clase mayoritaria como la pérdida de realismo en los datos.

    * Para reducir sesgos en las variables numéricas, se corrigió la asimetría mediante transformaciones. En particular, la variable Total se ajustó con una raíz cuadrada.

    * Todas las variables numéricas se escalaron a un rango [0,1] con MinMaxScaler, dado que uno de los modelos seleccionados (KNN) es sensible a las diferencias de escala.

3. Reducción de multicolinealidad:

    * Se eliminaron variables duplicadas tras la codificación.

    * A partir de una matriz de correlación, se identificaron pares con correlación mayor a 0.7 en valor absoluto. En esos casos, se conservó la variable con mayor relación con la variable objetivo y se eliminaron las otras.

    * De esta forma, se descartaron Monthly y Total para evitar redundancias.

4. Separación de datos:

    * Los datos se dividieron en un 70% para entrenamiento y un 30% para prueba, garantizando suficiente información para ambos procesos.

5. Justificación de las decisiones:

    * La normalización se realizó pensando en KNN, modelo sensible a escalas.

    * El balanceo de clases se aplicó para mejorar la capacidad predictiva y reducir sesgos, manteniendo al mismo tiempo cierta proporción realista entre clientes que cancelan y los que no.

    * La eliminación de variables redundantes ayudó a simplificar los modelos y reducir el riesgo de sobreajuste.


## Instrucciones de ejecución

Para ejecutar el cuaderno y reproducir los resultados del proyecto, sigue estos pasos:

1. Requisitos previos

    Asegúrate de tener instalado **Python 3.9 o superior** y un entorno virtual.

2. Instalación de librerías

    En el entorno donde ejecutes el proyecto, instala las siguientes bibliotecas:

    ```bash
    pip install scikit-learn imbalanced-learn numpy matplotlib seaborn
    ```

3. Cargar los datos

    * El proyecto parte de un dataset de clientes (formato CSV).
    * Para cargar los datos, simplemente ajusta la ruta en la celda correspondiente del cuaderno:

    ```python
    import pandas as pd

    df = pd.read_csv("ruta/TelecomX_tratados.csv")
    ```

    *(reemplaza `"ruta/TelecomX_tratados.csv"` con la ubicación real de tu archivo)*

4. Ejecución del cuaderno

    1. Abre el archivo `.ipynb` en Jupyter Notebook, VS Code o Google Colab con la extensión de Jupyter.
    2. Ejecuta las celdas en orden, desde la preparación de datos hasta la evaluación de los modelos.
    3. Los resultados incluyen gráficas de análisis, métricas de desempeño y tablas comparativas entre KNN y Random Forest.

5. Resultados esperados
    Al finalizar, obtendrás:
    * Variables transformadas y listas para modelado.
    * Balanceo de clases aplicado (60/40).
    * Comparación de métricas entre modelos (KNN y Random Forest).
    * Identificación de los factores clave que explican la cancelación de clientes.
