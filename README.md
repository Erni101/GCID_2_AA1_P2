# GCID_2_AA1_P2

Practica 2 de AA1 sobre clasificacion del nivel de obesidad usando distintos modelos de aprendizaje automatico en Julia.

## Objetivo

Este repositorio implementa el flujo del ejercicio 4:

- Preparacion y preprocesado de datos.
- Evaluacion con validacion cruzada estratificada.
- Comparativa de 5 tecnicas: ANN, DoME, SVM, arboles de decision y kNN.
- Exportacion de tablas de resultados para documentar la memoria.

## Estructura del repositorio

- [data/ObesityDataSet_raw_and_data_sinthetic.csv](data/ObesityDataSet_raw_and_data_sinthetic.csv): dataset original.
- [instrucciones/ejericio_4_enunciado.txt](instrucciones/ejericio_4_enunciado.txt): enunciado del apartado 4.
- [scr/Codigo_profesor_ejercicio_2.jl](scr/Codigo_profesor_ejercicio_2.jl): funciones base (normalizacion, metricas, CV y modelos).
- [scr/main.jl](scr/main.jl): script principal del ejercicio 4.
- [test/](test): carpeta donde se guardan resultados de ejecucion.

## Que hace el script principal

El script [scr/main.jl](scr/main.jl) ejecuta automaticamente:

1. Carga del dataset.
2. Codificacion one-hot de variables categoricas.
3. Normalizacion Min-Max de entradas.
4. Descripcion de la configuracion experimental (parte 4.1).
5. Validacion cruzada estratificada con 5 folds.
6. Barrido de hiperparametros para:
   - ANN: 8 arquitecturas (1 y 2 capas ocultas).
   - DoME: 8 valores de numero de nodos.
   - SVM: 8 configuraciones de kernel/C (y parametros extra cuando aplica).
   - Arboles de decision: 6 profundidades maximas.
   - kNN: 6 valores de k.
7. Guardado de resultados en CSV (parte 4.2).

## Requisitos

Julia 1.10 o superior (recomendado).

Paquetes necesarios:

- CSV
- DataFrames
- Flux
- MLJ
- LIBSVM
- MLJLIBSVMInterface
- NearestNeighborModels
- MLJDecisionTreeInterface
- SymDoME

Instalacion rapida:

```julia
using Pkg
Pkg.add([
	"CSV", "DataFrames", "Flux", "MLJ", "LIBSVM",
	"MLJLIBSVMInterface", "NearestNeighborModels",
	"MLJDecisionTreeInterface", "SymDoME"
])
```

## Ejecucion

Desde la raiz del repositorio:

```bash
julia scr/main.jl
```

## Salidas generadas

Tras ejecutar, se crea la carpeta [test/resultados_ej4](test/resultados_ej4) con:

- `normalizacion_minmax.csv`: parametros min/max por feature final.
- `resultados_ann.csv`
- `resultados_dome.csv`
- `resultados_svm.csv`
- `resultados_decisiontree.csv`
- `resultados_knn.csv`
- `resumen_mejores_modelos.csv`: mejor configuracion por tecnica.
- `metadata_experimento.csv`: informacion de ejecucion (fecha, seed, folds, etc.).

## Reproducibilidad

- Semilla fija en [scr/main.jl](scr/main.jl): `SEED = 2026`.
- Numero de folds: `KFOLDS = 5`.
- Si cambias hiperparametros o semilla, los resultados numericos tambien cambiaran.

## Nota para la memoria

Para mantener coherencia con la correccion del profesor, las tablas incluidas en la memoria deben salir directamente de los CSV generados por [scr/main.jl](scr/main.jl).
