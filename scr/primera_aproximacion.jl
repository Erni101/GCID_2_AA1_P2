
using Pkg

# Instalar paquetes solo la primera vez
# Pkg.add("CSV")
# Pkg.add("DataFrames")
# Pkg.add("Flux")
# Pkg.add("Statistics")

using CSV
using DataFrames
using Statistics

# Cargar funciones del archivo del ejercicio
include("ejs_hechos.jl")

println("======================================")
println(" CLASIFICACIÓN POR SUBPROBLEMAS ")
println("======================================")

# --------------------------------------------------
# 1. CARGAR DATASET
# --------------------------------------------------
df = CSV.read("datos.csv", DataFrame)

println("\nDataset cargado correctamente.")
println("Número de filas: ", size(df, 1))
println("Número de columnas: ", size(df, 2))

# --------------------------------------------------
# 2. ELEGIR SUBPROBLEMA
# --------------------------------------------------
# Aquí eliges solo unas pocas clases
selected_classes = [
    "Insufficient_Weight",
    "Normal_Weight",
    "Obesity_Type_I"
]

println("\nClases seleccionadas para este subproblema:")
println(selected_classes)

# Filtrar solo las filas que pertenezcan a esas clases
df_sub = filter(row -> row.NObeyesdad in selected_classes, df)

println("\nNúmero de filas tras filtrar las clases: ", size(df_sub, 1))

# --------------------------------------------------
# 3. DEFINIR COLUMNA OBJETIVO
# --------------------------------------------------
target_col = :NObeyesdad

# --------------------------------------------------
# 4. DEFINIR COLUMNAS NUMÉRICAS
# --------------------------------------------------
numeric_cols = [:Age, :Height, :Weight, :FCVC, :NCP, :CH2O, :FAF, :TUE]

# --------------------------------------------------
# 5. CODIFICAR COLUMNAS CATEGÓRICAS
# --------------------------------------------------
# Cada columna categórica se transforma en variables 0/1

gender = oneHotEncoding(Vector(df_sub.Gender), ["Female", "Male"])

family_history = oneHotEncoding(
    Vector(df_sub.family_history_with_overweight),
    ["yes", "no"]
)

favc = oneHotEncoding(
    Vector(df_sub.FAVC),
    ["yes", "no"]
)

caec = oneHotEncoding(
    Vector(df_sub.CAEC),
    ["no", "Sometimes", "Frequently", "Always"]
)

smoke = oneHotEncoding(
    Vector(df_sub.SMOKE),
    ["yes", "no"]
)

scc = oneHotEncoding(
    Vector(df_sub.SCC),
    ["yes", "no"]
)

calc = oneHotEncoding(
    Vector(df_sub.CALC),
    ["no", "Sometimes", "Frequently", "Always"]
)

mtrans = oneHotEncoding(
    Vector(df_sub.MTRANS),
    ["Automobile", "Motorbike", "Bike", "Public_Transportation", "Walking"]
)

# --------------------------------------------------
# 6. CONSTRUIR MATRIZ DE ENTRADA X
# --------------------------------------------------
X_numeric = Float64.(Matrix(df_sub[:, numeric_cols]))

X = hcat(
    X_numeric,
    Float64.(gender),
    Float64.(family_history),
    Float64.(favc),
    Float64.(caec),
    Float64.(smoke),
    Float64.(scc),
    Float64.(calc),
    Float64.(mtrans)
)

println("\nTamaño de la matriz de entrada X: ", size(X))

# --------------------------------------------------
# 7. CONSTRUIR SALIDA y
# --------------------------------------------------
# Usamos solo las clases seleccionadas
y_labels = Vector(df_sub[:, target_col])
y = oneHotEncoding(y_labels, selected_classes)

println("Tamaño de la matriz de salida y: ", size(y))

# --------------------------------------------------
# 8. NORMALIZAR LAS ENTRADAS
# --------------------------------------------------
X = normalizeMinMax(X)

println("\nEntradas normalizadas correctamente.")

# --------------------------------------------------
# 9. SEPARAR EN TRAIN / VALID / TEST
# --------------------------------------------------
# 20% validación, 20% test, 60% entrenamiento
(train_idx, val_idx, test_idx) = holdOut(size(X, 1), 0.2, 0.2)

Xtrain = X[train_idx, :]
ytrain = y[train_idx, :]

Xval = X[val_idx, :]
yval = y[val_idx, :]

Xtest = X[test_idx, :]
ytest = y[test_idx, :]

println("\nTamaño train: ", size(Xtrain))
println("Tamaño validation: ", size(Xval))
println("Tamaño test: ", size(Xtest))

# --------------------------------------------------
# 10. DEFINIR TOPOLOGÍA DE LA RED
# --------------------------------------------------
topology = [16, 8]

println("\nTopología de la red: ", topology)

# --------------------------------------------------
# 11. ENTRENAR RED NEURONAL
# --------------------------------------------------
ann, trainingLosses, validationLosses, testLosses = trainClassANN(
    topology,
    (Xtrain, ytrain);
    validationDataset = (Xval, yval),
    testDataset = (Xtest, ytest),
    maxEpochs = 200,
    learningRate = 0.01,
    maxEpochsVal = 20
)

println("\nEntrenamiento terminado.")

# --------------------------------------------------
# 12. OBTENER PREDICCIONES EN TEST
# --------------------------------------------------
# La red espera patrones en columnas, por eso se transpone Xtest
raw_outputs = ann(Float32.(Xtest'))
test_outputs = collect(raw_outputs')   # volvemos a poner cada patrón por filas

# --------------------------------------------------
# 13. EVALUAR RESULTADOS
# --------------------------------------------------
(acc, errorRate, recall, specificity, precision, NPV, F1, confMatrix) =
    confusionMatrix(test_outputs, ytest)

println("\n======================================")
println(" RESULTADOS TEST ")
println("======================================")
println("Accuracy = ", acc)
println("Error rate = ", errorRate)
println("Recall = ", recall)
println("Specificity = ", specificity)
println("Precision = ", precision)
println("NPV = ", NPV)
println("F1 = ", F1)

println("\nMatriz de confusión:")
println(confMatrix)

# --------------------------------------------------
# 14. VER CLASES Y EJEMPLOS
# --------------------------------------------------
println("\nOrden de las clases usado en la codificación:")
println(selected_classes)

# Convertir predicciones a etiquetas para ver algunos ejemplos
pred_bool = classifyOutputs(test_outputs)

pred_labels = Vector{String}(undef, size(pred_bool, 1))
real_labels = Vector{String}(undef, size(ytest, 1))

for i in 1:size(pred_bool, 1)
    pred_idx = findfirst(pred_bool[i, :])
    real_idx = findfirst(ytest[i, :])

    pred_labels[i] = selected_classes[pred_idx]
    real_labels[i] = selected_classes[real_idx]
end

println("\nPrimeras 10 predicciones:")
for i in 1:min(10, length(pred_labels))
    println("Real: ", real_labels[i], "   |   Predicción: ", pred_labels[i])
end