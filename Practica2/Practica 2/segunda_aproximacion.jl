using CSV
using DataFrames
using Statistics

include("ejs_hechos.jl")

println("======================================")
println(" SEGUNDA APROXIMACIÓN ")
println("======================================")

# 1. Cargar dataset
df = CSV.read("datos.csv", DataFrame)

# 2. Seleccionar MÁS clases (segunda aproximación)
selected_classes = [
    "Insufficient_Weight",
    "Normal_Weight",
    "Overweight_Level_I",
    "Overweight_Level_II",
    "Obesity_Type_I"
]

println("Clases usadas:")
println(selected_classes)

# Filtrar dataset
df_sub = filter(row -> row.NObeyesdad in selected_classes, df)

# 3. Target
target_col = :NObeyesdad

# 4. Columnas numéricas
numeric_cols = [:Age, :Height, :Weight, :FCVC, :NCP, :CH2O, :FAF, :TUE]

# 5. Codificación categórica
gender = oneHotEncoding(Vector(df_sub.Gender), ["Female", "Male"])

family_history = oneHotEncoding(
    Vector(df_sub.family_history_with_overweight),
    ["yes", "no"]
)

favc = oneHotEncoding(Vector(df_sub.FAVC), ["yes", "no"])

caec = oneHotEncoding(
    Vector(df_sub.CAEC),
    ["no", "Sometimes", "Frequently", "Always"]
)

smoke = oneHotEncoding(Vector(df_sub.SMOKE), ["yes", "no"])

scc = oneHotEncoding(Vector(df_sub.SCC), ["yes", "no"])

calc = oneHotEncoding(
    Vector(df_sub.CALC),
    ["no", "Sometimes", "Frequently", "Always"]
)

mtrans = oneHotEncoding(
    Vector(df_sub.MTRANS),
    ["Automobile", "Motorbike", "Bike", "Public_Transportation", "Walking"]
)

# 6. Construir X
X = hcat(
    Float64.(Matrix(df_sub[:, numeric_cols])),
    Float64.(gender),
    Float64.(family_history),
    Float64.(favc),
    Float64.(caec),
    Float64.(smoke),
    Float64.(scc),
    Float64.(calc),
    Float64.(mtrans)
)

# 7. Salida y
y_labels = Vector(df_sub[:, target_col])
y = oneHotEncoding(y_labels, selected_classes)

# 8. Normalizar
X = normalizeMinMax(X)

# 9. Split
(train_idx, val_idx, test_idx) = holdOut(size(X, 1), 0.2, 0.2)

Xtrain, ytrain = X[train_idx, :], y[train_idx, :]
Xval,   yval   = X[val_idx,   :], y[val_idx,   :]
Xtest,  ytest  = X[test_idx,  :], y[test_idx,  :]

# 10. Red
topology = [16, 8]

# 11. Entrenamiento
ann, trainingLosses, validationLosses, testLosses = trainClassANN(
    topology,
    (Xtrain, ytrain);
    validationDataset = (Xval, yval),
    testDataset = (Xtest, ytest),
    maxEpochs = 200,
    learningRate = 0.01,
    maxEpochsVal = 20
)

# 12. Predicción
test_outputs = collect(ann(Float32.(Xtest'))')

# 13. Evaluación
(acc, errorRate, recall, specificity, precision, NPV, F1, confMatrix) =
    confusionMatrix(test_outputs, ytest)

println("\nRESULTADOS TEST")
println("Accuracy = ", acc)
println("F1 = ", F1)

println("\nMatriz de confusión:")
println(confMatrix)