using Random
using CSV
using DataFrames
using Statistics

include("ejs_hechos.jl")

println("======================================")
println(" PRIMERA APROXIMACIÓN - RNA ")
println("======================================")

# --------------------------------------------------
# 1. FIJAR SEMILLA
# --------------------------------------------------
Random.seed!(1234)

# --------------------------------------------------
# 2. CARGAR DATASET
# --------------------------------------------------
df = CSV.read("datos.csv", DataFrame)

println("\nDataset cargado correctamente.")
println("Número de filas: ", size(df, 1))
println("Número de columnas: ", size(df, 2))

# --------------------------------------------------
# 3. ELEGIR SUBPROBLEMA
# --------------------------------------------------
# Primera aproximación: pocas clases
selected_classes = [
    "Insufficient_Weight",
    "Normal_Weight",
    "Obesity_Type_I"
]

println("\nClases seleccionadas:")
println(selected_classes)

# Filtrar solo esas clases
df = filter(row -> row.NObeyesdad in selected_classes, df)

println("Número de filas tras filtrar: ", size(df, 1))

# --------------------------------------------------
# 4. DEFINIR COLUMNA OBJETIVO
# --------------------------------------------------
target_col = :NObeyesdad

# --------------------------------------------------
# 5. COLUMNAS NUMÉRICAS
# --------------------------------------------------
numeric_cols = [:Age, :Height, :Weight, :FCVC, :NCP, :CH2O, :FAF, :TUE]

# --------------------------------------------------
# 6. CODIFICAR COLUMNAS CATEGÓRICAS
# --------------------------------------------------
gender = oneHotEncoding(Vector(df.Gender), ["Female", "Male"])

family_history = oneHotEncoding(
    Vector(df.family_history_with_overweight),
    ["yes", "no"]
)

favc = oneHotEncoding(
    Vector(df.FAVC),
    ["yes", "no"]
)

caec = oneHotEncoding(
    Vector(df.CAEC),
    ["no", "Sometimes", "Frequently", "Always"]
)

smoke = oneHotEncoding(
    Vector(df.SMOKE),
    ["yes", "no"]
)

scc = oneHotEncoding(
    Vector(df.SCC),
    ["yes", "no"]
)

calc = oneHotEncoding(
    Vector(df.CALC),
    ["no", "Sometimes", "Frequently", "Always"]
)

mtrans = oneHotEncoding(
    Vector(df.MTRANS),
    ["Automobile", "Motorbike", "Bike", "Public_Transportation", "Walking"]
)

# --------------------------------------------------
# 7. CONSTRUIR MATRIZ DE ENTRADA X
# --------------------------------------------------
X_numeric = Float64.(Matrix(df[:, numeric_cols]))

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

println("\nTamaño de X: ", size(X))

# --------------------------------------------------
# 8. CONSTRUIR SALIDA y
# --------------------------------------------------
# IMPORTANTE:
# Para ANNCrossValidation, se pasan las etiquetas originales
# y la propia función las codifica internamente.
y_labels = Vector(df[:, target_col])

println("Número de etiquetas: ", length(y_labels))
println("Clases reales presentes: ", unique(y_labels))

# --------------------------------------------------
# 9. NORMALIZAR ENTRADAS
# --------------------------------------------------
X = normalizeMinMax(X)

println("\nDatos normalizados correctamente.")

# --------------------------------------------------
# 10. VALIDACIÓN CRUZADA
# --------------------------------------------------
k = 5
indices = crossvalidation(y_labels, k)

println("\nNúmero de folds: ", k)
println("Índices de validación cruzada generados correctamente.")

# --------------------------------------------------
# 11. ARQUITECTURAS A PROBAR
# --------------------------------------------------
topologies = [
    [8],
    [16],
    [32],
    [8, 4],
    [16, 8],
    [32, 16],
    [64, 32],
    [64, 32, 16]
]

# --------------------------------------------------
# 12. PROBAR TODAS LAS ARQUITECTURAS
# --------------------------------------------------
for topology in topologies

    println("\n==============================")
    println("Topología: ", topology)
    println("==============================")

    (acc_res,
     err_res,
     rec_res,
     spec_res,
     prec_res,
     npv_res,
     f1_res,
     confMatrix) = ANNCrossValidation(
        topology,
        (X, y_labels),
        indices;
        maxEpochs = 200,
        learningRate = 0.01,
        validationRatio = 0.2,
        numExecutions = 1
    )

    acc_mean, acc_std   = acc_res
    err_mean, err_std   = err_res
    rec_mean, rec_std   = rec_res
    spec_mean, spec_std = spec_res
    prec_mean, prec_std = prec_res
    npv_mean, npv_std   = npv_res
    f1_mean, f1_std     = f1_res

    println("Accuracy medio: ", acc_mean, " ± ", acc_std)
    println("Error medio: ", err_mean, " ± ", err_std)
    println("Recall medio: ", rec_mean, " ± ", rec_std)
    println("Specificity media: ", spec_mean, " ± ", spec_std)
    println("Precision media: ", prec_mean, " ± ", prec_std)
    println("NPV medio: ", npv_mean, " ± ", npv_std)
    println("F1 medio: ", f1_mean, " ± ", f1_std)

    println("\nMatriz de confusión acumulada:")
    println(confMatrix)
end

println("\n======================================")
println(" FIN DEL EXPERIMENTO ")
println("======================================")