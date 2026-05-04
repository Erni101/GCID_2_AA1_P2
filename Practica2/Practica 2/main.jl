import Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

using Random
using Statistics
using Printf
using Dates
using CSV
using DataFrames
using DelimitedFiles
using PrettyTables

include(joinpath(@__DIR__, "ejs_hechos.jl"))

const DATA_PATH = joinpath(@__DIR__, "datos", "datos.csv")
const OUTPUT_DIR = joinpath(@__DIR__, "resultados")
const INDICES_PATH = joinpath(@__DIR__, "resultados", "indices_cv.csv")

const SEED = 2026
const KFOLDS = 5


function config_to_string(cfg::AbstractDict)
    ordered_keys = sort(collect(keys(cfg)))
    parts = String[]

    for key in ordered_keys
        value = cfg[key]

        if value isa AbstractVector
            push!(parts, string(key, "=", join(value, "-")))
        else
            push!(parts, string(key, "=", value))
        end
    end

    return join(parts, ", ")
end


function round_numeric_df(df::DataFrame; digits::Int=4)
    df2 = copy(df)

    for col in names(df2)
        df2[!, col] = [x isa Real ? round(x, digits=digits) : x for x in df2[!, col]]
    end

    return df2
end


function build_input_matrix(df::DataFrame, target_col::String)
    feature_columns = [col for col in names(df) if col != target_col]

    input_blocks = Matrix{Float64}[]
    expanded_feature_names = String[]

    for col in feature_columns
        values = df[!, col]

        if eltype(values) <: Real
            push!(input_blocks, reshape(Float64.(values), :, 1))
            push!(expanded_feature_names, col)
        else
            values_as_text = string.(values)
            classes = sort(unique(values_as_text))
            encoded = oneHotEncoding(values_as_text, classes)
            encoded_float = Float64.(encoded)
            push!(input_blocks, encoded_float)

            if length(classes) <= 2
                push!(expanded_feature_names, string(col, "==", classes[1]))
            else
                append!(expanded_feature_names, [string(col, "==", class) for class in classes])
            end
        end
    end

    return hcat(input_blocks...), feature_columns, expanded_feature_names
end


function save_cv_indices(indices::Vector{Int64}, path::String)
    writedlm(path, reshape(indices, :, 1), ',')
end


function load_cv_indices(path::String)
    return Int64.(vec(readdlm(path, ',', Int)))
end


function get_or_create_cv_indices(targets::AbstractVector)
    if isfile(INDICES_PATH)
        println("Cargando indices de validacion cruzada desde: ", INDICES_PATH)
        indices = load_cv_indices(INDICES_PATH)

        if length(indices) != length(targets)
            error("El archivo de indices no coincide con el numero de patrones.")
        end

        return indices
    else
        println("Generando indices de validacion cruzada en: ", INDICES_PATH)
        Random.seed!(SEED)
        indices = crossvalidation(targets, Int64(KFOLDS))
        save_cv_indices(indices, INDICES_PATH)
        return indices
    end
end


function save_bar_chart_svg(labels::Vector{String}, values::Vector{Float64}, title::String, ylabel::String, path::String)
    width = 1000
    height = 600
    margin_left = 100
    margin_right = 50
    margin_top = 80
    margin_bottom = 140

    chart_width = width - margin_left - margin_right
    chart_height = height - margin_top - margin_bottom

    max_value = maximum(values)
    max_value = max(max_value, 1e-9)

    n = length(labels)
    bar_gap = 30
    bar_width = (chart_width - bar_gap * (n + 1)) / n

    colors = [
        "#4E79A7",
        "#F28E2B",
        "#59A14F",
        "#E15759",
        "#B07AA1",
        "#76B7B2",
        "#EDC948",
        "#FF9DA7"
    ]

    open(path, "w") do io
        println(io, """<svg xmlns="http://www.w3.org/2000/svg" width="$width" height="$height">""")
        println(io, """<rect width="100%" height="100%" fill="#FAFAFA"/>""")
        println(io, """<text x="$(width/2)" y="40" text-anchor="middle" font-size="26" font-weight="bold" font-family="Arial" fill="#222">$title</text>""")

        x0 = margin_left
        y0 = height - margin_bottom

        for j in 0:5
            value = max_value * j / 5
            y = y0 - chart_height * j / 5
            println(io, """<line x1="$x0" y1="$y" x2="$(width-margin_right)" y2="$y" stroke="#DDDDDD" stroke-width="1"/>""")
            println(io, """<text x="$(x0-12)" y="$(y+5)" text-anchor="end" font-size="12" font-family="Arial" fill="#555">$(round(value, digits=2))</text>""")
        end

        println(io, """<line x1="$x0" y1="$margin_top" x2="$x0" y2="$y0" stroke="#333" stroke-width="2"/>""")
        println(io, """<line x1="$x0" y1="$y0" x2="$(width-margin_right)" y2="$y0" stroke="#333" stroke-width="2"/>""")
        println(io, """<text x="30" y="$(height/2)" transform="rotate(-90,30,$(height/2))" text-anchor="middle" font-size="17" font-family="Arial" fill="#333">$ylabel</text>""")

        for i in 1:n
            bar_h = values[i] / max_value * chart_height
            x = margin_left + bar_gap + (i - 1) * (bar_width + bar_gap)
            y = y0 - bar_h
            color = colors[mod1(i, length(colors))]

            println(io, """<rect x="$(x+4)" y="$(y+4)" width="$bar_width" height="$bar_h" rx="8" fill="#000000" opacity="0.12"/>""")
            println(io, """<rect x="$x" y="$y" width="$bar_width" height="$bar_h" rx="8" fill="$color"/>""")
            println(io, """<text x="$(x + bar_width/2)" y="$(y - 10)" text-anchor="middle" font-size="14" font-weight="bold" font-family="Arial" fill="#222">$(round(values[i], digits=4))</text>""")
            println(io, """<text x="$(x + bar_width/2)" y="$(y0 + 30)" text-anchor="end" font-size="14" font-family="Arial" fill="#333" transform="rotate(-35,$(x + bar_width/2),$(y0 + 30))">$(labels[i])</text>""")
        end

        println(io, "</svg>")
    end

    println("Grafica guardada en: ", path)
end


function describe_setup(
    df::DataFrame,
    target_col::String,
    raw_inputs::AbstractMatrix{<:Real},
    normalized_inputs::AbstractMatrix{<:Real},
    expanded_feature_names::Vector{String},
    min_values,
    max_values,
    mean_values
)
    targets = Vector{String}(string.(df[!, target_col]))
    classes = sort(unique(targets))

    println("\n======================")
    println("PARTE 4.1 - DESCRIPCION")
    println("======================")

    println("\n1) Base de datos final")
    println("   - Patrones: ", nrow(df))
    println("   - Entradas originales: ", ncol(df) - 1)
    println("   - Entradas finales tras codificacion: ", size(raw_inputs, 2))
    println("   - Clases de salida: ", length(classes), " -> ", join(classes, ", "))
    println("   - Atributos eliminados: ninguno")

    println("\n2) Preprocesado")
    println("   - Codificacion: one-hot para categoricas (binaria en 1 columna, multiclase en varias)")
    println("   - Normalizacion: Min-Max [0,1] sobre todas las entradas numericas finales")
    println("   - Justificacion: evita dominancia de escala entre atributos para RRNN, SVM y kNN")
    println("   - Rango observado tras normalizar: [",
        @sprintf("%.4f", minimum(normalized_inputs)), ", ",
        @sprintf("%.4f", maximum(normalized_inputs)), "]")

    println("\n3) Configuracion experimental")
    println("   - Semilla global: ", SEED)
    println("   - Evaluacion: validacion cruzada estratificada con ", KFOLDS, " folds")
    println("   - Archivo de indices: ", INDICES_PATH)
    println("   - Metricas: Accuracy, ErrorRate, Recall, Specificity, Precision, NPV, F1")

    println("\n4) Distribucion de clases")
    class_df = combine(groupby(DataFrame(clase=targets), :clase), nrow => :conteo)
    sort!(class_df, :clase)
    pretty_table(class_df; display_size=(-1, -1))

    params_df = DataFrame(
        feature = expanded_feature_names,
        min = vec(min_values),
        max = vec(max_values),
        media = vec(mean_values)
    )

    norm_file = joinpath(OUTPUT_DIR, "normalizacion_minmax.csv")
    CSV.write(norm_file, params_df)
    println("\nParametros de normalizacion guardados en: ", norm_file)
end


function evaluate_configs(
    model_name::String,
    model_type::Symbol,
    configs::Vector{Dict{String,Any}},
    dataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{<:Any,1}},
    cv_indices::Vector{Int64}
)
    rows = NamedTuple{
        (:Modelo, :Configuracion, :AccuracyMedia, :AccuracySTD, :ErrorMedia, :ErrorSTD, :RecallMedia, :RecallSTD, :SpecificityMedia, :SpecificitySTD, :PrecisionMedia, :PrecisionSTD, :NPVMedia, :NPVSTD, :F1Media, :F1STD),
        Tuple{String, String, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64}
    }[]

    for (i, cfg) in enumerate(configs)
        println("\n[", model_name, "] Configuracion ", i, "/", length(configs), ": ", config_to_string(cfg))

        (acc, err, rec, spec, prec, npv, f1, _) = modelCrossValidation(model_type, cfg, dataset, cv_indices)

        push!(rows, (
            Modelo = model_name,
            Configuracion = config_to_string(cfg),
            AccuracyMedia = Float64(acc[1]),
            AccuracySTD = Float64(acc[2]),
            ErrorMedia = Float64(err[1]),
            ErrorSTD = Float64(err[2]),
            RecallMedia = Float64(rec[1]),
            RecallSTD = Float64(rec[2]),
            SpecificityMedia = Float64(spec[1]),
            SpecificitySTD = Float64(spec[2]),
            PrecisionMedia = Float64(prec[1]),
            PrecisionSTD = Float64(prec[2]),
            NPVMedia = Float64(npv[1]),
            NPVSTD = Float64(npv[2]),
            F1Media = Float64(f1[1]),
            F1STD = Float64(f1[2])
        ))
    end

    result_df = DataFrame(rows)
    sort!(result_df, :F1Media, rev=true)
    return result_df
end


function run_exercise_4()
    mkpath(OUTPUT_DIR)
    Random.seed!(SEED)

    println("Cargando datos desde: ", DATA_PATH)

    if !isfile(DATA_PATH)
        error("No existe el archivo de datos: $(DATA_PATH)")
    end

    df = CSV.read(DATA_PATH, DataFrame; delim=',')

    target_col = "NObeyesdad"

    if !(target_col in names(df))
        error("No existe la columna objetivo $(target_col). Columnas detectadas: $(names(df))")
    end

    targets = Vector{String}(string.(df[!, target_col]))

    raw_inputs, original_feature_columns, expanded_feature_names = build_input_matrix(df, target_col)

    normalization_params = calculateMinMaxNormalizationParameters(raw_inputs)
    normalized_inputs = normalizeMinMax(raw_inputs, normalization_params)
    min_values, max_values = normalization_params
    mean_values = mean(raw_inputs, dims=1)

    describe_setup(df, target_col, raw_inputs, normalized_inputs, expanded_feature_names, min_values, max_values, mean_values)

    dataset = (normalized_inputs, targets)

    cv_indices = get_or_create_cv_indices(targets)

    println("\n====================")
    println("PARTE 4.2 - RESULTADOS")
    println("====================")

    ann_base = Dict{String,Any}(
        "numExecutions" => 3,
        "maxEpochs" => 250,
        "learningRate" => 0.01,
        "validationRatio" => 0.15,
        "maxEpochsVal" => 20
    )

    ann_topologies = [[8], [16], [32], [64], [8, 8], [16, 8], [32, 16], [64, 32]]
    ann_configs = [merge(copy(ann_base), Dict{String,Any}("topology" => topology)) for topology in ann_topologies]

    dome_nodes = [5, 8, 12, 16, 20, 25, 30, 40]
    dome_configs = [Dict{String,Any}("maximumNodes" => n) for n in dome_nodes]

    svm_configs = [
        Dict{String,Any}("kernel" => "linear", "C" => 0.1),
        Dict{String,Any}("kernel" => "linear", "C" => 1.0),
        Dict{String,Any}("kernel" => "rbf", "C" => 0.1, "gamma" => 0.1),
        Dict{String,Any}("kernel" => "rbf", "C" => 1.0, "gamma" => 0.1),
        Dict{String,Any}("kernel" => "rbf", "C" => 10.0, "gamma" => 0.1),
        Dict{String,Any}("kernel" => "poly", "C" => 1.0, "degree" => 2, "gamma" => 0.1, "coef0" => 1.0),
        Dict{String,Any}("kernel" => "poly", "C" => 1.0, "degree" => 3, "gamma" => 0.1, "coef0" => 1.0),
        Dict{String,Any}("kernel" => "sigmoid", "C" => 1.0, "gamma" => 0.1, "coef0" => 0.0)
    ]

    tree_depths = [2, 4, 6, 8, 10, 12]
    dt_configs = [Dict{String,Any}("max_depth" => d) for d in tree_depths]

    knn_ks = [1, 3, 5, 7, 9, 11]
    knn_configs = [Dict{String,Any}("n_neighbors" => k) for k in knn_ks]

    model_results = Dict{String,DataFrame}()

    model_results["ANN"] = evaluate_configs("ANN", :ANN, ann_configs, dataset, cv_indices)
    model_results["DoME"] = evaluate_configs("DoME", :DoME, dome_configs, dataset, cv_indices)
    model_results["SVM"] = evaluate_configs("SVM", :SVC, svm_configs, dataset, cv_indices)
    model_results["DecisionTree"] = evaluate_configs("DecisionTree", :DecisionTreeClassifier, dt_configs, dataset, cv_indices)
    model_results["kNN"] = evaluate_configs("kNN", :KNeighborsClassifier, knn_configs, dataset, cv_indices)

    println("\nResumen por tecnica (ordenado por F1):")
    best_rows = NamedTuple{
        (:Modelo, :MejorConfiguracion, :AccuracyMedia, :AccuracySTD, :ErrorMedia, :ErrorSTD, :RecallMedia, :RecallSTD, :SpecificityMedia, :SpecificitySTD, :PrecisionMedia, :PrecisionSTD, :NPVMedia, :NPVSTD, :F1Media, :F1STD),
        Tuple{String, String, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64}
    }[]

    for model_name in ["ANN", "DoME", "SVM", "DecisionTree", "kNN"]
        df_model = model_results[model_name]
        output_file = joinpath(OUTPUT_DIR, string("resultados_", lowercase(model_name), ".csv"))
        CSV.write(output_file, df_model)

        best = df_model[1, :]

        push!(best_rows, (
            Modelo = model_name,
            MejorConfiguracion = best.Configuracion,

            AccuracyMedia = Float64(best.AccuracyMedia),
            AccuracySTD = Float64(best.AccuracySTD),

            ErrorMedia = Float64(best.ErrorMedia),
            ErrorSTD = Float64(best.ErrorSTD),

            RecallMedia = Float64(best.RecallMedia),
            RecallSTD = Float64(best.RecallSTD),

            SpecificityMedia = Float64(best.SpecificityMedia),
            SpecificitySTD = Float64(best.SpecificitySTD),

            PrecisionMedia = Float64(best.PrecisionMedia),
            PrecisionSTD = Float64(best.PrecisionSTD),

            NPVMedia = Float64(best.NPVMedia),
            NPVSTD = Float64(best.NPVSTD),

            F1Media = Float64(best.F1Media),
            F1STD = Float64(best.F1STD)
        ))

        println("\n", model_name, ":")
        print_df = round_numeric_df(first(df_model, min(5, nrow(df_model))), digits=4)
        rename!(print_df, :Configuracion => :Conf, :AccuracyMedia => :Acc, :AccuracySTD => :Acc_sd, :ErrorMedia => :Err, :ErrorSTD => :Err_sd, :RecallMedia => :Rec, :RecallSTD => :Rec_sd, :SpecificityMedia => :Spec, :SpecificitySTD => :Spec_sd, :PrecisionMedia => :Prec, :PrecisionSTD => :Prec_sd, :NPVMedia => :NPV, :NPVSTD => :NPV_sd, :F1Media => :F1, :F1STD => :F1_sd)
        pretty_table(print_df; display_size=(-1, -1), show_subheader=false)
        println("\nGuardado: ", output_file)
    end

    best_df = DataFrame(best_rows)
    sort!(best_df, :F1Media, rev=true)

    best_file = joinpath(OUTPUT_DIR, "resumen_mejores_modelos.csv")
    CSV.write(best_file, best_df)

    labels = String.(best_df.Modelo)
    f1_values = Float64.(best_df.F1Media)
    acc_values = Float64.(best_df.AccuracyMedia)

    save_bar_chart_svg(
        labels,
        f1_values,
        "Comparativa de modelos - F1",
        "F1 medio",
        joinpath(OUTPUT_DIR, "grafica_comparativa_f1.svg")
    )

    save_bar_chart_svg(
        labels,
        acc_values,
        "Comparativa de modelos - Accuracy",
        "Accuracy medio",
        joinpath(OUTPUT_DIR, "grafica_comparativa_accuracy.svg")
    )

    println("\n==========================")
    println("Resumen mejores modelos")
    println("==========================")
    print_best_df = round_numeric_df(best_df, digits=4)
    rename!(print_best_df, :MejorConfiguracion => :Conf, :AccuracyMedia => :Acc, :AccuracySTD => :Acc_sd, :ErrorMedia => :Err, :ErrorSTD => :Err_sd, :RecallMedia => :Rec, :RecallSTD => :Rec_sd, :SpecificityMedia => :Spec, :SpecificitySTD => :Spec_sd, :PrecisionMedia => :Prec, :PrecisionSTD => :Prec_sd, :NPVMedia => :NPV, :NPVSTD => :NPV_sd, :F1Media => :F1, :F1STD => :F1_sd)
    pretty_table(print_best_df; display_size=(-1, -1), show_subheader=false)
    println("\n\nResumen guardado en: ", best_file)

    info_df = DataFrame(
        clave = [
            "fecha_ejecucion",
            "dataset",
            "num_patrones",
            "num_features_originales",
            "num_features_codificadas",
            "num_clases",
            "k_folds",
            "seed",
            "archivo_indices_cv"
        ],
        valor = [
            string(now()),
            DATA_PATH,
            string(nrow(df)),
            string(length(original_feature_columns)),
            string(length(expanded_feature_names)),
            string(length(unique(targets))),
            string(KFOLDS),
            string(SEED),
            INDICES_PATH
        ]
    )

    metadata_file = joinpath(OUTPUT_DIR, "metadata_experimento.csv")
    CSV.write(metadata_file, info_df)

    println("\n==========================")
    println("ARCHIVOS GENERADOS")
    println("==========================")
    println("- ", INDICES_PATH)
    println("- ", joinpath(OUTPUT_DIR, "normalizacion_minmax.csv"))
    println("- ", joinpath(OUTPUT_DIR, "resultados_ann.csv"))
    println("- ", joinpath(OUTPUT_DIR, "resultados_dome.csv"))
    println("- ", joinpath(OUTPUT_DIR, "resultados_svm.csv"))
    println("- ", joinpath(OUTPUT_DIR, "resultados_decisiontree.csv"))
    println("- ", joinpath(OUTPUT_DIR, "resultados_knn.csv"))
    println("- ", joinpath(OUTPUT_DIR, "resumen_mejores_modelos.csv"))
    println("- ", joinpath(OUTPUT_DIR, "grafica_comparativa_f1.svg"))
    println("- ", joinpath(OUTPUT_DIR, "grafica_comparativa_accuracy.svg"))
    println("- ", joinpath(OUTPUT_DIR, "metadata_experimento.csv"))

    println("\nEjecucion finalizada correctamente.")
end


run_exercise_4()