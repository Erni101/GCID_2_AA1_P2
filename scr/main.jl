using Random
using Statistics
using Printf
using Dates
using CSV
using DataFrames

include(joinpath(@__DIR__, "Codigo_profesor_ejercicio_2.jl"))

const DATA_PATH = joinpath(@__DIR__, "..", "data", "ObesityDataSet_raw_and_data_sinthetic.csv")
const OUTPUT_DIR = joinpath(@__DIR__, "..", "test", "resultados_ej4")

const SEED = 2026
const KFOLDS = 5


function config_to_string(cfg::Dict{String,Any})
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


function build_input_matrix(df::DataFrame, target_col::Symbol)
	feature_columns = [col for col in names(df) if col != target_col]

	input_blocks = Matrix{Float64}[]
	expanded_feature_names = String[]

	for col in feature_columns
		values = df[!, col]

		if eltype(values) <: Real
			push!(input_blocks, reshape(Float64.(values), :, 1))
			push!(expanded_feature_names, String(col))
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


function describe_setup(
	df::DataFrame,
	target_col::Symbol,
	raw_inputs::AbstractMatrix{<:Real},
	normalized_inputs::AbstractMatrix{<:Real},
	expanded_feature_names::Vector{String},
	min_values::AbstractMatrix{<:Real},
	max_values::AbstractMatrix{<:Real}
)
	targets = string.(df[!, target_col])
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
	println("   - Metricas: Accuracy, ErrorRate, Recall, Specificity, Precision, NPV, F1")

	println("\n4) Distribucion de clases")
	class_df = combine(groupby(DataFrame(clase=targets), :clase), nrow => :conteo)
	sort!(class_df, :clase)
	show(class_df, allrows=true, allcols=true)
	println()

	params_df = DataFrame(
		feature = expanded_feature_names,
		min = vec(min_values),
		max = vec(max_values)
	)
	CSV.write(joinpath(OUTPUT_DIR, "normalizacion_minmax.csv"), params_df)
	println("\nParametros de normalizacion guardados en: ", joinpath(OUTPUT_DIR, "normalizacion_minmax.csv"))
end


function evaluate_configs(
	model_name::String,
	model_type::Symbol,
	configs::Vector{Dict{String,Any}},
	dataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{<:Any,1}},
	cv_indices::Vector{Int64}
)
	rows = NamedTuple[]

	for (i, cfg) in enumerate(configs)
		println("\n[", model_name, "] Configuracion ", i, "/", length(configs), ": ", config_to_string(cfg))

		(acc, err, rec, spec, prec, npv, f1, _) = modelCrossValidation(model_type, cfg, dataset, cv_indices)

		push!(rows, (
			Modelo = model_name,
			Configuracion = config_to_string(cfg),
			AccuracyMedia = acc[1],
			AccuracySTD = acc[2],
			ErrorMedia = err[1],
			ErrorSTD = err[2],
			RecallMedia = rec[1],
			RecallSTD = rec[2],
			SpecificityMedia = spec[1],
			SpecificitySTD = spec[2],
			PrecisionMedia = prec[1],
			PrecisionSTD = prec[2],
			NPVMedia = npv[1],
			NPVSTD = npv[2],
			F1Media = f1[1],
			F1STD = f1[2]
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
	df = CSV.read(DATA_PATH, DataFrame)

	target_col = :NObeyesdad
	targets = string.(df[!, target_col])

	raw_inputs, original_feature_columns, expanded_feature_names = build_input_matrix(df, target_col)
	normalization_params = calculateMinMaxNormalizationParameters(raw_inputs)
	normalized_inputs = normalizeMinMax(raw_inputs, normalization_params)
	min_values, max_values = normalization_params

	describe_setup(df, target_col, raw_inputs, normalized_inputs, expanded_feature_names, min_values, max_values)

	dataset = (normalized_inputs, targets)
	cv_indices = crossvalidation(targets, Int64(KFOLDS))

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
	ann_configs = [merge(copy(ann_base), Dict("topology" => topology)) for topology in ann_topologies]

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
	best_rows = NamedTuple[]

	for model_name in ["ANN", "DoME", "SVM", "DecisionTree", "kNN"]
		df_model = model_results[model_name]
		output_file = joinpath(OUTPUT_DIR, string("resultados_", lowercase(model_name), ".csv"))
		CSV.write(output_file, df_model)

		best = df_model[1, :]
		push!(best_rows, (
			Modelo = model_name,
			MejorConfiguracion = best.Configuracion,
			AccuracyMedia = best.AccuracyMedia,
			F1Media = best.F1Media,
			RecallMedia = best.RecallMedia,
			PrecisionMedia = best.PrecisionMedia
		))

		println("\n", model_name, ":")
		show(first(df_model, min(5, nrow(df_model))), allrows=true, allcols=true)
		println("\nGuardado: ", output_file)
	end

	best_df = DataFrame(best_rows)
	sort!(best_df, :F1Media, rev=true)

	best_file = joinpath(OUTPUT_DIR, "resumen_mejores_modelos.csv")
	CSV.write(best_file, best_df)

	println("\n==========================")
	println("Resumen mejores modelos")
	println("==========================")
	show(best_df, allrows=true, allcols=true)
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
			"seed"
		],
		valor = [
			string(now()),
			DATA_PATH,
			string(nrow(df)),
			string(length(original_feature_columns)),
			string(length(expanded_feature_names)),
			string(length(unique(targets))),
			string(KFOLDS),
			string(SEED)
		]
	)
	CSV.write(joinpath(OUTPUT_DIR, "metadata_experimento.csv"), info_df)
end


if abspath(PROGRAM_FILE) == @__FILE__
	run_exercise_4()
end
