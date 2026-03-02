using Plots, DrWatson 

@quickactivate "TMI&MD"

include(srcdir("vis_results.jl"))

default(dpi=300, fontfamily="computer modern")

function get_best_models(models, scores)
    best_models = []
    for i=1:length(models)
        scores_i = scores[i]
        models_i = models[i]
        _, ind = findmin(scores_i)
        push!(best_models, models_i[ind])
    end
    return best_models
end


rand_br_data = wload(datadir("models", "brownian_models_rand.jld2"))
rand_ou_data = wload(datadir("models", "ou_models_rand_2.jld2"))
rand_np_data = wload(datadir("models", "np_models_rand.jld2"))
rand_ne_data = wload(datadir("models", "neuron_models_rand.jld2"))


# ==================================== Brownian ==================================================
best_br_models = get_best_models(rand_br_data["models"], rand_br_data["ic_scores"])

ind = 1
display(VisModel(best_br_models[ind]["Ξy"], "Ξy, ind=$(ind)"))
display(VisModel(best_br_models[ind]["Ξz"], "Ξz, ind=$(ind)"))

Ξy_terms_br = zeros(size(best_br_models[ind]["Ξy"]))
Ξz_terms_br = zeros(size(best_br_models[ind]["Ξz"]))

for i=1:10
    Ξy = abs.(best_br_models[i]["Ξy"]) .> 0
    Ξz = abs.(best_br_models[i]["Ξz"]) .> 0
    
    Ξz_terms_br += Ξz/10 
    Ξy_terms_br += Ξy/10
end

p_model_y_br = VisModel(Ξy_terms_br, "Brownian Ξy Rand, % terms", clims=(0., 1.))
p_model_z_br = VisModel(Ξz_terms_br, "Brownian Ξz Rand, % terms", clims=(0., 1.))

wsave(plotsdir("brownian_models_y_rand.png"), p_model_y_br)
wsave(plotsdir("brownian_models_z_rand.png"), p_model_z_br)


# ==================================== OU ==================================================
best_ou_models = get_best_models(rand_ou_data["models"], rand_ou_data["ic_scores"])

ind = 10
display(VisModel(best_ou_models[ind]["Ξy"] * [0 1; 1 0], "Ξy, ind=$(ind)"))
display(VisModel(best_ou_models[ind]["Ξz"]* [0 1; 1 0], "Ξz, ind=$(ind)"))

Ξy_terms_ou = zeros(size(best_ou_models[ind]["Ξy"]))
Ξz_terms_ou = zeros(size(best_ou_models[ind]["Ξz"]))

for i=1:10
    Ξy = abs.(best_ou_models[i]["Ξy"]) .> 0
    Ξz = abs.(best_ou_models[i]["Ξz"]) .> 0
    # aligning latents to that in main text (swap dimensionality label 1 <-> 2)
    if iszero(Ξy[1,1])
        Ξy = Ξy * [0 1; 1 0]
        Ξz = Ξz * [0 1; 1 0]
        Ξz[:, 2] .= 0 
        Ξz[2,2] = 1
        Ξz[end, 2] = 1 
    end
    Ξz_terms_ou += Ξz/10 
    Ξy_terms_ou += Ξy/10
end

p_model_y_ou = VisModel(Ξy_terms_ou, "OU Ξy Rand, % terms", clims=(0., 1.))
p_model_z_ou = VisModel(Ξz_terms_ou, "OU Ξz Rand, % terms", clims=(0., 1.))


wsave(plotsdir("ou_models_y_rand.png"), p_model_y_ou)
wsave(plotsdir("ou_models_z_rand.png"), p_model_z_ou)


# ==================================== NP ==================================================
best_np_models = get_best_models(rand_np_data["models"], rand_np_data["ic_scores"])

ind = 1
display(VisModel(best_np_models[ind]["Ξy"], "Ξy, ind=$(ind)"))
display(VisModel(best_np_models[ind]["Ξz"], "Ξz, ind=$(ind)"))

Ξy_terms_np = zeros(size(best_np_models[ind]["Ξy"]))
Ξz_terms_np = zeros(size(best_np_models[ind]["Ξz"]))

for i=1:10
    Ξy = abs.(best_np_models[i]["Ξy"]) .> 0
    Ξz = abs.(best_np_models[i]["Ξz"]) .> 0
    
    Ξz_terms_np += Ξz/10 
    Ξy_terms_np += Ξy/10
end

p_model_y_np = VisModel(Ξy_terms_np, "NP Ξy Rand, % terms", clims=(0., 1.))
p_model_z_np = VisModel(Ξz_terms_np, "NP Ξz Rand, % terms", clims=(0., 1.))

wsave(plotsdir("np_models_y_rand.png"), p_model_y_np)
wsave(plotsdir("np_models_z_rand.png"), p_model_z_np)


# ==================================== Neurons ==================================================
best_ne_models1 = get_best_models(rand_ne_data["models_1"], rand_ne_data["ic_scores_1"])
best_ne_models2 = get_best_models(rand_ne_data["models_2"], rand_ne_data["ic_scores_2"])

ind = 2
display(VisModel(best_ne_models1[ind]["Ξz"], "Ξz_1, ind=$(ind)"))
display(VisModel(best_ne_models2[ind]["Ξz"], "Ξz_2, ind=$(ind)"))

Ξy_terms_np = zeros(size(best_np_models[ind]["Ξy"]))
Ξz_terms_np = zeros(size(best_np_models[ind]["Ξz"]))

for i=1:10
    Ξy = abs.(best_np_models[i]["Ξy"]) .> 0
    Ξz = abs.(best_np_models[i]["Ξz"]) .> 0
    
    Ξz_terms_np += Ξz/10 
    Ξy_terms_np += Ξy/10
end

p_model_y_np = VisModel(Ξy_terms_np, "NP Ξy Rand, % terms", clims=(0., 1.))
p_model_z_np = VisModel(Ξz_terms_np, "NP Ξz Rand, % terms", clims=(0., 1.))

wsave(plotsdir("np_models_y_rand.png"), p_model_y_np)
wsave(plotsdir("np_models_z_rand.png"), p_model_z_np)
