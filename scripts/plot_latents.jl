using DrWatson, Plots
@quickactivate "TMI&MD"

# maybe just temp code for now?  

include(srcdir("vis_results.jl"))


# =========================================== BR ===========================================
data_br = wload(datadir("sims", "brownian_data.jld2"))
models_br = wload(datadir("models", "brownian_models.jld2"))
ic_scores = models_br["ic_scores"]
score, ind = findmin(ic_scores)
best_model = models_br["models"][ind]
ts, Z, xs, ys, Y = data_br["ts"], best_model["z"], data_br["xs"],data_br["ys"], best_model["Y"]

pz, py = PlotLatents(ts, Z, xs, ys, -reshape(Y, (41,41)), lw=3)
display(pz)
display(py)

pmz = VisModel(best_model["Ξz"], "")
pmy = VisModel(best_model["Ξy"], "")
display(pmz)
display(pmy)


wsave(plotsdir("Brownian_Z.png"), pz)
wsave(plotsdir("Brownian_Y.png"), py)
wsave(plotsdir("Brownian_XiZ.png"), pmz)
wsave(plotsdir("Brownian_XiY.png"), pmy)
 
# ==========================================================================================


# =========================================== OU ===========================================
data_ou = wload(datadir("sims", "ou_data.jld2"))
models_ou = wload(datadir("models", "ou_models.jld2"))
ic_scores = models_ou["ic_scores"]
score, ind = findmin(ic_scores)
best_model = models_ou["models"][ind]
ts, Z, xs, Y = data_ou["ts"], best_model["z"], data_ou["xs"], best_model["Y"]

p = PlotLatents(ts, Z, xs, Y, lw=3)
display(p)

pmz = VisModel(best_model["Ξz"], "")
pmy = VisModel(best_model["Ξy"], "")
display(pmz)
display(pmy)

wsave(plotsdir("OU_Z&Y.png"), p)
wsave(plotsdir("OU_XiZ.png"), pmz)
wsave(plotsdir("OU_XiY.png"), pmy)
 
# ==========================================================================================


# =========================================== NP ===========================================
data_np = wload(datadir("sims", "np_data.jld2"))
models_np = wload(datadir("models", "np_models.jld2"))
ic_scores = models_np["ic_scores"]
score, ind = findmin(ic_scores)
best_model = models_np["models"][ind]
ts, Z, nmax, Y = data_np["ts"], best_model["z"], data_np["nmax"], best_model["Y"]
ns = 1:nmax

p = PlotLatents(ts, Z, ns, Y, lw=3)
display(p)

pmz = VisModel(best_model["Ξz"], "")
pmy = VisModel(best_model["Ξy"], "")
display(pmz)
display(pmy)


wsave(plotsdir("NP_Z&Y.png"), p)
wsave(plotsdir("NP_XiZ.png"), pmz)
wsave(plotsdir("NP_XiY.png"), pmy)
 

# ==========================================================================================


# =========================================== Neuron ===========================================
data_ne = wload(datadir("sims", "neuron_data3.jld2"))
models_ne = wload(datadir("models", "neuron_models3.jld2"))

# first system 
ic_scores1 = models_ne["ic_scores_1"]
score1, ind1 = findmin(ic_scores1)
best_model1 = models_ne["models_1"][ind1]
ts, Z1 = data_ne["ts"], best_model1["z"]

color = 1
p_learned1 = PlotLatents(Z1, lw=2, color=color)
p_true1 = PlotLatents(permutedims(data_ne["Xss_tr"][1], (2,1,3)), lw=2, color=color)
display(p_learned1)
display(p_true1)

pmz1 = VisModel(best_model1["Ξz"], "")
display(pmz1)

# second system 
ic_scores2 = models_ne["ic_scores_2"]
score2, ind2 = findmin(ic_scores2)
best_model2 = models_ne["models_2"][ind2]
ts, Z2 = data_ne["ts"], best_model2["z"]

p_learned2 = PlotLatents(Z2, lw=2, color=color)
p_true2 = PlotLatents(permutedims(data_ne["Xss_tr"][2], (2,1,3)), lw=2, color=color)
display(p_learned2)
display(p_true2)

pmz2 = VisModel(best_model2["Ξz"], "")
display(pmz2)


wsave(plotsdir("Neuron_1_XiZ.png"), pmz1)
wsave(plotsdir("Neuron_2_XiZ.png"), pmz2)

 
# ==========================================================================================