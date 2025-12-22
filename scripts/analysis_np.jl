using DrWatson, Plots
@quickactivate "TMI&MD"

# plot defaults 
default(dpi=300, grid=false, fontfamily="computer modern")
#scalefontsizes(1.2) # makes fonts larger for legibility

# load src
include(srcdir("pred_lats.jl"))
include(srcdir("vis_results.jl"))
include(srcdir("build_weight_mats.jl"))
include(srcdir("tmi.jl"))

# load data and models 
data = wload(datadir("sims", "np_data.jld2"))
models = wload(datadir("models", "np_models.jld2"))

# find the model with the best (lowest) score 
ic_scores = models["ic_scores"]
score, ind = findmin(ic_scores)
best_model = models["models"][ind]
# load data 
ts, ns, P = data["ts"], 1:data["nmax"], data["Ns"]
P = P ./ sum(P,dims=1)
z,Y,q = best_model["z"], best_model["Y"], best_model["q"]

# plot data 
numptcles = 5
colors = cgrad(:roma, categorical=true, numptcles)
p = plot(ts, P[1:numptcles,:]', color=[colors[1] colors[2] colors[3] colors[4] colors[5]], lw=4, 
labels = ["Monomer" "Dimer" "Trimer" "4-mer" "5-mer"], aspect_ratio=80)
ylims!(0,1)
xlims!(0,60)
xlabel!("Time")
ylabel!("k-mers")
#wsave(plotsdir("NP_DataPlot.png"), p)

lats, preds = PlotResults((ts, ns, P), (z,Y,q), "k-mers", logtrans=true, eps=1e-4)
display(lats)
display(preds)
#wsave(plotsdir("NP_lats.png"), lats)
#wsave(plotsdir("NP_PvsQvsSVD.png"), preds)

Ξz = best_model["Ξz"]
Ξy = best_model["Ξy"]
px = VisModel(Ξz, "Ξz")
py = VisModel(Ξy, "Ξy")
display(px)
display(py)


# look at prediction 
ts = data["ts"]
P = data["Ns"]
P = P ./ sum(P, dims=1)
N = [1; 2 ; 3; 4; 5]
zpred = PredTraj(N, Ξz, z[1,:], ts)
p = plot(ts, z, color=1 , lw=5, label="Z")
plot!(ts, zpred[1:length(ts)], lw=3, ls=:dash, color=:red, label="Z pred.")
plot!(size=(400,400))
ylims!(0,11)
xlims!(-5,105)
xlabel!("Time")
ylabel!("Z")
display(p)
#wsave(plotsdir("NP_zpred.png"), p)

qpred = exp.(-Y*zpred') 
qpred = qpred ./ sum(qpred, dims=1)

Plog = log10.(P .+ 1e-3)
clims = (min(Plog...), max(Plog...))
p1 = heatmap(ts ,ns, Plog, clims=clims)
vline!([ts[end]], lw=3, ls=:dash, color=:red, label=false)
p2 = heatmap(ts, ns, log10.(qpred .+ 1e-3), clims=clims)
p = plot(p1,p2, layout=(1,2), size=(800,300))
display(p)


# ======== Pred Evolution of New Data ===========
#Ns_new nmaxs_new
data_test = wload(datadir("sims", "np_test_data.jld2"))
Ns_new = data_test["Ns_new"]
nmaxs_new = data_test["nmaxs_new"]
zpreds = []
Ynews = []
Ξzs_new = []
qpreds = []
t_end = 200
errs = zeros(length(nmaxs_new), length(ts[t_end:end]))

for i=1:length(nmaxs_new)
    # get inputs 
    nmax_new = nmaxs_new[i]
    ns_new = 1:nmax_new
    θy = [ns_new ns_new.^2 ns_new.^3 ns_new.^4 ns_new.^5] # library 
    Ynew = θy*Ξy # new Ys 
    # build mats 
    wind = 11
    W = BuildWeightMat(ts[1:t_end], wind, 0)
    DW = BuildWeightMat(ts[1:t_end], wind, 1) 
    N = [1; 2 ; 3; 4; 5]
    # estimate initial Z and update nonzero model coefficients  
    Nnew = Ns_new[i]
    Nnew = Nnew ./ sum(Nnew, dims=1)
    z_new, Ξz_new, q0 = get_z0_and_update_model(Nnew[:, 1:t_end], Ynew, Ξz, N, DW, W, η=1e-3, tol=1e-9, λz=1e1, maxiters=150000)
    zpred = PredTraj(N, Ξz_new, z_new[t_end, :], ts[t_end:end]) # proj fwd using dynamics 
    qpred = exp.(-Ynew*zpred') # predicted dist 
    qpred = qpred ./ sum(qpred, dims=1)
    # compute relative absolute error for every timepoint 
    err = mean(abs, Nnew[:, t_end:end] - qpred, dims=1)./mean(abs, Nnew[:, t_end:end], dims=1)
    errs[i, :] = err
    # add data to arrays 
    push!(zpreds, zpred)
    push!(Ynews, Ynew)
    push!(Ξzs_new, Ξz_new)
    push!(qpreds, qpred)
end

mean_forecast_err = vec(mean(errs, dims=1))
std_forecast_err = vec(std(errs, dims=1))

# plot forecasting error
yticks = exp10.(-4:1.:0)
p = plot(ts[t_end:end], mean_forecast_err, ribbon=std_forecast_err/sqrt(10), yscale=:log10, label=false, 
color=:black, yticks=yticks, lw=3, size=(400,400), margins=5*Measures.mm)
ylims!(yticks[1], yticks[end])
xlims!(ts[t_end:end][1]-1, ts[t_end:end][end])
xlabel!("Time")
ylabel!("Relative L1 Error")
#wsave(plotsdir("NP_forecast_err.png"), p)
