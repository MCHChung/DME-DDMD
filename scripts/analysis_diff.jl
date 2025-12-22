using DrWatson, Plots, LinearAlgebra
@quickactivate "TMI&MD"

# plot defaults 
default(dpi=300, grid=false, fontfamily="computer modern")
#scalefontsizes(1.2^2) # makes fonts larger for legibility

# load src
include(srcdir("pred_lats.jl"))
include(srcdir("vis_results.jl"))
include(srcdir("build_weight_mats.jl"))
include(srcdir("tmi.jl"))

# load data and models 
data = wload(datadir("sims", "diff_data.jld2"))
models = wload(datadir("models", "diff_models.jld2"))

# find the model with the best (lowest) score 
ic_scores = models["ic_scores"]
score, ind = findmin(ic_scores)
best_model = models["models"][ind]
# load data 
ts, xs, P = data["ts"], data["xs"], data["P"]
z,Y,q = best_model["z"], best_model["Y"], best_model["q"]

# plot of data 
numpts = 100
colors = cgrad(:roma, categorical=true, numpts)
inds = Int.(floor.(range(1, length(ts), numpts)))
p = plot(xs, P[:,inds[1]], color=colors[1], lw=3, label=nothing)
ylims!(0, 0.065)
for i=2:numpts
    plot!(xs, P[:, inds[i]], color=colors[i], lw=3, label=nothing)
    ylims!(0, 0.065)
end
plot!(aspect_ratio=50)
xlims!(-1.5,1.5)
xlabel!("x")
ylabel!("Density")
#wsave(plotsdir("Diff_DataPlot.png"), p)

#lats, preds = PlotResults((ts_tr, xs, log10.(Ptr .+ 1e-3)), (z,Y,log10.(q .+ 1e-3)))
lats, preds = PlotResults((ts, xs, P), (z,Y,q), "x", logtrans=true, eps=1e-4)
display(lats)
display(preds)
#wsave(plotsdir("Diff_lats.png"), lats)
#wsave(plotsdir("Diff_PvsQvsSVD.png"), preds)

Ξz = best_model["Ξz"]
Ξy = best_model["Ξy"]
px = VisModel(Ξz, "Ξz")
py = VisModel(Ξy, "Ξy")
display(px)
display(py)

# look 
N = [1 ; 2; 3; 4; 5]
zpred = PredTraj(N, Ξz, z[1,:], ts)
p = plot(ts, z, color=1 , lw=5, label="Z")
plot!(ts, zpred, lw=3, ls=:dash, color=:red, label="Z pred.")
plot!(size=(400,400))
ylims!(0,z[1,1]+1)
xlims!(-0.1,ts[end]+0.1)
xlabel!("Time")
ylabel!("Z")
display(p)
#wsave(plotsdir("Diff_zpred.png"), p)


# ========== Pred New Data ===========
data_test = wload(datadir("sims", "diff_test_data.jld2"))
Ps_new = data_test["Ps_new"]
x0s_new = data_test["x0s_new"]
zpreds = []
Ynews = []
Ξzs_new = []
qpreds = []
errs = []
t_end = 30
errs = zeros(length(x0s_new), length(ts[t_end:end]))

for i=1:length(x0s_new)
    # get inputs 
    x0 = x0s_new[i]
    xn = xs .- x0 
    θy = [xn xn.^2 xn.^3 xn.^4 xn.^5] # library 
    Ynew = θy*Ξy # new Ys 
    # build mats 
    wind = 11
    W = BuildWeightMat(ts[5:t_end], wind, 0)
    DW = BuildWeightMat(ts[5:t_end], wind, 1) 
    N = [1; 2 ; 3; 4; 5]
    # estimate initial Z and update nonzero model coefficients  
    Pnew = Ps_new[i]
    z_new, Ξz_new, q0 = get_z0_and_update_model(Pnew[:,5:t_end], Ynew, Ξz, N, DW, W, η=1e-1, tol=1e-6, λz=1e1, maxiters=150000)
    zpred = PredTraj(N, Ξz_new, z_new[end, :], ts[t_end:end]) # proj fwd using dynamics 
    qpred = exp.(-Ynew*zpred') # predicted dist 
    qpred = qpred ./ sum(qpred, dims=1)
    # compute relative absolute error for every timepoint 
    err = mean(abs, Pnew[:,t_end:end] - qpred, dims=1)./mean(abs, Pnew[:, t_end:end], dims=1)
    errs[i, :] = err
    # add data to arrays 
    push!(zpreds, zpred)
    push!(Ynews, Ynew)
    push!(Ξzs_new, Ξz_new)
    push!(qpreds, qpred)
    #push!(errs, err)
end

mean_forecast_err = vec(mean(errs, dims=1))
std_forecast_err = vec(std(errs, dims=1))

yticks = exp10.(-5:1.:0)
p = plot(ts[t_end:end], mean_forecast_err, ribbon=std_forecast_err/sqrt(10), yscale=:log10, label=false, 
color=:black, yticks=yticks, lw=3, size=(400,400), margins=5*Measures.mm)
ylims!(yticks[1], yticks[end])
xlabel!("Time")
ylabel!("Relative L1 Error")
#wsave(plotsdir("Diff_forecast_err.png"), p)


