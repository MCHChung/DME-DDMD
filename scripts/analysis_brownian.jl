using DrWatson, Plots, Measures
@quickactivate "TMI&MD"

# plot defaults 
default(dpi=300, grid=false, fontfamily="computer modern")
#scalefontsizes(1.2) # makes fonts larger for legibility

# load src
include(srcdir("pred_lats.jl"))
include(srcdir("vis_results.jl"))
include(srcdir("get_ens_data.jl"))
include(srcdir("build_weight_mats.jl"))
include(srcdir("tmi.jl"))


# load data and models 
data = wload(datadir("sims", "brownian_data.jld2"))
models = wload(datadir("models", "brownian_models.jld2"))

# find the model with the best (lowest) score 
ic_scores = models["ic_scores"]
score, ind = findmin(ic_scores)
best_model = models["models"][ind]
# load data 
ts, xs, ys, Xd = data["ts"], data["xs"], data["ys"], data["Xd"]
z,Y,q = best_model["z"], best_model["Y"], best_model["q"]
Ξz = best_model["Ξz"]
Ξy = best_model["Ξy"]

# plot prob as heatmap 
#=
Plog = log10.(Xd .+ 1e-4)
clims = (min(Plog...), max(Plog...))
p = heatmap(ts, xs, Plog, label=false, aspect_ratio=5, axis=false, margins=5*Measures.mm, clims=clims)
xlabel!("Time")
ylabel!("x")
wsave(plotsdir("OU_DataPlot_Dist.png"), p)
=# 

#=
#lats, preds = PlotResults((ts_tr, xs, log10.(Xd_tr .+ 1e-3)), (z,Y,log10.(q .+ 1e-3)))
_, preds = PlotResults((ts, xs, Xd), (z,Y,q), "x", logtrans=true, eps=1e-4)
#display(lats)
display(preds)
#wsave(plotsdir("OU_lats.png"), lats)
#wsave(plotsdir("OU_PvsQvsSVD.png"), preds)

lats = plot(xs, Y, color=[1 :green], lw=7, label=["Y1" "Y2"], size=(400,400))
xlims!(-1.6, 1.6)
xlabel!("x")
ylabel!("Y")
#wsave(plotsdir("OU_lats.png"), lats)

Ξz = best_model["Ξz"]
Ξy = best_model["Ξy"]
px = VisModel(Ξz, "Ξz")
py = VisModel(Ξy, "Ξy")
display(px)
display(py)


# look at prediction 
ts = data["ts"]
Xd = data["Xd"]
N = [1 0; 0 1 ; 2 0 ; 1 1; 0 2]
zpred = PredTraj(N, Ξz, z[1,:], ts)
p = plot(ts, z, lw=7, label=["Z1" "Z2"], color = [1 :green])
plot!(ts, zpred[1:length(ts), :], lw=3, ls=:dash, color=:red, label=["Z pred." nothing], size=(400,400))
xlims!(-0.5, 8.)
xlabel!("Time")
ylabel!("Z")
display(p)
#wsave(plotsdir("OU_zpred.png"), p)

qpred = exp.(-Y*zpred') 
qpred = qpred ./ sum(qpred, dims=1)

Plog = log10.(Xd .+ 1e-3)
clims = (min(Plog...), max(Plog...))
p1 = heatmap(ts,xs, Plog, clims=clims)
#vline!([ts[end]], lw=3, ls=:dash, color=:red, label=false)
p2 = heatmap(ts, xs, log10.(qpred .+ 1e-3), clims=clims)
p = plot(p1,p2, layout=(1,2), size=(800,300))
display(p)
=# 


# ========== Pred New Data ===========

data_test = wload(datadir("sims", "brownian_test_data.jld2"))
Ps_new = data_test["Xds"]
#x0s_new = data_test["u0s"]
zpreds = []
Ynews = []
Ξzs_new = []
qpreds = []
errs = []
t_end = 50
errs = zeros(size(Ps_new,1), length(ts[t_end:end]))
#errs = zeros(size(x0s_new,2), length(ts))

for i=1:size(Ps_new,1)
    # get inputs  
    xs_rp = reshape(repeat(xs,1,length(ys))', (length(xs)*length(ys),1))
    ys_rp = reshape(repeat(reverse(ys),1,length(xs)), (length(xs)*length(ys),1))
    θy = [xs_rp ys_rp xs_rp.^2 xs_rp.*ys_rp ys_rp.^2] # library 
    Ynew = θy*Ξy # new Ys 
    # build mats 
    wind = 11
    W = BuildWeightMat(ts[1:t_end], wind, 0)
    DW = BuildWeightMat(ts[1:t_end], wind, 1) 
    N = [1 ;2 ;3; 4; 5]
    # estimate initial Z and update nonzero model coefficients
    Pnew = Ps_new[i]  
    Pnew_rs = zeros((size(Pnew,1)*size(Pnew,2), size(Pnew,3)))
    for i=1:size(Pnew,3)
        Pnew_rs[:,i] = reshape(Pnew[:,:,i], (size(Pnew,1)*size(Pnew,2),1))
    end
    Pnew_rs = Pnew_rs ./ sum(Pnew_rs, dims=1)  
    z_new, Ξz_new, q0 = get_z0_and_update_model(Pnew_rs[:,1:t_end], Ynew, Ξz, N, DW, W, η=1e-2, tol=1e-6, λz=1e1, maxiters=150000)
    zpred = PredTraj(N, Ξz_new, z_new[end, :], ts[t_end:end]) # proj fwd using dynamics 
    #zpred = PredTraj(N, Ξz_new, z_new[1, :], ts) # proj fwd using dynamics 

    qpred = exp.(-Ynew*zpred') # predicted dist 
    qpred = qpred ./ sum(qpred, dims=1)
    # compute relative absolute error for every timepoint 
    err = mean(abs, Pnew_rs[:,t_end:end] - qpred, dims=1)./mean(abs, Pnew_rs[:, t_end:end], dims=1)
    #err = mean(abs, Pnew_rs - qpred, dims=1)./mean(abs, Pnew_rs, dims=1)

    errs[i, :] = err

    # add data to arrays 
    push!(zpreds, zpred)
    push!(Ynews, Ynew)
    push!(Ξzs_new, Ξz_new)
    push!(qpreds, qpred)
    #push!(errs, err)
end

# get stats
mean_forecast_err = vec(mean(errs, dims=1))
std_forecast_err = vec(std(errs, dims=1))
# plot
yticks = exp10.(-1.5:0.5:0)
p = plot(ts[t_end:end], mean_forecast_err, ribbon=std_forecast_err/sqrt(10), yscale=:log10, label=false, 
color=:black, yticks=yticks, lw=3, size=(400,400), margins=5*Measures.mm)
ylims!(yticks[1], yticks[end])
xlabel!("Time")
ylabel!("Relative L1 Error")
#wsave(plotsdir("Brownian_forecast_err.png"), p)
 
# make plot of prediction 
ind = 8
eps = 1e-5
P1 = Ps_new[ind]  
P1_rs = zeros((size(P1,1)*size(P1,2), size(P1,3)))
for i=1:size(P1,3)
    P1_rs[:,i] = reshape(P1[:,:,i], (size(P1,1)*size(P1,2),1))
end
P1_rs = P1_rs ./ sum(P1_rs, dims=1)
P1_log = log10.(P1_rs .+ eps)
clims = (min(P1_log...), max(P1_log...))
p1 = heatmap(ts, 1:size(P1_rs,1), P1_log, clims=clims, size=(440,400))

# make prediction for entire trajectory 
qpred = hcat(P1_rs[:,1:t_end], qpreds[ind][:,2:end])
p2 = heatmap(ts, 1:size(P1_rs,1), log10.(qpred .+ eps), clims=clims, size=(440,400))
vline!([ts[t_end]], lw=3, ls=:dash, color=:red ,label=false)
p = plot(p1,p2, layout=(1,2), size= (800,350), margins=5*Measures.mm)
#wsave(plotsdir("Brownian_pred.png"), p)