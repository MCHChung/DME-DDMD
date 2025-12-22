using DrWatson, Plots, TensorOperations
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
data = wload(datadir("sims", "neuron_data3.jld2"))
models = wload(datadir("models", "neuron_models3.jld2"))

# find the model with the best (lowest) score 
ic_scores_1 = models["ic_scores_1"]
score1, ind1 = findmin(ic_scores_1)
best_model_1 = models["models_1"][ind1]

ic_scores_2 = models["ic_scores_2"]
score2, ind2 = findmin(ic_scores_2)
best_model_2 = models["models_2"][ind2]

# get data 
ts, P1, P2 = data["ts"], data["Psms_tr"][1], data["Psms_tr"][2]
P1 = abs.(P1 ./ sum(P1,dims=1))
P2 = abs.(P2 ./ sum(P2,dims=1))
E1,z1,Y1,q1 = best_model_1["E"], best_model_1["z"], best_model_1["Y"], best_model_1["q"]
E2,z2,Y2,q2 = best_model_2["E"], best_model_2["z"], best_model_2["Y"], best_model_2["q"]

# look at results 
#lats1, preds1 = PlotResults((ts, 1:size(P1[:,:,1],1), P1[:,:,1]), (z1,Y1,q1), "k-mers", logtrans=true, eps=1e-4)
#lats2, preds2 = PlotResults((ts, 1:size(P2[:,:,1],1), P2[:,:,1]), (z2,Y2,q2), "k-mers", logtrans=true, eps=1e-4)

#display(lats1)
#display(preds1)
#wsave(plotsdir("NP_lats.png"), lats)
#wsave(plotsdir("NP_PvsQvsSVD.png"), preds)

# get models 
Ξz1 = best_model_1["Ξz"]
Ξz2 = best_model_2["Ξz"]
px1 = VisModel(Ξz1, "Ξz1")
px2 = VisModel(Ξz2, "Ξz2")
display(px1)
display(px2)

# look at prediction 
ts = data["ts"]
N = [1 0; 0 1 ; 2 0; 1 1; 0 2]

zpred1 = similar(z1)
zpred2 = similar(z2)
for i=1:size(z1,3)
    # find trajs 
    zpred1_i = PredTraj(N, Ξz1, z1[1,:,i], ts; m=1)
    zpred2_i = PredTraj(N, Ξz2, z2[1,:,i], ts; m=1)
    # add to arrays
    zpred1[:,:,i] = zpred1_i
    zpred2[:,:,i] = zpred2_i
end


function plot_zs(zs::Array{Float64,3}; color=:black)
    p = plot()
    for i=1:size(zs,3)
        plot!(zs[:,1,i], zs[:,2,i], color=color, label=false)
        scatter!([zs[1,1,i]], [zs[1,2,i]], color=:magenta, label=false)
    end
    xlabel!("Z1")
    ylabel!("Z2")
    return p
end

function plot_zs(zs::Array{Float64,3}, zpred::Array{Float64,3})
    p = plot()
    for i=1:size(zs,3)
        plot!(zs[:,1,i], zs[:,2,i], color=:black, label=false)
        plot!(zpred[:,1,i], zpred[:,2,i], color=:red, ls=:dash, label=false)

    end
    xlabel!("Z1")
    ylabel!("Z2")
    return p
end

display(plot_zs(z1, zpred1))
display(plot_zs(z2, zpred2))

@tensor Cpred1[i,t,j] := Y1[i,k]*zpred1[t,k,j]
@tensor Cpred2[i,t,j] := Y2[i,k]*zpred2[t,k,j]

qpred1 = exp.(-E1 .- Cpred1)
qpred1 = qpred1 ./ sum(qpred1, dims=1)
qpred2 = exp.(-E2 .- Cpred2)
qpred2 = qpred2 ./ sum(qpred2, dims=1)

ind = 1
Plog1 = log10.(P1[:,:,ind] .+ 1e-3)
clims = (min(Plog1...), max(Plog1...))
p1 = heatmap(ts , 1:size(Plog1,1), Plog1, clims=clims)
vline!([ts[end]], lw=3, ls=:dash, color=:red, label=false)
p2 = heatmap(ts, 1:size(Plog1,1), log10.(qpred1[:,:,ind] .+ 1e-3), clims=clims)
p = plot(p1,p2, layout=(1,2), size=(800,300))
display(p)

Plog2 = log10.(P2[:,:,ind] .+ 1e-3)
clims = (min(Plog2...), max(Plog2...))
p1 = heatmap(ts ,1:size(Plog2,1),Plog2, clims=clims)
vline!([ts[end]], lw=3, ls=:dash, color=:red, label=false)
p2 = heatmap(ts, 1:size(Plog2,1), log10.(qpred2[:,:,ind] .+ 1e-3), clims=clims)
p = plot(p1,p2, layout=(1,2), size=(800,300))
display(p)

#wsave(plotsdir("NP_zpred.png"), p)

#=
qpred = exp.(-Y*zpred') 
qpred = qpred ./ sum(qpred, dims=1)

Plog = log10.(P .+ 1e-3)
clims = (min(Plog...), max(Plog...))
p1 = heatmap(ts ,ns, Plog, clims=clims)
vline!([ts[end]], lw=3, ls=:dash, color=:red, label=false)
p2 = heatmap(ts, ns, log10.(qpred .+ 1e-3), clims=clims)
p = plot(p1,p2, layout=(1,2), size=(800,300))
display(p)
=# 


# ======== Pred Evolution of New Data ===========
Ps_new1 = data["Psms_test"][1]
Ps_new2 = data["Psms_test"][2]

t_end = 50
errs1 = zeros(size(Ps_new1,3), length(ts[t_end:end]))
errs2 = zeros(size(Ps_new2,3), length(ts[t_end:end]))
z1s_new = []
z2s_new = []

function get_closest_z(P::Array{Float64,2}, Ptr::Array{Float64,3}; eps=1e-8)
    bestind = 1
    #bestdiff = mean(abs, P - Ptr[:,:,1])
    bestdiff = sum(P .* log.((P .+ eps) ./ (Ptr[:,:,1] .+ eps)))
    for i=1:size(Ptr,3)
        diff = bestdiff = sum(P .* log.((P .+ eps) ./ (Ptr[:,:,i] .+ eps)))
        if diff < bestdiff
            i = bestind
            bestdiff = diff
        end
    end

    return bestind
end

for i=1:size(Ps_new1,3)
    # build mats 
    wind = 11
    W = BuildWeightMat(ts[1:t_end], wind, 0)
    DW = BuildWeightMat(ts[1:t_end], wind, 1) 
    N = [1 0; 0 1 ; 2 0; 1 1; 0 2]
    # estimate initial Z and update nonzero model coefficients for both datasets 
    # 'experiment' 1
    Pnew1 = abs.(Ps_new1[:,:,i])
    Pnew1 = Pnew1 ./ sum(Pnew1, dims=1)
    ind1 = get_closest_z(Pnew1[:,1:t_end], P1[:,1:t_end,:])
    z_new1, Ξz_new1, q01 = get_z0_and_update_model(Pnew1[:,1:t_end], Y1, E1[:,:,1], Ξz1, N, DW, W, η=1e-2, tol=1e-6,
    λz=1e1, maxiters=150000, z = z1[1:t_end,:,ind1])
    # 'experiment 2'
    Pnew2 = abs.(Ps_new2[:,:,i])
    Pnew2 = Pnew2 ./ sum(Pnew2, dims=1)
    ind2 = get_closest_z(Pnew2[:,1:t_end], P2[:,1:t_end,:])
    z_new2, Ξz_new2, q02 = get_z0_and_update_model(Pnew2[:,1:t_end], Y2, E2[:,:,1], Ξz2, N, DW, W, η=1e-2, tol=1e-6, 
    λz=1e1, maxiters=150000, z = z2[1:t_end,:,ind2])
    
    zpred1 = PredTraj(N, Ξz1, z_new1[end, :], ts[t_end:end]) # proj fwd using dynamics 
    qpred1 = exp.(-Y1*zpred1' .-E1) # predicted dist 
    qpred1 = qpred1 ./ sum(qpred1, dims=1)
    zpred2 = PredTraj(N, Ξz2, z_new2[end, :], ts[t_end:end]) # proj fwd using dynamics 
    qpred2 = exp.(-Y2*zpred2' .- E2) # predicted dist 
    qpred2 = qpred2 ./ sum(qpred2, dims=1)
    # compute relative absolute error for every timepoint 
    err1 = mean(abs, Pnew1[:,t_end:end] - qpred1, dims=1)./mean(abs, Pnew1[:, t_end:end], dims=1)
    err2 = mean(abs, Pnew2[:,t_end:end] - qpred2, dims=1)./mean(abs, Pnew2[:, t_end:end], dims=1)
    errs1[i, :] = err1
    errs2[i, :] = err2
    push!(z1s_new, z_new1)
    push!(z2s_new, z_new2)

end

mean_forecast_err1 = vec(mean(errs1, dims=1))
std_forecast_err1 = vec(std(errs1, dims=1))
mean_forecast_err2 = vec(mean(errs2, dims=1))
std_forecast_err2 = vec(std(errs2, dims=1))

yticks = exp10.(-3:1.:0)
p1 = plot(ts[t_end:end], mean_forecast_err1, ribbon=std_forecast_err1/sqrt(5), yscale=:log10, label=false, 
color=:black, yticks=yticks, lw=3, size=(400,400), margins=5*Measures.mm)
ylims!(yticks[1], yticks[end])
xlabel!("Time")
ylabel!("Relative L1 Error")
display(p1)
#wsave(plotsdir("Neuron_forecast_sys1_err.png"), p1)

yticks = exp10.(-2:0.5:0)
p2 = plot(ts[t_end:end], mean_forecast_err2, ribbon=std_forecast_err2/sqrt(5), yscale=:log10, label=false, 
color=:black, yticks=yticks, lw=3, size=(400,400), margins=5*Measures.mm)
ylims!(yticks[1], yticks[end])
xlabel!("Time")
ylabel!("Relative L1 Error")
display(p2)
#wsave(plotsdir("Neuron_forecast_sys2_err.png"), p2)


# plot example from each 
ind = 4
zpred1 = PredTraj(N, Ξz1, z1s_new[ind][1, :], ts) # proj fwd using dynamics 
qpred1 = exp.(-Y1*zpred1' .-E1) # predicted dist 
qpred1 = qpred1 ./ sum(qpred1, dims=1)
zpred2 = PredTraj(N, Ξz2, z2s_new[ind][1, :], ts) # proj fwd using dynamics 
qpred2 = exp.(-Y2*zpred2' .- E2) # predicted dist 
qpred2 = qpred2 ./ sum(qpred2, dims=1)

P1_new = abs.(Ps_new1[:,:,ind])
P1_new = P1_new ./ sum(P1_new, dims=1)
P2_new = abs.(Ps_new2[:,:,ind])
P2_new = P2_new ./ sum(P2_new, dims=1)
p11 = heatmap(ts, 1:100, log10.(P1_new .+ 1e-4))
p21 = heatmap(ts, 1:100, log10.(qpred1[:,:,1] .+ 1e-4))
vline!(p21, [ts[t_end]], ls=:dash, color=:red, lw=3, label=false)
ps1 = plot(p11, p21, layout=(1,2), size=(800,300), margins=5*Measures.mm)
xlabel!("Time")
display(ps1)
p21 = heatmap(ts, 1:100, log10.(P2_new .+ 1e-4))
p22 = heatmap(ts, 1:100, log10.(qpred2[:,:,1] .+ 1e-4))
vline!(p22, [ts[t_end]], ls=:dash, color=:red, lw=3, label=false)
ps2 = plot(p21, p22, layout=(1,2), size=(800,300), margins=5*Measures.mm)
xlabel!("Time")
display(ps2)