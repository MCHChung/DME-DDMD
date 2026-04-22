using Plots, DrWatson, JLD2 

@quickactivate "TMI&MD"

include(srcdir("tmi.jl"))
include(srcdir("tmi_np.jl"))
include(srcdir("build_weight_mats.jl"))
include(srcdir("vis_results.jl"))
include(srcdir("pred_lats.jl"))
include(srcdir("smooth.jl"))

default(dpi=300, fontfamily="computer modern")
#scalefontsizes(1.2^2)

data = wload(datadir("exp_raw", "exp_particle_data.jld2"))

# look at mean over experimental trials 
X = mean(data["oneX"], dims=3)[:, :]
X = X ./ sum(X, dims=1)

ts = 0:size(X,2)-1
k = 1:size(X,1)
θy = [k k.^2 k.^3 k.^4 k.^5 log.(k) k.*log.(k) k.^2 .*log.(k) k.*log.(k).^2]

K = 1 
wind = 5
W = BuildWeightMat(ts ,wind ,0) 
DW = BuildWeightMat(ts ,wind ,1)
#DW, W = BuildDiscWeightMat(ts)
N = [0; 1 ;2; 3]
N = reshape(N, (length(N),1))

Ks = collect(1:10) 
_, klds = latent_dim_scan(X, Ks)
p_np_klds = scatter(log10.(klds), color=:black, label=false, ylims=(-3,1), ms=6, size=(450,400))
xlabel!("Latent Dimension")
ylabel!("log10(KLD)")
#wsave(plotsdir("particle_exp.png"), p_np_klds)

# warm start of the latents
(zhs , Yhs, _), _ = tmi(X, K, maxiters=250000, tol=1e-6, η=1e-3, use_rand_init=false, use_E=false)#, Y=reshape(k, (length(k),1)))

(z , Y, q) , (Ξz, Ξy), _ = tmi(X, K, W, DW, N, θy;
    use_E = false,
    normalize=true,
    use_rand_init=false,
    maxiters=1000000, # max number of iterations to update params 
    tol=1e-5, # termination condition if total L2 error of derivatives across iter < tol
    η=1e-5,
    cz=1e-4,#1e-8, 
    cy=1e-10,#1e-8,
    λy=1e1,
    λz=1e1,
    z=zhs, # warm start the latents
    Y=Yhs,
    #Ξz=Ξz, 
    #Ξy=Ξy,
    sparse_iter=200000
)

# functions for plotting 
function make_log_plot(X; normalize = true)
    Xn = copy(X)
    Xn = normalize ? Xn ./ sum(Xn, dims=1) : Xn
    p = plot(size=(450,400))
    colors = cgrad(:roma, categorical=true, size(X,2))
    ks = 1:size(X,1)
    for i=1:size(X,2)
        plot!(log.(ks), log.(X[:,i]), lw=2, color=colors[i], legend=false)
    end
    xlabel!("log(k)")
    ylabel!("log(X)")
    return p 
end

function plot_moments_vs_pred(X, q)
    k = 1:size(X,1)
    km = sum(k .* X, dims=1)[:]
    km_pred = sum(k .* q, dims=1)[:]
    km2 = sum(k.^2 .* X, dims=1)[:]
    km2_pred = sum(k.^2 .* q, dims=1)[:]
    km3 = sum(k.^3 .* X, dims=1)[:]
    km3_pred = sum(k.^3 .* q, dims=1)[:]

    p = scatter(min_max_scale(km_pred, km), min_max_scale(km, km), label="1st mom.", color=1, alpha=0.8, ms=5)
    scatter!(min_max_scale(km2_pred, km2), min_max_scale(km2, km2), label="2nd mom.", color=2, markershape=:utriangle, alpha=0.8, ms=5)
    scatter!(min_max_scale(km3_pred, km3), min_max_scale(km3, km3), label="3rd mom.", color=3, markershape=:square, alpha=0.8, ms=5)
    plot!(x -> x, color=:black, lw=4, ls=:dash, label=false, size=(400,350))
    xlabel!("Pred.")
    ylabel!("Data")

    return (km, km2, km3), (km_pred, km2_pred, km3_pred), p
end


p1 = make_log_plot(X)
ylims!(-9,0)
p2 = make_log_plot(q)
ylims!(-9,0)
p_log = plot(p1,p2,layout=(1,2), size=(900,400), margins=7*Measures.mm)
#wsave(plotsdir("particle_exp_log.png"), p_log)

wind = 5
plot(log.(k), log.(X[:,ind]), color=:black, lw=2, label="Data")
plot!(log.(k), log.(q[:,ind]), color=:red, lw=2, ls=:dash, label="Pred.")


p_model_z = VisModel(Ξz, "Ξz")
p_model_y = VisModel(Ξy, "Ξy")
#wsave(plotsdir("particle_exp_model_z.png"), p_model_z)
#wsave(plotsdir("particle_exp_model_y.png"), p_model_y)


Ξz_active = abs.(Ξz) .> 1e-2
Ξy_active = abs.(Ξy) .> 1e-2

Ξz = Ξz .* Ξz_active
Ξy = Ξy .* Ξy_active

eps = 8e-3
ind = 1
logX = log10.(X[:,:,ind] .+ eps)
clims = (min(logX...), max(logX...))
p1 = heatmap(logX, clims=clims, yticks = 1:12)
xlabel!("Time (min)")
ylabel!("k-mer")
p2 = heatmap(log10.(q[:,:,ind] .+ eps), clims=clims, yticks=1:12)
xlabel!("Time (min)")
phm = plot(p1,p2,layout=(1,2), size=(900,350), margins=7*Measures.mm)
#wsave(plotsdir("particle_exp_heatmap.png"), phm)

ind = 1
z_pred = PredTraj(N, Ξz[:,:,ind], z[1,:,ind], ts)
p1 = plot(z[:,1,ind], color=:black, lw=3, label="Data")
plot!(z_pred, color=:red, lw=3, ls=:dash, label="Pred.")
p2 = plot(Y, color=:black, label=false)
plot(p1,p2,layout=(1,2),size=(900,350))

plot()
for i=1:size(X,3)
    z_pred = PredTraj(N, Ξz2[:,:,i], z[1,:,i], ts)
    plot(z[:,1,i], color=:black, lw=3, label="Data, i=$(i)")
    plot!(z_pred, color=:red, lw=3, ls=:dash, label="Pred., i=$(i)")
end
plot!()

inds = 1:5
p = plot(X[inds,:]', color=:black, label=false)
plot!(q[inds,:]', color=:red, ls=:dash, label=false)




min_max_scale(x, y) = (x .- min(y...))/(max(y...) - min(y...))
min_max_scale(x) = (x .- min(x...))/(max(x...) - min(x...))

(km, km2, km3), (km_pred, km2_pred, km3_pred), p = plot_moments_vs_pred(X, q)
display(p)



function get_models_for_latents(X::AbstractMatrix, zhs, Y, Ξz,  W, DW, N; eps=8e-3)
   
    z, Ξz2, q = get_z0_and_update_model(X, Y, Ξz, N, DW, W; normalize=true,
        maxiters=1000000, 
        tol=1e-5, 
        η=1e-5,
        λz=1e1,
        z=zhs
    )

    # heatmap 
    logX = log10.(X .+ eps)
    clims = (min(logX...), max(logX...))
    p1 = heatmap(logX, clims=clims)
    p2 = heatmap(log10.(q .+ eps), clims=clims)
    phm = plot(p1,p2,layout=(1,2), size=(900,350))

    # trajs 
    ptr = plot(X[1:5,:]', color=:black, label=false)
    plot!(q[1:5,:]', color=:red, label=false, size=(450,400))

    # moments 
    (km, km2, km3), (km_pred, km2_pred, km3_pred), pm = plot_moments_vs_pred(X, q)

    return (z, Y, q, Ξz2), (phm, ptr, pm), (km, km2, km3), (km_pred, km2_pred, km3_pred)
end

function get_models_for_latents(X::Array{<: Real, 3}, zhs, Y, Ξz,  W, DW, N; eps=8e-3) 
    preds = []
    kms = []
    km_preds = []
    plots = []
    Ξzs = zeros(4,size(X,3))
    for i=1:size(X,3)
        pred, ps, km, km_pred = get_models_for_latents(X[:,:,i], zhs, Y, Ξz,  W, DW, N; eps=eps)
        Ξzs[:,i] = pred[4]
        push!(preds, pred)
        push!(kms, km)
        push!(km_preds, km_pred)
        push!(plots, ps)
    end

    return Ξzs, preds, plots, kms, km_preds
end

Y = reshape(log.(k), (length(k), 1))
(z, Y, q, Ξz2), (phm, ptr, pm), (km, km2, km3), (km_pred, km2_pred, km3_pred) = 
get_models_for_latents(X, zhs, Y, Ξz,  W, DW, N,; eps=8e-3) 

Ξzs_1, _, _, _, _ = get_models_for_latents(data["oneX"], zhs, Y, Ξz,  W, DW, N,; eps=8e-3)

function get_Xi_stats(Ξzs)
    rs = Ξzs[2, :] 
    taus = -Ξzs[2,:]./Ξzs[3,:]

    return mean(rs), std(rs)/sqrt(size(Ξzs,2)), mean(taus), std(taus)/sqrt(size(Ξzs,2))
end

r1_avg , r1_std, tau1_avg, tau1_std = get_Xi_stats(Ξzs_1)

#wsave(plotsdir("particle_exp_moments1.png"), pm)


# fit other  
# dilutions
twoX = data["twoX"]
twoX = twoX ./ sum(twoX, dims=1)

# RNA-Lip
Xpt3 = data["oneX_theta_pt3"]
Xpt3 = Xpt3 ./ sum(Xpt3, dims=1)

Xpt5= data["oneX_theta_pt5"]
Xpt5 = Xpt5 ./ sum(Xpt5, dims=1)


Ξzs_2x, _, _, _, _ = get_models_for_latents(twoX, zhs, Y, Ξz,  W, DW, N,; eps=8e-3)
r_2x_avg , r_2x_std, tau_2x_avg, tau_2x_std = get_Xi_stats(Ξzs_2x)

Ξzs_pt3, _, _, _, _ = get_models_for_latents(Xpt3, zhs, Y, Ξz,  W, DW, N,; eps=8e-3)
r_pt3_avg , r_pt3_std, tau_pt3_avg, tau_pt3_std = get_Xi_stats(Ξzs_pt3)

Ξzs_pt5, _, _, _, _ = get_models_for_latents(Xpt5, zhs, Y, Ξz,  W, DW, N,; eps=8e-3)
r_pt5_avg , r_pt5_std, tau_pt5_avg, tau_pt5_std = get_Xi_stats(Ξzs_pt5)


# plotting coeffs 
ms = 10
p1 = scatter([1,2,3,4], [r1_avg, r_2x_avg, r_pt3_avg, r_pt5_avg], color=:black, ms=ms, label=false,
 yerr = [r1_std, r_2x_std, r_pt3_std, r_pt5_std])
ylims!(0,0.32)
xlims!(0.5, 4.5)
ylabel!("r")

p2 = scatter([1,2,3,4], [tau1_avg, tau_2x_avg, tau_pt3_avg, tau_pt5_avg], color=:black, ms=ms, label=false,
 yerr = [tau1_std, tau_2x_std, tau_pt3_std, tau_pt5_std])
ylims!(0.,2.5)
xlims!(0.5, 4.5)
ylabel!("τ")

p12 = plot(p1,p2,layout=(1,2),size=(900,350), margins=5*Measures.mm)


