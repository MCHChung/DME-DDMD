using DrWatson , Plots

@quickactivate "TMI&MD"

default(dpi=300, fontfamily="computer modern", grid=false)
#scalefontsizes(1.2^2)

# load factorization schemes 
include(srcdir("nnmf.jl"))
include(srcdir("elPOD.jl"))

# load data 
diff_data = wload(datadir("sims", "diff_data.jld2"))
ou_data = wload(datadir("sims", "ou_data.jld2"))
np_data = wload(datadir("sims", "np_data2.jld2"))
# load models 
diff_models = wload(datadir("models", "diff_models2.jld2"))
ou_models = wload(datadir("models", "ou_models4.jld2"))
np_models = wload(datadir("models", "np_models2.jld2"))

# convenience function for getting best model from each run
function FindBestModel(data, models, featlabel::String, distlabel::String)
    # get best models 
    ic_scores = models["ic_scores"]
    _, ind = findmin(ic_scores)
    best_model = models["models"][ind]

    # load data 
    ts, xs, Xd = data["ts"], data[featlabel], data[distlabel]
    Xd = Xd ./ sum(Xd, dims=1) # normalize if not normalized
    z,Y,q = best_model["z"], best_model["Y"], best_model["q"]

    return (ts,xs,Xd), (z,Y,q)
end

# get best models
(ts_diff, xs_diff, Xd_diff), (z_diff,Y_diff,q_diff) = FindBestModel(diff_data, diff_models, "xs", "P")
(ts_ou, xs_ou, Xd_ou), (z_ou, Y_ou, q_ou) = FindBestModel(ou_data, ou_models, "xs", "Xd")
(ts_np, nmax, Xd_np), (z_np, Y_np, q_np) = FindBestModel(np_data, np_models, "nmax", "Ns")
xs_np = 1:nmax

# compute KLDs 
kld_diff = kld_loss(Xd_diff, q_diff)
kld_ou = kld_loss(Xd_ou, q_ou)
kld_np = kld_loss(Xd_np, q_np)

# for same latent space dim, how is reconstruction of other methods? 
Xd_diff_pod = elPOD(Xd_diff, 1)
Xd_ou_pod = elPOD(Xd_ou, 2)
Xd_np_pod = elPOD(Xd_np, 1)

Xd_diff_nnmf = nnmf(Xd_diff, 1, maxiters=100000, grad_thresh=1e-6)
Xd_ou_nnmf = nnmf(Xd_ou, 2, maxiters=100000, grad_thresh=1e-6)
Xd_np_nnmf = nnmf(Xd_np, 1, maxiters=100000, grad_thresh=1e-6)

# what does latent dim have to be to reach a comparable KLD as our method? 
function SweepKsForKLD(Ks, X::Array{Float64,2}; eps=1e-8, maxiters=100000, grad_thresh=1e-6)
    klds_nnmf = zeros(length(Ks))
    klds_pod = similar(klds_nnmf)
    
    for i=1:length(Ks)
        K = Ks[i]
        println("K => $(K)")
        q_pod = elPOD(X, K)
        q_nnmf = nnmf(X, K, maxiters=maxiters, grad_thresh=grad_thresh)[3]
        klds_nnmf[i] = kld_loss(X, q_nnmf)
        klds_pod[i] = kld_loss(X, q_pod)
    end

    return klds_nnmf, klds_pod 
end

Ks = 1:5
klds_nnmf_diff, klds_pod_diff = SweepKsForKLD(Ks, Xd_diff; eps=1e-8)
klds_nnmf_ou, klds_pod_ou = SweepKsForKLD(Ks, Xd_ou; eps=1e-8)
klds_nnmf_np, klds_pod_np = SweepKsForKLD(Ks, Xd_np; eps=1e-8)

compare_dict = @strdict Xd_diff_pod Xd_ou_pod Xd_np_pod Xd_diff_nnmf Xd_ou_nnmf Xd_np_nnmf klds_nnmf_diff klds_pod_diff klds_nnmf_ou klds_pod_ou klds_nnmf_np klds_pod_np 
wsave(datadir("sims", "fact_comparisons.jld2"), compare_dict)

# ============== KLDs ===============
# KLDs of our method 
our_klds_diff = wload(datadir("latent_dim", "diff_latents.jld2"))["klds_diff"]
our_klds_ou = wload(datadir("latent_dim", "ou_latents.jld2"))["klds_ou"]
our_klds_np = wload(datadir("latent_dim", "np_latents.jld2"))["klds_np"]

# make plots
ms , alpha = 7 , 0.7
#pkld_diff = hline([kld_diff], color=:red, ls=:dash, label=false, lw=3, size=(400,400))
pkld_diff = scatter(our_klds_diff, color=:red, ms=ms, alpha=alpha, label="Ours", size=(400,400))
scatter!(klds_pod_diff, label="el-POD", color=:black, ms=ms, alpha=alpha, marker=:utriangle)
scatter!(klds_nnmf_diff, label="NNMF", color=:blue, marker=:square,  ms=ms, alpha=alpha)
xlabel!("Latent dimension (K)")
ylabel!("KLD")
#wsave(plotsdir("klds_comp_diff.png"), pkld_diff)
display(pkld_diff)

#pkld_ou = hline([kld_ou], color=:red, ls=:dash, label=false, lw=3, size=(400,400))
pkld_ou = scatter(our_klds_ou, color=:red, ms=ms, alpha=alpha, label="Ours", size=(400,400))
scatter!(klds_pod_ou, label="el-POD", color=:black,  ms=ms, alpha=alpha, marker=:utriangle)
scatter!(klds_nnmf_ou, label="NNMF", color=:blue, marker=:square, ms=ms, alpha=alpha)
xlabel!("Latent dimension (K)")
ylabel!("KLD")
#wsave(plotsdir("klds_comp_ou.png"), pkld_ou)
display(pkld_ou)

#pkld_np = hline([kld_np], color=:red, ls=:dash, label=false, lw=3, size=(400,400))
pkld_np = scatter(our_klds_np, color=:red, ms=ms, alpha=alpha, label="Ours", size=(400,400))
scatter!(klds_pod_np, label="el-POD", color=:black,  ms=ms, alpha=alpha, marker=:utriangle)
scatter!(klds_nnmf_np, label="NNMF", color=:blue, marker=:square,  ms=ms, alpha=alpha)
xlabel!("Latent dimension (K)")
ylabel!("KLD")
#wsave(plotsdir("klds_comp_np.png"), pkld_np)
display(pkld_np)

p1 = heatmap(log10.(Xd_diff .+ 1e-6))
p2 = heatmap(log10.(elPOD(Xd_diff, 1) .+ 1e-6))
p12 = plot(p1,p2,layout=(1,2), size=(800,300))
display(p12)

