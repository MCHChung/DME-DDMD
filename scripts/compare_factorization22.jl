using DrWatson , Plots, Measures

@quickactivate "TMI&MD"

default(dpi=300, fontfamily="computer modern", grid=false)
#scalefontsizes(1.2^2)

# load factorization schemes 
include(srcdir("nnmf.jl"))
include(srcdir("elPOD.jl"))

# load data 
#diff_data = wload(datadir("sims", "diff_data.jld2"))
br_data = wload(datadir("sims", "brownian_data.jld2"))
ou_data = wload(datadir("sims", "ou_data.jld2"))
np_data = wload(datadir("sims", "np_data.jld2"))
neuron_data = wload(datadir("sims", "neuron_data3.jld2"))

# load models 
br_models = wload(datadir("models", "brownian_models.jld2"))
ou_models = wload(datadir("models", "ou_models.jld2"))
np_models = wload(datadir("models", "np_models.jld2"))
neuron_models = wload(datadir("models", "neuron_models3.jld2"))

# convenience function for getting best model from each run
function FindBestModel(data, models, featlabel::String, distlabel::String)
    #=
    # get best models 
    ic_scores = models["ic_scores"]
    _, ind = findmin(ic_scores)
    best_model = models["models"][ind]
    =# 
    if "ys" ∉ keys(data) && "Cs" ∉ keys(data) # OU, NP
        # get best models 
        ic_scores = models["ic_scores"]
        _, ind = findmin(ic_scores)
        best_model = models["models"][ind]
        # load data 
        ts, xs, Xd = data["ts"], data[featlabel], data[distlabel]
        Xd = Xd ./ sum(Xd, dims=1) # normalize if not normalized
        z,Y,q = best_model["z"], best_model["Y"], best_model["q"]
        return (ts,xs,Xd), (z,Y,q)
    elseif "ys" ∈ keys(data) # brownian
        # get best models 
        ic_scores = models["ic_scores"]
        _, ind = findmin(ic_scores)
        best_model = models["models"][ind]
        # load data 
        ts, xs, ys, Xd = data["ts"], data["xs"], data["ys"], data[distlabel]
        # reshape the matrix 
        Xd_rs = zeros((size(Xd,1)*size(Xd,2)), size(Xd,3))
        for i=1:size(Xd,3)
            Xd_rs[:,i] = reshape(Xd[:,:,i], (size(Xd,1)*size(Xd,2)), 1)
        end
        Xd_rs = Xd_rs ./ sum(Xd_rs, dims=1) # normalize if not normalized
        z,Y,q = best_model["z"], best_model["Y"], best_model["q"]
        return (ts,xs,ys,Xd_rs), (z,Y,q)
    else # neuron
        # load data 
        ts, Psms_tr = data["ts"], data[distlabel]
        # normalize
        P1 = abs.(Psms_tr[1])
        P1 = P1 ./ sum(P1,dims=1)
        P2 = abs.(Psms_tr[2])
        P2 = P2 ./ sum(P2,dims=1)

        ic_scores1 = models["ic_scores_1"]
        _, ind1 = findmin(ic_scores1)
        best_model1 = models["models_1"][ind1]
        ic_scores2 = models["ic_scores_2"]
        _, ind2 = findmin(ic_scores2)
        best_model2 = models["models_2"][ind2]

        E1,z1,Y1,q1 = best_model1["z"], best_model1["z"], best_model1["Y"], best_model1["q"]
        E2,z2,Y2,q2 = best_model2["z"], best_model2["z"], best_model2["Y"], best_model2["q"]

        return (ts, P1, P2), (E1,z1,Y1,q1), (E2,z2,Y2,q2)
    end
end

# get best models
(ts_br, xs_br, ys_br, Xd_br), (z_br,Y_br,q_br) = FindBestModel(br_data, br_models, "", "Xd")
(ts_ou, xs_ou, Xd_ou), (z_ou, Y_ou, q_ou) = FindBestModel(ou_data, ou_models, "xs", "Xd")
(ts_np, nmax, Xd_np), (z_np, Y_np, q_np) = FindBestModel(np_data, np_models, "nmax", "Ns")
(ts_neuron, Xd_n1, Xd_n2), (E1,z1,Y1,q1), (E2,z2,Y2,q2) = FindBestModel(neuron_data, neuron_models, "", "Psms_tr")
#xs_np = 1:nmax

#=
# compute KLDs 
kld_diff = kld_loss(Xd_br, q_br)
kld_ou = kld_loss(Xd_ou, q_ou)
kld_np = kld_loss(Xd_np, q_np)
=# 
# for same latent space dim, how is reconstruction of other methods? 
Xd_br_pod = elPOD(Xd_br, 1)
Xd_ou_pod = elPOD(Xd_ou, 2)
Xd_np_pod = elPOD(Xd_np, 1)
Xd_n1_pod = elPOD(Xd_n1, 2)
Xd_n2_pod = elPOD(Xd_n2, 2)

Xd_br_nnmf = nnmf(Xd_br, 1, maxiters=100000, grad_thresh=1e-6)
Xd_ou_nnmf = nnmf(Xd_ou, 2, maxiters=100000, grad_thresh=1e-6)
Xd_np_nnmf = nnmf(Xd_np, 1, maxiters=100000, grad_thresh=1e-6)
Xd_n1_nnmf = nnmf(Xd_n1, 2, maxiters=100000, grad_thresh=1e-6)
Xd_n2_nnmf = nnmf(Xd_n2, 2, maxiters=100000, grad_thresh=1e-6)

# what does latent dim have to be to reach a comparable KLD as our method? 
function SweepKsForKLD(Ks, X; eps=1e-8, maxiters=100000, grad_thresh=1e-6)
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

#=
function SweepKsForKLD(Ks, X::Array{Float64,3}; eps=1e-8, maxiters=100000, grad_thresh=1e-6)
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
=# 

Ks = 1:20
klds_nnmf_br, klds_pod_br = SweepKsForKLD(Ks, Xd_br; eps=1e-8)
klds_nnmf_ou, klds_pod_ou = SweepKsForKLD(Ks, Xd_ou; eps=1e-8)
klds_nnmf_np, klds_pod_np = SweepKsForKLD(Ks, Xd_np; eps=1e-8)
klds_nnmf_n1, klds_pod_n1 = SweepKsForKLD(Ks, Xd_n1; eps=1e-8)
klds_nnmf_n2, klds_pod_n2 = SweepKsForKLD(Ks, Xd_n2; eps=1e-8)
 
#compare_dict = @strdict Xd_diff_pod Xd_ou_pod Xd_np_pod Xd_diff_nnmf Xd_ou_nnmf Xd_np_nnmf klds_nnmf_diff klds_pod_diff klds_nnmf_ou klds_pod_ou klds_nnmf_np klds_pod_np 
compare_dict = @strdict Xd_br_pod Xd_ou_pod Xd_np_pod Xd_br_nnmf Xd_ou_nnmf Xd_np_nnmf klds_nnmf_br klds_pod_br klds_nnmf_ou klds_pod_ou klds_nnmf_np klds_pod_np 
wsave(datadir("sims", "fact_comparisons3.jld2"), compare_dict)

#=
klds_comp = wload(datadir("sims", "fact_comparisons2.jld2"))

klds_nnmf_br = klds_comp["klds_nnmf_br"]
klds_nnmf_ou = klds_comp["klds_nnmf_ou"]
klds_nnmf_np = klds_comp["klds_nnmf_np"]
klds_nnmf_n1 = klds_comp["klds_nnmf_n1"]
klds_nnmf_n2 = klds_comp["klds_nnmf_n2"]

klds_pod_br = klds_comp["klds_pod_br"]
klds_pod_ou = klds_comp["klds_pod_ou"]
klds_pod_np = klds_comp["klds_pod_np"]
klds_pod_n1 = klds_comp["klds_pod_n1"]
klds_pod_n2 = klds_comp["klds_pod_n2"]
=# 
# ============== KLDs ===============
# KLDs of our method 
#our_klds_diff = wload(datadir("latent_dim", "diff_latents.jld2"))["klds_diff"]
our_klds_br = wload(datadir("latent_dim", "brownian_latents2.jld2"))["klds_br"]
our_klds_ou = wload(datadir("latent_dim", "ou_latents2.jld2"))["klds_ou"]
our_klds_np = wload(datadir("latent_dim", "np_latents2.jld2"))["klds_np"]
our_klds_neuron1 = wload(datadir("latent_dim", "neuron_latents2.jld2"))["klds_neuron1"]
our_klds_neuron2 = wload(datadir("latent_dim", "neuron_latents2.jld2"))["klds_neuron2"]

# make plots
function make_kld_plot(klds_ours, klds_pod, klds_nnmf; ms = 7, alpha=0.7, lw=2, yscale=:log10, yticks=exp10.(-1:4), ylims=(1e-1, 1e4))
    p = plot(klds_ours, color=:red, lw=lw, alpha=alpha, label=false, size=(400,400), yscale=yscale, yticks=yticks)
    plot!(klds_pod, label=false, color=:black, lw=lw, alpha=alpha, marker=:utriangle, yticks=yticks)
    plot!(klds_nnmf, label=false, color=:blue, marker=:square, lw=lw, alpha=alpha, yticks=yticks)
    scatter!(klds_ours, color=:red, ms=ms, alpha=alpha, label="Ours", size=(400,400), yticks=yticks)
    scatter!(klds_pod, label="el-POD", color=:black, ms=ms, alpha=alpha, marker=:utriangle, yticks=yticks)
    scatter!(klds_nnmf, label="NNMF", color=:blue, marker=:square,  ms=ms, alpha=alpha, yticks=yticks)
    xlabel!("Latent dimension (K)")
    ylabel!("KLD")
    ylims!(ylims)
    return p
end

function make_kld_plot(kld_ours::Float64, K::Int, klds_pod, klds_nnmf; ms = 7, alpha=0.7, lw=2, yscale=:log10, yticks=exp10.(-1:4), ylims=(1e-1, 1e4))
    p = hline([kld_ours], color=:red, lw=3, ls=:dash, alpha=alpha, label="Ours, K=$(K)", size=(400,400), yscale=yscale, yticks=yticks)
    plot!(klds_pod, label=false, color=:black, lw=lw, alpha=alpha, marker=:utriangle, yticks=yticks)
    plot!(klds_nnmf, label=false, color=:blue, marker=:square, lw=lw, alpha=alpha, yticks=yticks)
    scatter!(klds_pod, label="el-POD", color=:black, ms=ms, alpha=alpha, marker=:utriangle, yticks=yticks)
    scatter!(klds_nnmf, label="NNMF", color=:blue, marker=:square,  ms=ms, alpha=alpha, yticks=yticks)
    xlabel!("Latent dimension (K)")
    ylabel!("KLD")
    ylims!(ylims)
    return p
end


pkld_br = make_kld_plot(our_klds_br[1], 1, klds_pod_br, klds_nnmf_br, ylims=(2e0, 5e2), yticks=exp10.(0.5:0.5:2.5))
pkld_ou = make_kld_plot(our_klds_ou[2], 2, klds_pod_ou, klds_nnmf_ou, ylims=(1e-1,1e3), yticks=exp10.(-1:0.5:3))
pkld_np = make_kld_plot(our_klds_np[1], 1, klds_pod_np, klds_nnmf_np, ylims=(5e-2,1e3))
pkld_n1 = make_kld_plot(our_klds_neuron1[2], 2, klds_pod_n1, klds_nnmf_n1, ylims=(1e-1,5e5))
pkld_n2 = make_kld_plot(our_klds_neuron2[2],2, klds_pod_n2, klds_nnmf_n2, ylims=(1e0,5e5))

display(pkld_br)
display(pkld_ou)
display(pkld_np)
display(pkld_n1)
display(pkld_n2)

#=
wsave(plotsdir("klds_comp_br.png"), pkld_br)
wsave(plotsdir("klds_comp_ou.png"), pkld_ou)
wsave(plotsdir("klds_comp_np.png"), pkld_np)
wsave(plotsdir("klds_comp_n1.png"), pkld_n1)
wsave(plotsdir("klds_comp_n2.png"), pkld_n2)
=# 

# make plots for Brownian system 
eps = 1e-5
Xd_br_log = log10.(Xd_br .+ eps) 
clims = (min(Xd_br_log...), max(Xd_br_log...))
p_br = heatmap(ts_br, 1:size(Xd_br,1), Xd_br_log, clims=clims, size=(440,400), margins=5*Measures.mm)
p_ours_br = heatmap(ts_br, 1:size(Xd_br,1), log10.(q_br .+ eps), clims=clims, size=(440,400), margins=5*Measures.mm)
p_pod_br = heatmap(ts_br, 1:size(Xd_br,1), log10.(Xd_br_pod .+ eps), clims=clims, size=(440,400), margins=5*Measures.mm)
p_nnmf_br = heatmap(ts_br, 1:size(Xd_br,1), log10.(Xd_br_nnmf[3] .+ eps), clims=clims, size=(440,400), margins=5*Measures.mm)

#=
wsave(plotsdir("Brownian_Dataplot.png"), p_br)
wsave(plotsdir("Brownian_Dataplot_Ours.png"), p_ours_br)
wsave(plotsdir("Brownian_Dataplot_POD.png"), p_pod_br)
wsave(plotsdir("Brownian_Dataplot_NNMF.png"), p_nnmf_br)
=# 


# make plots for neuron system 
eps = 1e-4 
ind = 1
Xd_n1_log = log10.(Xd_n1[:,:,ind] .+ eps) 
clims = (min(Xd_n1_log...), max(Xd_n1_log...))
p_n1 = heatmap(ts_neuron, 1:size(Xd_n1,1), Xd_n1_log, clims=clims, size=(400,400), margins=5*Measures.mm, title="Data")
p_ours_n1= heatmap(ts_neuron, 1:size(Xd_n1,1), log10.(q1[:,:,ind] .+ eps), clims=clims, size=(400,400), margins=5*Measures.mm, title="Ours")
p_pod_n1 = heatmap(ts_neuron, 1:size(Xd_n1,1), log10.(Xd_n1_pod[:,:,ind] .+ eps), clims=clims, size=(400,400), margins=5*Measures.mm, title="POD")
p_nnmf_n1 = heatmap(ts_neuron, 1:size(Xd_n1,1), log10.(Xd_n1_nnmf[3][:,:,ind] .+ eps), clims=clims, size=(400,400), margins=5*Measures.mm, title="NNMF")

#=
wsave(plotsdir("Neuron1_Dataplot.png"), p_n1)
wsave(plotsdir("Neuron1_Dataplot_Ours.png"), p_ours_n1)
wsave(plotsdir("Neuron1_Dataplot_POD.png"), p_pod_n1)
wsave(plotsdir("Neuron1_Dataplot_NNMF.png"), p_nnmf_n1)
=# 

