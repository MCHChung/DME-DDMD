using Plots, DrWatson 

@quickactivate "TMI&MD"

include(srcdir("nnmf.jl"))
include(srcdir("dynamic_nnmf.jl"))

br_data = wload(datadir("sims", "brownian_data.jld2"))
ou_data = wload(datadir("sims", "ou_data.jld2"))
np_data = wload(datadir("sims", "np_data.jld2"))
neuron_data = wload(datadir("sims", "neuron_data.jld2"))

Ks = collect(1:20)
epochs = 1000
rng = Random.default_rng()
Random.seed!(rng, 0)

# load data
# ======= brownian ========
Xbr = br_data["Xd"]
Xbr = reshape(Xbr, (size(Xbr,1)*size(Xbr,2), size(Xbr,3)))
Xbr = Xbr ./ sum(Xbr, dims=1)
# ======= OU =========
Xou = ou_data["Xd"]
Xou = Xou ./ sum(Xou, dims=1)
# ======= NP =========
Xnp = np_data["Ns"]
Xnp = Xnp ./ sum(Xnp, dims=1)
# ======= neuron =========
P = neuron_data["Psms_tr"]
Xne1, Xne2 = abs.(P[1]), abs.(P[2])
Xne1 = Xne1 ./ sum(Xne1, dims=1)
Xne2 = Xne2 ./ sum(Xne2, dims=1)


function sweep_dyn_nnmf(sys::Int, X, Ks; niter=1000, J=1, M=50, q=0.15, seed=0)
    klds = zeros(length(Ks))
    Ws = []
    Hs = []
    As = []
    for i=1:length(Ks)
        Kl = Ks[i]
        println("======= K => $(Kl) ============")
        W, H, A = dynamic_nmf(X, Kl; J=J, niter=niter, M=M, q=q, seed=seed)
        if sys==4
            kld = 0 
            num_exp = length(H)
            for j=1:num_exp
                Xpred_j = W*H[j]
                kld_j = kld_loss(X[:,:,j], Xpred_j)
                kld += kld_j 
            end
            klds[i] = kld/num_exp
        else
            Xpred = W*H
            kld = kld_loss(X, Xpred)
            klds[i] = kld
        end
        push!(Ws, W)
        push!(Hs, H)
        push!(As, A) 
    end

    return klds, Ws , Hs, As
end

niter=1000 
J=3 
M=50 
q=0.15 
seed=0 

println(" ========= Brownian ==========")
klds_br, Ws_br, Hs_br, As_br = sweep_dyn_nnmf(1, Xbr, Ks; niter=niter, J=J, M=M, q=q, seed=seed)
println(" ========= OU ==========")
klds_ou, Ws_ou, Hs_ou, As_ou = sweep_dyn_nnmf(2, Xou, Ks; niter=niter, J=J, M=M, q=q, seed=seed)
println(" ========= NP ==========")
klds_np, Ws_np, Hs_np, As_np = sweep_dyn_nnmf(3, Xnp, Ks; niter=niter, J=J, M=M, q=q, seed=seed)
println(" ========= Neuron 1 ==========")
klds_ne1, Ws_ne1, Hs_ne1, As_ne1 = sweep_dyn_nnmf(4, Xne1, Ks; niter=niter, J=J, M=M, q=q, seed=seed)
println(" ========= Neuron 2 ==========")
klds_ne2, Ws_ne2, Hs_ne2, As_ne2 = sweep_dyn_nnmf(4, Xne2, Ks; niter=niter, J=J, M=M, q=q, seed=seed)

klds_dict = @strdict klds_br klds_ou klds_np klds_ne1 klds_ne2
W_dict = @strdict Ws_br Ws_ou Ws_np Ws_ne1 Ws_ne2
H_dict = @strdict Hs_br Hs_ou Hs_np Hs_ne1 Hs_ne2
A_dict = @strdict As_br As_ou As_np As_ne1 As_ne2

datadict = Dict(
    "klds_dict" => klds_dict, 
    "Ws_dict" => W_dict,
    "Hs_dict" => H_dict,
    "As_dict" => A_dict
)

wsave(datadir("latent_dim", "dyn_nnmf.jld2"), datadict)

