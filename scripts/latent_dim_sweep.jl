using DrWatson, Random
@quickactivate "TMI&MD"

# load tmi functions 
include(srcdir("tmi.jl"))

# load data 
# diffusion data 
#=
diff_data = wload(datadir("sims", "diff_data.jld2"))
P = diff_data["Ptr"]
=# 
# Brownian data 
br_data = wload(datadir("sims", "brownian_data.jld2"))
Xd_br = br_data["Xd"]
Xd_br_rs = zeros((size(Xd_br,1)*size(Xd_br,2), size(Xd_br,3)))
for i=1:size(Xd_br,3)
    Xd_br_rs[:,i] = reshape(Xd_br[:,:,i], (size(Xd_br,1)*size(Xd_br,2), 1))
end
Xd_br_rs = Xd_br_rs ./ sum(Xd_br_rs, dims=1)

#=
# OU data
ou_data = wload(datadir("sims", "ou_data.jld2"))
Xd_ou = ou_data["Xd"]

# NP data 
np_data = wload(datadir("sims", "np_data.jld2"))
Ns = np_data["Ns"]
Ns_n = Ns ./ sum(Ns, dims=1)
=# 
# Neuron data
neuron_data = wload(datadir("sims", "neuron_data3.jld2"))
P = neuron_data["Psms_tr"]
P1, P2 = abs.(P[1]), abs.(P[2])
P1 = P1 ./ sum(P1, dims=1)
P2 = P2 ./ sum(P1, dims=1)

# sweep latents 
Ks = collect(1:20)
qs_br, klds_br = latent_dim_scan(Xd_br_rs, Ks::Vector)
qs_ou, klds_ou = latent_dim_scan(Xd_ou, Ks::Vector)
qs_np, klds_np = latent_dim_scan(Ns_n, Ks::Vector)
qs_neuron1, klds_neuron1 = latent_dim_scan(P1, Ks, use_E=true)
qs_neuron2, klds_neuron2 = latent_dim_scan(P2, Ks, use_E=true)

# save data 
br_lats = @strdict Ks qs_br klds_br
wsave(datadir("latent_dim", "brownian_latents2.jld2" ), br_lats)

diff_lats = @strdict Ks qs_diff klds_diff
wsave(datadir("latent_dim", "diff_latents2.jld2" ), diff_lats)

ou_lats = @strdict Ks qs_ou klds_ou
wsave(datadir("latent_dim", "ou_latents2.jld2" ), ou_lats)

np_lats = @strdict Ks qs_np klds_np
wsave(datadir("latent_dim", "np_latents2.jld2" ), np_lats)

neuron_lats = @strdict Ks qs_neuron1 klds_neuron1 qs_neuron2 klds_neuron2
wsave(datadir("latent_dim", "neuron_latents2.jld2"), neuron_lats)