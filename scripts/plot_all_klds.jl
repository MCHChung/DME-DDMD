using Plots, DrWatson 

@quickactivate "TMI&MD"

default(dpi=300, fontfamily="computer modern")
scalefontsizes(1.1/1.2)

br_dme = wload(datadir("latent_dim", "brownian_latents.jld2"))
ou_dme = wload(datadir("latent_dim", "ou_latents.jld2"))
np_dme = wload(datadir("latent_dim", "np_latents.jld2"))
ne_dme = wload(datadir("latent_dim", "neuron_latents.jld2"))
nnmf_pod = wload(datadir("sims", "nnmf_pod_klds.jld2"))
dyn_nnmf = wload(datadir("latent_dim", "dyn_nnmf.jld2"))["klds_dict"]
ae = wload(datadir("latent_dim", "ae_klds.jld2"))
ae_ne = wload(datadir("latent_dim", "ae_ne_klds.jld2"))
koop = wload(datadir("latent_dim", "klds_koopman_ae.jld2"))
sindy = wload(datadir("latent_dim", "klds_sindy_ae.jld2"))
sindy_ne = wload(datadir("latent_dim", "klds_sindy_ne.jld2"))


function make_kld_plot(klds, ylims; 
    colors=[:blue, :black, :green, :purple, :gold, :orange, :red], 
    markers=[:utriangle, :square, :diamond, :pentagon, :star, :hexagon, :circle],
    labels = ["NNMF", "el-POD", "Dyn. NNMF", "AE", "Koopman AE", "SINDy AE", "Edwin"],
    ms = 7,
    alpha=0.75,
    legend=false
    )
    p = plot(ylims=ylims, size=(450,400), legend=legend, yscale=:log10)
    for i=1:length(colors)
        #plot!(log10.(klds[i]), color=colors[i], label=false, alpha=alpha)
        #scatter!(log10.(klds[i]),  color=colors[i], label=labels[i], markershape=markers[i], ms=ms, alpha=alpha)
        plot!(klds[i], color=colors[i], label=false, alpha=alpha)
        scatter!(klds[i],  color=colors[i], label=labels[i], markershape=markers[i], ms=ms, alpha=alpha)
    end
    xlabel!("Latent Dimension (K)")
    ylabel!("KLD")
    return p
end

# ======= BROWNIAN ==========
klds_br_dme = br_dme["klds_br"]
klds_br_nnmf = nnmf_pod["klds_nnmf_br"]
klds_br_pod = nnmf_pod["klds_pod_br"]
klds_br_dyn_nnmf = dyn_nnmf["klds_br"]
klds_br_ae = ae["klds_br_ae"]
klds_br_koop = koop["klds_br"]
klds_br_sindy = sindy["klds_br_sindy"]

klds_br = [klds_br_nnmf, klds_br_pod, klds_br_dyn_nnmf, klds_br_ae, klds_br_koop, klds_br_sindy, klds_br_dme]

ylims = (1e0, 1e3)
pbr = make_kld_plot(klds_br, ylims, ms=7)

# =========== OU ========
klds_ou_dme = ou_dme["klds_ou"]
klds_ou_nnmf = nnmf_pod["klds_nnmf_ou"]
klds_ou_pod = nnmf_pod["klds_pod_ou"]
klds_ou_dyn_nnmf = dyn_nnmf["klds_ou"]
klds_ou_ae = ae["klds_ou_ae"]
klds_ou_koop = koop["klds_ou"]
klds_ou_sindy = sindy["klds_ou_sindy"]

klds_ou = [klds_ou_nnmf, klds_ou_pod, klds_ou_dyn_nnmf, klds_ou_ae, klds_ou_koop, klds_ou_sindy, klds_ou_dme]

ylims = (1e-1, 1e3)
pou = make_kld_plot(klds_ou, ylims)


# =========== NP ========
klds_np_dme = np_dme["klds_np"]
klds_np_nnmf = nnmf_pod["klds_nnmf_np"]
klds_np_pod = nnmf_pod["klds_pod_np"]
klds_np_dyn_nnmf = dyn_nnmf["klds_np"]
klds_np_ae = ae["klds_np_ae"]
klds_np_koop = koop["klds_np"]
klds_np_sindy = sindy["klds_np_sindy"]
#klds_np_sindy = sindy["klds_np"]

klds_np = [klds_np_nnmf, klds_np_pod, klds_np_dyn_nnmf, klds_np_ae, klds_np_koop, klds_np_sindy, klds_np_dme]

ylims = (1e-4, 1e4)
pnp = make_kld_plot(klds_np, ylims)


# =========== Neuron 1 ========
klds_ne1_dme = ne_dme["klds_neuron1"]
klds_ne1_nnmf = nnmf_pod["klds_nnmf_n1"]
klds_ne1_pod = nnmf_pod["klds_pod_n1"]
klds_ne1_dyn_nnmf = dyn_nnmf["klds_ne1"]
klds_ne1_ae = ae_ne["klds_ne1_ae"]
klds_ne1_koop = koop["klds_ne1"]
klds_ne1_sindy = sindy_ne["klds_ne1"]

klds_ne1 = [klds_ne1_nnmf, klds_ne1_pod, klds_ne1_dyn_nnmf, klds_ne1_ae, klds_ne1_koop, klds_ne1_sindy, klds_ne1_dme]

ylims = (1e-1, 1e5)
pne1 = make_kld_plot(klds_ne1, ylims)

# =========== Neuron 2 ========
klds_ne2_dme = ne_dme["klds_neuron2"]
klds_ne2_nnmf = nnmf_pod["klds_nnmf_n2"]
klds_ne2_pod = nnmf_pod["klds_pod_n2"]
klds_ne2_dyn_nnmf = dyn_nnmf["klds_ne2"]
klds_ne2_ae = ae_ne["klds_ne2_ae"]
klds_ne2_koop = koop["klds_ne2"]
klds_ne2_sindy = sindy_ne["klds_ne2"]

klds_ne2 = [klds_ne2_nnmf, klds_ne2_pod, klds_ne2_dyn_nnmf, klds_ne2_ae, klds_ne2_koop, klds_ne2_sindy, klds_ne2_dme]

ylims = (1e-1, 1e5)
pne2 = make_kld_plot(klds_ne2, ylims)


display(pbr)
display(pou)
display(pnp)
display(pne1)
display(pne2)

#=
wsave(plotsdir("klds_all_br_2026-4-17.png"), pbr)
wsave(plotsdir("klds_all_ou_2026-4-17.png"), pou)
wsave(plotsdir("klds_all_np_2026-4-17.png"), pnp)
wsave(plotsdir("klds_all_ne1_2026-4-17.png"), pne1)
wsave(plotsdir("klds_all_ne2_2026-4-17.png"), pne2)
=# 
