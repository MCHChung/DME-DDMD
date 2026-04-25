using Plots, DrWatson , Measures , StatsPlots

@quickactivate "TMI&MD"

default(dpi=300, fontfamily="computer modern", grid=false)
#scalefontsizes(1.2^2)

br_dme = wload(datadir("models", "brownian_forecast.jld2"))
br_koop = wload(datadir("models", "koop_br_forecast_errs_all_combos_2026-3-14.jld2"))
br_sindy = wload(datadir("models", "sindy_br_forecast_errs_all_combos_2026-3-14.jld2"))
br_dnmf = wload(datadir("models", "dyn_nmf_brownian_forecast.jld2"))

ou_dme = wload(datadir("models", "ou_forecast.jld2"))
ou_koop = wload(datadir("models", "koop_ou_forecast_errs_all_combos_2026-3-14.jld2"))
ou_sindy = wload(datadir("models", "sindy_ou_forecast_errs_all_combos_2026-3-14.jld2"))
ou_dnmf = wload(datadir("models", "dyn_nmf_ou_forecast.jld2"))

np_dme = wload(datadir("models", "np_forecast.jld2"))
np_koop = wload(datadir("models", "koopman_ae_np_forecast_2026-3-12.jld2"))
np_sindy = wload(datadir("models", "sindy_np_forecast_errs_2026-3-14.jld2"))
np_dnmf = wload(datadir("models", "dyn_nmf_np_forecast.jld2"))

ne_dme = wload(datadir("models", "neuron_forecast.jld2"))
ne_koop = wload(datadir("models", "koopman_ae_ne_forecast_2026-4-3.jld2"))
ne_sindy = wload(datadir("models", "sindy_ne_forecast_errs_freeze_encdec_2026-4-2_2.jld2"))
ne_dnmf = wload(datadir("models", "dyn_nmf_ne_forecast.jld2"))


function plot_errs(errs_avg, errs_std;
    labels = ["Ours" "Dyn-NMF" "SINDy-AE" "Koop-AE"],
    colors = [:red :black :blue :green],
    alpha = 0.75,
    ylims = [1e-2, 2e1],
    legend = nothing
    )
    p = nothing
    if isnothing(legend)
        p = groupedbar(errs_avg, yerr=errs_std, color=colors, alpha=alpha, 
        labels=labels, yscale=:log10, margins= 5*Measures.mm, size=(400,325)) 
    else 
        p = groupedbar(errs_avg, yerr=errs_std, color=colors, alpha=alpha, 
        labels=labels, yscale=:log10, margins= 5*Measures.mm, size=(400,325), legend=legend)
    end
    ylims!(ylims...)
    ylabel!("Max. Relative L1 Error")
    return p
end

function errs_stats(errs)
    
    err_max_avg = zeros(length(errs))
    err_max_std = similar(err_max_avg)

    for i=1:length(errs)
        
        if typeof(errs[i]) <: Array{<:Real, 3}
            err_i = [max(errs[i][1,:,j]...) for j=1:size(errs[i],3)]
            err_max_avg[i] = mean(err_i)
            err_max_std[i] = std(err_i)
        else
            err_i = [max(errs[i][j,:]...) for j=1:size(errs[i],1)]
            err_max_avg[i] = mean(err_i)
            err_max_std[i] = std(err_i)
        end
    end

    return err_max_avg, err_max_std
end


# ================== Brownian ===================
errs_br_dme = errs_stats(br_dme["errs"])
errs_br_koop = errs_stats(br_koop["koop_forecast_errs_dyn"]["errs_dyn"])
errs_br_sindy = errs_stats(br_sindy["sindy_forecast_errs_dyn"]["errs_dyn"])
errs_br_dnmf = errs_stats(br_dnmf["errs"])

errs_br_avg = hcat(errs_br_dme[1], errs_br_dnmf[1], errs_br_sindy[1], errs_br_koop[1])
errs_br_std = hcat(errs_br_dme[2], errs_br_dnmf[2], errs_br_sindy[2], errs_br_koop[2])

pbr = plot_errs(errs_br_avg, errs_br_std/sqrt(10), alpha=0.75, ylims=[1e-2, 1e2])
#wsave(plotsdir("forecast_all_br_gbar.png"), pbr)

# ================== OU =======================
errs_ou_dme = errs_stats(ou_dme["errs"])
errs_ou_koop = errs_stats(ou_koop["koop_forecast_errs_dyn"]["errs_dyn"])
errs_ou_sindy = errs_stats(ou_sindy["sindy_forecast_errs_dyn"]["errs_dyn"])
errs_ou_dnmf = errs_stats(ou_dnmf["errs"])

errs_ou_avg = hcat(errs_ou_dme[1], errs_ou_dnmf[1], errs_ou_sindy[1], errs_ou_koop[1])
errs_ou_std = hcat(errs_ou_dme[2], errs_ou_dnmf[2], errs_ou_sindy[2], errs_ou_koop[2])

pou = plot_errs(errs_ou_avg, errs_ou_std/sqrt(10), alpha=0.75, ylims=[1e-2, 1e2])
#wsave(plotsdir("forecast_all_ou_gbar.png"), pou)

# ================== NP ============================

errs_np_dme = errs_stats(np_dme["errs"])
errs_np_koop = errs_stats(np_koop["errs"][1:4])
errs_np_sindy = errs_stats(np_sindy["errs"])
errs_np_dnmf = errs_stats(np_dnmf["errs"])

errs_np_avg = hcat(errs_np_dme[1], errs_np_dnmf[1], errs_np_sindy[1], errs_np_koop[1])
errs_np_std = hcat(errs_np_dme[2], errs_np_dnmf[2], errs_np_sindy[2], errs_np_koop[2])

pnp = plot_errs(errs_np_avg, errs_np_std/sqrt(10), alpha=0.75, ylims=[1e-2, 1e2])
#wsave(plotsdir("forecast_all_np_gbar.png"), pnp)

# ================== Neuron 1 ============================

errs_ne1_dme = errs_stats(ne_dme["errs1"])
errs_ne1_koop = errs_stats(ne_koop["errs1"])
errs_ne1_sindy = errs_stats(ne_sindy["errs1"][1:4])
errs_ne1_dnmf = errs_stats(ne_dnmf["errs1"])

errs_ne1_avg = hcat(errs_ne1_dme[1], errs_ne1_dnmf[1], errs_ne1_sindy[1], errs_ne1_koop[1])
errs_ne1_std = hcat(errs_ne1_dme[2], errs_ne1_dnmf[2], errs_ne1_sindy[2], errs_ne1_koop[2])

pne1 = plot_errs(errs_ne1_avg, errs_ne1_std/sqrt(5), alpha=0.75, ylims=[1e-2, 1e2], legend=false)
wsave(plotsdir("forecast_all_ne1_gbar_2026-4-2_2.png"), pne1)

# ================== Neuron 2 ============================

errs_ne2_dme = errs_stats(ne_dme["errs2"])
errs_ne2_koop = errs_stats(ne_koop["errs2"])
errs_ne2_sindy = errs_stats(ne_sindy["errs2"][1:4])
errs_ne2_dnmf = errs_stats(ne_dnmf["errs2"])

errs_ne2_avg = hcat(errs_ne2_dme[1], errs_ne2_dnmf[1], errs_ne2_sindy[1], errs_ne2_koop[1])
errs_ne2_std = hcat(errs_ne2_dme[2], errs_ne2_dnmf[2], errs_ne2_sindy[2], errs_ne2_koop[2])

pne2 = plot_errs(errs_ne2_avg, errs_ne2_std/sqrt(5), alpha=0.75, ylims=[1e-2, 1e2], legend=false)
wsave(plotsdir("forecast_all_ne2_gbar_2026-4-2_2.png"), pne2)
