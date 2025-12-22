using DrWatson , Plots

@quickactivate "TMI&MD"

include(srcdir("nnmf.jl"))

# load data 
data = wload(datadir("sims", "diff_data.jld2"))

P = data["P"]
P = P ./ sum(P, dims=1)

K = 1
A,B,q = nnmf(P, K, maxiters=50000, η=1e-5, loss_thresh=Inf, grad_thresh=1e-7, use_rand_init=false, eps=0.)

pa = heatmap(A)
pb = heatmap(B)
pab = plot(pa,pb, layout=(1,2), size=(800,300))
display(pab)


p1 = heatmap(log10.(P .+ 1e-6))
p2 = heatmap(log10.(q .+ 1e-6))
p12 = plot(p1,p2,layout=(1,2), size=(800,300))
display(p12)