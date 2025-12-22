using DrWatson, Random
@quickactivate "TMI&MD"

# Here you may include files from the source directory
include(srcdir("get_ens_data.jl"))

# ============= generate train data =============
# Params
β = 0.3
μ = 0.5
σ = 2e-1 
u0 = 0.
f(u,p,t) = @. β*(μ-u)
g(u,p,t) = σ
dt = 0.01
tspan = (0.0, 8.0) 

# set seed 
rng = Random.default_rng()
Random.seed!(rng, 0)

# generate anomalous trag
Xanom, ts = Get1DEnsData(f,g,u0,dt,tspan, rng; num_traj=10, noise_mag=0.05)

# upload regular data 
ou_data = wload(datadir("sims", "ou_data.jld2"))
Xreg = ou_data["X"]
Xd = ou_data["Xd"]

# concat w/ anomaly data 
X = vcat(Xreg, Xanom)

# estimate distribution using KDE
xb = 1.5
xs = -xb:0.01:xb
# train data 
Xd_anom = EstDist(X,xs)
Xd_anom = Xd_anom ./ sum(Xd_anom, dims=1)

# create and save data 
ou_anom_data = @strdict ts xs Xanom X Xd_anom Xd
#wsave(datadir("sims", "ou_anom_data.jld2"), ou_data)

p1 = heatmap(ts,xs, log10.(Xd .+ 1e-4))
plot!(ts, Xanom', color=:yellow, label=false)
display(p1)

p2 = heatmap(ts,xs, log10.(Xd .+ 1e-4))
p3 = heatmap(ts,xs, log10.(Xd_anom .+ 1e-4))
p23 = plot(p2,p3,layout=(1,2), size=(800,300))
display(p23)

p4 = plot(ts, Xreg[1:100:end, :]', color=:black, label=false)
plot!(ts, Xanom', color=:red, label=false)
display(p4)