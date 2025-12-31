using Plots, DrWatson, JLD2 , LinearAlgebra , Random, BenchmarkTools

@quickactivate "TMI&MD"

include(srcdir("tmi.jl"))
include(srcdir("build_weight_mats.jl"))

default(dpi=300, fontfamily="computer modern", grid=false)
#scalefontsizes(1.2^2)

BenchmarkTools.DEFAULT_PARAMETERS.samples = 10 

function run_scaling_tests(rng, Ns::Vector, Ts::Vector, Ks::Vector)
    combos = []
    mean_times = []
    std_times = []
   

    for K in Ks 
        for N in Ns 
            for T in Ts
                println("N => $(N), T => $(T), K => $(K)")
                #a = @benchmark tmi(rand(rng, $N, $T), K, tol=1e-6, maxiters=100000) seconds=10
                a = @benchmark tmi(X, $K, tol=1e-6, maxiters=100000) seconds=10 setup=(X=rand(rng, $N, $T))

                push!(combos, [N, T, K])
                push!(mean_times, time(mean(a)))
                push!(std_times, time(std(a)))
            end
        end
    end
    

    return combos, mean_times, std_times
end


rng = Random.default_rng()
Random.seed!(rng, 0)
Ns = [10, 50, 100, 500, 1000]
Ts = [10, 50, 100, 500, 1000]
Ks = collect(1:10)

combos, mean_times, std_times = run_scaling_tests(rng, Ns, Ts, Ks)

data_dict = @strdict combos mean_times std_times 
#wsave(datadir("scaling.jld2"), data_dict)

# plot log ntk vs log time 
ntk = [prod(i) for i in combos]
#nt = [i[1]*i[2] for i in combos]
y = log10.(mean_times) 
#b,m = [fill(5, length(y)) log10.(ntk)] \ y
b,m = [ones(length(y)) log10.(ntk)] \ y

y_fit = m*log10.(ntk) .+ b*ones(length(y))
m ,b = round(m, digits=2), round(b, digits=2)
r2 = 1 - sum(abs2, y-y_fit)/sum(abs2, y .- mean(y))
r2 = round(r2, digits=2)
p = scatter(log10.(ntk), y, color=:black, size=(550,500), label="Data", mz=[i[3] for i in combos], 
cbartitle="Latent dimension, K") 
plot!(log10.(ntk), y_fit, color=:red, lw=3, label="y=$(m)x+$(b), R2=$(r2)")
xlabel!("log10(NTK)")
ylabel!("log10(Mean Run Time (ns))")
display(p)

# plot log nt vs log time 
nt = [i[1]*i[2] for i in combos]
y = log10.(mean_times) 
b,m = [ones(length(y)) log10.(nt)] \ y

y_fit = m*log10.(nt) .+ b*ones(length(y))
m ,b = round(m, digits=2), round(b, digits=2)
r2 = 1 - sum(abs2, y-y_fit)/sum(abs2, y .- mean(y))
r2 = round(r2, digits=3)
p = scatter(log10.(nt), y, color=:black, size=(500,450), label="Data", alpha=0.5, ms=6) 
plot!(log10.(nt), y_fit, color=:red, lw=2.5, label="y=$(m)x+$(b), R2=$(r2)")
xlabel!("log10(NT)")
ylabel!("log10(Mean Run Time (ns))")
display(p)
wsave(plotsdir("nt_scaling.png"), p)

# log-log fits
y = log10.(mean_times) 
logN = [log10.(i[1]) for i in combos]
logT = [log10.(i[2]) for i in combos]
logK = [log10.(i[3]) for i in combos]
θ = [fill(5, length(logN)) logN logT logK] 
θ \ y