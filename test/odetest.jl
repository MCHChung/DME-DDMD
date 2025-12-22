using DrWatson, DifferentialEquations
@quickactivate "TMI&MD"

function dudt(du, u, p, t)
    a , b , c = p
    du[1] = a*u[1] - b*u[1]*u[2]
    du[2] = -c*u[2] + b*u[1]*u[2]
end

rng = Random.default_rng()
Random.seed!(rng, 0)

p = [1, 0.05, 1] 
u0 = [15,5]
tspan = (0, 20)
dt = 0.01
prob = ODEProblem(dudt, u0, tspan,p)
soln = solve(prob, Tsit5(), saveat=dt)

ts = soln.t
X = Array(soln)