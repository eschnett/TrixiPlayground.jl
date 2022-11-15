module TrixiPlayground

using CairoMakie
using OrdinaryDiffEq
using Printf
using SixelTerm
using StaticArrays
using Trixi

################################################################################

struct ScalarWaveEquation1D <: Trixi.AbstractEquations{1,3} # NDIMS, NVARS
end

Trixi.varnames(_, ::ScalarWaveEquation1D) = ("ϕ", "ϕt", "ϕx")

function Trixi.flux(U, orientation::Integer, equation::ScalarWaveEquation1D)
    ϕ, ϕt, ϕx = U
    ifx(val) = ifelse(orientation == 1, val, 0)
    Fϕ = 0
    Fϕt = ifx(-ϕx)
    Fϕx = ifx(-ϕt)
    F = SVector(Fϕ, Fϕt, Fϕx)
    return F
end

function source_terms_scalar_wave(U, x, t, equations::ScalarWaveEquation1D)
    ϕ, ϕt, ϕx = U
    Sϕ = ϕt
    Sϕt = 0
    Sϕx = 0
    S = SVector(Sϕ, Sϕt, Sϕx)
    return S
end

have_constant_speed(::ScalarWaveEquation1D) = Val(true)
function max_abs_speed(u_ll, u_rr, orientation::Integer, equations::ScalarWaveEquation1D)
    λ_max = 1
    return λ_max
end

function initial_condition_sine(x, t, equations::ScalarWaveEquation1D)
    T = typeof(t)
    k = SVector(T(π))
    ω = sqrt(T(sum(k)))
    Uϕ = sin(ω * t) * sin(k[1] * x[1])
    Uϕt = ω * cos(ω * t) * sin(k[1] * x[1])
    Uϕx = k[1] * sin(ω * t) * cos(k[1] * x[1])
    U = SVector(Uϕ, Uϕt, Uϕx)
    return U
end

################################################################################

struct ScalarWaveEquation3D <: Trixi.AbstractEquations{3,5} # NDIMS, NVARS
end

@inline Trixi.varnames(_, ::ScalarWaveEquation3D) = ("ϕ", "ϕt", "ϕx", "ϕy", "ϕz")

@inline function Trixi.flux(U, orientation::Integer, equation::ScalarWaveEquation3D)
    ϕ, ϕt, ϕx, ϕy, ϕz = U
    ifx(val) = ifelse(orientation == 1, val, 0)
    ify(val) = ifelse(orientation == 2, val, 0)
    ifz(val) = ifelse(orientation == 3, val, 0)
    Fϕ = 0
    Fϕt = ifx(-ϕx) + ify(-ϕy) + ifz(-ϕz)
    Fϕx = ifx(-ϕt)
    Fϕy = ify(-ϕt)
    Fϕz = ifz(-ϕt)
    F = SVector(Fϕ, Fϕt, Fϕx, Fϕy, Fϕz)
    return F
end

@inline function source_terms_scalar_wave(U, x, t, equations::ScalarWaveEquation3D)
    ϕ, ϕt, ϕx, ϕy, ϕz = U
    Sϕ = ϕt
    Sϕt = 0
    Sϕx = 0
    Sϕy = 0
    Sϕz = 0
    S = SVector(Sϕ, Sϕt, Sϕx, Sϕy, Sϕz)
    return S
end

@inline have_constant_speed(::ScalarWaveEquation3D) = Val(true)
@inline function max_abs_speed(u_ll, u_rr, orientation::Integer, equations::ScalarWaveEquation3D)
    λ_max = 1
    return λ_max
end

@inline function initial_condition_sine(x, t, equations::ScalarWaveEquation3D)
    T = typeof(t)
    k = SVector(T(π), T(π), T(π))
    ω = sqrt(T(sum(k)))
    Uϕ = sin(ω * t) * sin(k[1] * x[1]) * sin(k[2] * x[2]) * sin(k[3] * x[3])
    Uϕt = ω * cos(ω * t) * sin(k[1] * x[1]) * sin(k[2] * x[2]) * sin(k[3] * x[3])
    Uϕx = k[1] * sin(ω * t) * cos(k[1] * x[1]) * sin(k[2] * x[2]) * sin(k[3] * x[3])
    Uϕy = k[2] * sin(ω * t) * sin(k[1] * x[1]) * cos(k[2] * x[2]) * sin(k[3] * x[3])
    Uϕz = k[3] * sin(ω * t) * sin(k[1] * x[1]) * sin(k[2] * x[2]) * cos(k[3] * x[3])
    U = SVector(Uϕ, Uϕt, Uϕx, Uϕy, Uϕz)
    return U
end

################################################################################

function main1d()
    println("Scalar Wave Equation in 1D with Trixi")

    println("Setting up problem...")
    equation = ScalarWaveEquation1D()

    xmin = -1.0
    xmax = +1.0
    mesh = TreeMesh((xmin,), (xmax,); initial_refinement_level=4, n_cells_max=10^4)

    NDEG = 3 # polynomial degree
    solver = DGSEM(NDEG, flux_central)

    semi = SemidiscretizationHyperbolic(mesh, equation, initial_condition_sine, solver;
                                        source_terms=source_terms_scalar_wave)

    # Create ODE problem with given time span
    tmin = 0.0
    tmax = 1.0
    ode = semidiscretize(semi, (tmin, tmax))

    println("Solving...")
    # OrdinaryDiffEq's `solve` method evolves the solution in time and executes the passed callbacks
    sol = solve(ode, SSPRK43())

    println("Plotting...")
    fig = Figure(; resolution=(1536, 864))

    var = "ϕ"
    ts = sol.t
    xs = PlotData1D(sol).x
    ϕs = Array{Float64,2}(undef, length(xs), length(ts))
    for j in 1:length(ts)
        t = ts[j]
        pdt = PlotData1D(sol[j], sol.prob.p)
        data = @view pdt.data[:, pdt[var].variable_id]
        @assert size(data) == (size(xs, 1),)
        ϕs[:, j] .= data[:]
    end

    Axis(fig[1, 1]; title="Scalar wave", xlabel="x", ylabel="t", aspect=(xmax - xmin) / (tmax - tmin))
    xlims!(xmin, xmax)
    ylims!(tmin, tmax)
    co = contourf!(fig[1, 1], xs, ts, ϕs; colormap=:plasma, levels=(-1.0:0.1:+1.0))
    Colorbar(fig[1, 2], co; label=var)

    save("scalarwave-1d.png", fig)
    display(fig)

    println("Done.")

    return
end

function main3d()
    println("Scalar Wave Equation in 3D with Trixi")

    println("Setting up problem...")
    equation = ScalarWaveEquation3D()

    println("    Creating mesh...")
    xmin = -1.0
    xmax = +1.0
    mesh = TreeMesh((xmin, xmin, xmin), (xmax, xmax, xmax); initial_refinement_level=4, n_cells_max=10^6)

    println("    Setting up solver...")
    NDEG = 3 # polynomial degree
    solver = DGSEM(NDEG, flux_central)

    println("    Setting up discretization...")
    semi = SemidiscretizationHyperbolic(mesh, equation, initial_condition_sine, solver;
                                        source_terms=source_terms_scalar_wave)

    println("    Discretizing initial conditions...")
    # Create ODE problem with given time span
    tmin = 0.0
    tmax = 1.0
    ode = semidiscretize(semi, (tmin, tmax))

    println("Solving...")
    # OrdinaryDiffEq's `solve` method evolves the solution in time and executes the passed callbacks

    # sol = solve(ode, SSPRK43())

    iter = 0
    start_walltime = time()
    old_walltime = start_walltime
    function output(simtime::Real, state; islast::Bool=false)
        local delta_walltime = 1
        local walltime = time()
        islast && (iter -= 1)
        if iter == 0 || islast || walltime - old_walltime ≥ delta_walltime
            @printf "    iteration: %4d   simulation time: %.3f   wall time: %.3f\n" iter simtime (walltime - start_walltime)
            old_walltime = walltime
        end
        iter += 1
        return nothing
    end

    integrator = init(ode, SSPRK43())
    for step in integrator
        output(step.t, step.u)
    end
    output(integrator.t, integrator.u; islast=true)

    sol = integrator.sol

    println("Plotting...")
    fig = Figure(; resolution=(1536, 864))

    var = "ϕ"
    ts = sol.t
    xs = PlotData1D(sol).x
    ϕs = Array{Float64,2}(undef, length(xs), length(ts))
    for j in 1:length(ts)
        t = ts[j]
        pdt = PlotData1D(sol[j], sol.prob.p; point=(0.5, 0.5, 0.5))
        data = @view pdt.data[:, pdt[var].variable_id]
        @assert size(data) == (size(xs, 1),)
        ϕs[:, j] .= data[:]
    end

    Axis(fig[1, 1]; title="Scalar wave", xlabel="x", ylabel="t", aspect=(xmax - xmin) / (tmax - tmin))
    xlims!(xmin, xmax)
    ylims!(tmin, tmax)
    co = contourf!(fig[1, 1], xs, ts, ϕs; colormap=:plasma, levels=(-1.0:0.1:+1.0))
    Colorbar(fig[1, 2], co; label=var)

    save("scalarwave-3d.png", fig)
    display(fig)

    println("Done.")

    return nothing
end

end
