# =============================================================================
# In-Class Activity: Model Fit and Counterfactuals
# ECON 6343 - Structural Econometrics
# =============================================================================

using Random, LinearAlgebra, Statistics, Optim, DataFrames, CSV, HTTP, ForwardDiff, FreqTables, Distributions

# =============================================================================
# 1. SETUP
# =============================================================================

# Load data
url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2026/master/ProblemSets/PS4-mixture/nlsw88t.csv"
df = CSV.read(HTTP.get(url).body, DataFrame)
X = [df.age df.white df.collgrad]
Z = hcat(df.elnwage1, df.elnwage2, df.elnwage3, df.elnwage4, 
         df.elnwage5, df.elnwage6, df.elnwage7, df.elnwage8)
y = df.occ_code

# Include likelihood function from PS4
include("../../ProblemSets/PS4-mixture/PS4_Ransom_source.jl")

# Starting values
θ_start = [.0403744; .2439942; -1.57132; .0433254; .1468556; -2.959103; 
           .1020574; .7473086; -4.12005; .0375628; .6884899; -3.65577; 
           .0204543; -.3584007; -4.376929; .1074636; -.5263738; -6.199197; 
           .1168824; -.2870554; -5.322248; 1.307477]

# Estimate
println("Estimating model...")
td = TwiceDifferentiable(b -> mlogit_with_Z(b, X, Z, y), θ_start; autodiff = :forward)
θ̂_optim = optimize(td, θ_start, LBFGS(), Optim.Options(g_tol=1e-5, iterations=100_000))
θ̂_mle = θ̂_optim.minimizer
H = Optim.hessian!(td, θ̂_mle)

println("Wage coefficient (γ): ", round(θ̂_mle[end], digits=4))

# =============================================================================
# 2. MODEL FIT
# =============================================================================

function plogit(θ, X, Z, J)
    # TODO: Fill in prediction function
    # α = θ[1:end-1]
    # γ = θ[end]
    # ...
    # return P
end

# Compute model fit
J = length(unique(y))
# P = plogit(θ̂_mle, X, Z, J)

# Create comparison table
# modelfit_df = DataFrame(...)

# =============================================================================
# 3. COUNTERFACTUAL 1: No Wage Effects
# =============================================================================

# Set γ = 0
# θ̂_cfl1 = ...

# Compute counterfactual predictions
# P_cfl1 = ...

# Add to table
# modelfit_df.cfl1_effect = ...
# modelfit_df.avg_wage = ...

# =============================================================================
# 4. COUNTERFACTUAL 2: 10% Wage Increase
# =============================================================================

# Increase all wages by 10%
# Z_cfl2 = Z .* 1.10
# P_cfl2 = ...

# =============================================================================
# 5. COUNTERFACTUAL 3: Targeted Subsidy
# =============================================================================

# 20% increase for occupations 5-8
# Z_cfl3 = copy(Z)
# Z_cfl3[:, 5:8] = ...

# =============================================================================
# 6. BOOTSTRAP (15 minutes)
# =============================================================================

Random.seed!(1234)
invH = inv(H)
invH_sym = (invH + invH') / 2
d = MvNormal(θ̂_mle, invH_sym)

B = 1000
cfl_bs = zeros(J, B)

println("Running bootstrap...")
for b = 1:B
    # TODO: Draw parameters
    # θ_draw = rand(d)
    
    # TODO: Compute baseline and counterfactual
    # ...
    
    # TODO: Store difference
    # cfl_bs[:, b] = ...
    
    if b % 200 == 0
        println("  $b/$B complete")
    end
end

# Compute CIs
# modelfit_df.ci_lower = ...
# modelfit_df.ci_upper = ...

println("\nFinal Results:")
println(modelfit_df)