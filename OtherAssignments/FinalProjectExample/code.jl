using Random, Distributions, LinearAlgebra, DataFrames, PrettyTables

Random.seed!(123)

struct Params
    N::Int
    M::Int
    L::Int
    λ::Matrix{Float64}
    log_w_base::Vector{Float64}
    ρ::Float64
    η::Vector{Float64}
    α_loc::Vector{Float64}
    α_occ::Float64
    β₁::Float64
    β₂::Float64
    γ_return::Vector{Float64}
    γ_new::Vector{Float64}
    δ::Float64
    α_major::Vector{Float64}
    α_college::Vector{Float64}
    β_college::Float64
    γ_college::Vector{Float64}
    δ_college::Float64
    β_discount::Float64
    p_birth::Vector{Float64}
    major_names::Vector{String}
end

function Params()
    Params(
        10_000, 5, 4,
        [0.50 0.30 0.15 0.05; 0.45 0.35 0.15 0.05; 0.35 0.30 0.25 0.10; 0.30 0.25 0.25 0.20; 0.25 0.25 0.25 0.25],
        [4.38, 4.32, 4.25, 4.17, 3.91],
        0.26,
        [0.18, 0.10, -0.05, -0.16],
        [0.0, -0.3, -0.5, -0.8],
        1.2, 0.05, -1.5,
        [-2.0, -2.2, -2.5, -2.8, -3.0],
        [-3.5, -3.7, -4.0, -4.3, -4.5],
        1.0,
        [0.0, -0.2, -0.3, -0.5, -0.8],
        [0.0, -0.4, -0.7, -1.0],
        -2.0,
        [-3.0, -3.2, -3.5, -3.8, -4.0],
        1.5, 0.9,
        [0.30, 0.25, 0.25, 0.20],
        ["CS", "Finance", "Engineering", "Business", "Education"]
    )
end

const p = Params()

wage(m, o, ℓ) = exp(p.log_w_base[m] + p.ρ * o + p.η[ℓ])

function moving_cost(m, ℓ₀, ℓ₁, ℓ₂)
    ℓ₂ == ℓ₁ && return 0.0
    ℓ₂ == ℓ₀ && ℓ₀ != ℓ₁ && return p.γ_return[m] + p.δ * (ℓ₂ != ℓ₀)
    return p.γ_new[m] + p.δ
end

college_moving_cost(m, ℓ₀, ℓ₁) = ℓ₁ == ℓ₀ ? 0.0 : p.γ_college[m] + p.δ_college

function period2_utility(m, o, ℓ₂, ℓ₀, ℓ₁, ε)
    mc = moving_cost(m, ℓ₀, ℓ₁, ℓ₂)
    return p.α_loc[ℓ₂] + p.α_occ * o + p.β₁ * log(wage(m, o, ℓ₂)) + p.β₂ * mc + ε
end

function compute_V2(m, ℓ₀, ℓ₁; λ_override=nothing)
    λ_use = isnothing(λ_override) ? p.λ[m, :] : λ_override[m, :]
    
    V = 0.0
    for ℓ_star in 1:p.L
        u_related = p.α_loc[ℓ_star] + p.α_occ + p.β₁ * log(wage(m, 1, ℓ_star)) + 
                   p.β₂ * moving_cost(m, ℓ₀, ℓ₁, ℓ_star)
        
        u_unrelated = [p.α_loc[ℓ] + p.β₁ * log(wage(m, 0, ℓ)) + 
                       p.β₂ * moving_cost(m, ℓ₀, ℓ₁, ℓ) for ℓ in 1:p.L]
        
        max_u = max(u_related, maximum(u_unrelated))
        V += λ_use[ℓ_star] * (max_u + log(exp(u_related - max_u) + sum(exp.(u_unrelated .- max_u))))
    end
    
    return V
end

function simulate_baseline()
    ℓ₀ = rand(Categorical(p.p_birth), p.N)
    ν = rand(Gumbel(0, 1), p.N, p.M, p.L)
    ε = rand(Gumbel(0, 1), p.N, 2, p.L)
    
    m_choice = zeros(Int, p.N)
    ℓ₁_choice = zeros(Int, p.N)
    
    for i in 1:p.N
        best_util = -Inf
        for m in 1:p.M, ℓ₁ in 1:p.L
            V2 = compute_V2(m, ℓ₀[i], ℓ₁)
            U = p.α_major[m] + p.α_college[ℓ₁] + p.β_college * college_moving_cost(m, ℓ₀[i], ℓ₁) + 
                p.β_discount * V2 + ν[i, m, ℓ₁]
            if U > best_util
                best_util = U
                m_choice[i] = m
                ℓ₁_choice[i] = ℓ₁
            end
        end
    end
    
    ℓ_star = [rand(Categorical(p.λ[m_choice[i], :])) for i in 1:p.N]
    o_choice = zeros(Int, p.N)
    ℓ₂_choice = zeros(Int, p.N)
    
    for i in 1:p.N
        m = m_choice[i]
        best_util = -Inf
        
        u = period2_utility(m, 1, ℓ_star[i], ℓ₀[i], ℓ₁_choice[i], ε[i, 2, ℓ_star[i]])
        if u > best_util
            best_util = u
            o_choice[i] = 1
            ℓ₂_choice[i] = ℓ_star[i]
        end
        
        for ℓ in 1:p.L
            u = period2_utility(m, 0, ℓ, ℓ₀[i], ℓ₁_choice[i], ε[i, 1, ℓ])
            if u > best_util
                best_util = u
                o_choice[i] = 0
                ℓ₂_choice[i] = ℓ
            end
        end
    end
    
    return (ℓ₀=ℓ₀, m=m_choice, ℓ₁=ℓ₁_choice, ℓ_star=ℓ_star, o=o_choice, ℓ₂=ℓ₂_choice, ν=ν, ε=ε)
end

function experiment1_uniform(data)
    λ_uniform = repeat([0.25 0.25 0.25 0.25], p.M, 1)
    
    m_choice = zeros(Int, p.N)
    ℓ₁_choice = zeros(Int, p.N)
    
    for i in 1:p.N
        best_util = -Inf
        for m in 1:p.M, ℓ₁ in 1:p.L
            V2 = compute_V2(m, data.ℓ₀[i], ℓ₁; λ_override=λ_uniform)
            U = p.α_major[m] + p.α_college[ℓ₁] + p.β_college * college_moving_cost(m, data.ℓ₀[i], ℓ₁) + 
                p.β_discount * V2 + data.ν[i, m, ℓ₁]
            if U > best_util
                best_util = U
                m_choice[i] = m
                ℓ₁_choice[i] = ℓ₁
            end
        end
    end
    
    ℓ_star = [rand(Categorical(λ_uniform[m_choice[i], :])) for i in 1:p.N]
    o_choice = zeros(Int, p.N)
    ℓ₂_choice = zeros(Int, p.N)
    
    for i in 1:p.N
        m = m_choice[i]
        best_util = -Inf
        
        u = period2_utility(m, 1, ℓ_star[i], data.ℓ₀[i], ℓ₁_choice[i], data.ε[i, 2, ℓ_star[i]])
        if u > best_util
            best_util = u
            o_choice[i] = 1
            ℓ₂_choice[i] = ℓ_star[i]
        end
        
        for ℓ in 1:p.L
            u = period2_utility(m, 0, ℓ, data.ℓ₀[i], ℓ₁_choice[i], data.ε[i, 1, ℓ])
            if u > best_util
                best_util = u
                o_choice[i] = 0
                ℓ₂_choice[i] = ℓ
            end
        end
    end
    
    return (m=m_choice, ℓ₁=ℓ₁_choice, o=o_choice, ℓ₂=ℓ₂_choice)
end

function experiment2_zero_costs(data)
    m_choice = zeros(Int, p.N)
    ℓ₁_choice = zeros(Int, p.N)
    
    for i in 1:p.N
        best_util = -Inf
        for m in 1:p.M, ℓ₁ in 1:p.L
            V2 = 0.0
            for ℓ_star in 1:p.L
                u_related = p.α_loc[ℓ_star] + p.α_occ + p.β₁ * log(wage(m, 1, ℓ_star))
                u_unrelated = [p.α_loc[ℓ] + p.β₁ * log(wage(m, 0, ℓ)) for ℓ in 1:p.L]
                max_u = max(u_related, maximum(u_unrelated))
                V2 += p.λ[m, ℓ_star] * (max_u + log(exp(u_related - max_u) + sum(exp.(u_unrelated .- max_u))))
            end
            
            U = p.α_major[m] + p.α_college[ℓ₁] + p.β_discount * V2 + data.ν[i, m, ℓ₁]
            if U > best_util
                best_util = U
                m_choice[i] = m
                ℓ₁_choice[i] = ℓ₁
            end
        end
    end
    
    ℓ_star = [rand(Categorical(p.λ[m_choice[i], :])) for i in 1:p.N]
    o_choice = zeros(Int, p.N)
    ℓ₂_choice = zeros(Int, p.N)
    
    for i in 1:p.N
        m = m_choice[i]
        best_util = -Inf
        
        u = p.α_loc[ℓ_star[i]] + p.α_occ + p.β₁ * log(wage(m, 1, ℓ_star[i])) + data.ε[i, 2, ℓ_star[i]]
        if u > best_util
            best_util = u
            o_choice[i] = 1
            ℓ₂_choice[i] = ℓ_star[i]
        end
        
        for ℓ in 1:p.L
            u = p.α_loc[ℓ] + p.β₁ * log(wage(m, 0, ℓ)) + data.ε[i, 1, ℓ]
            if u > best_util
                best_util = u
                o_choice[i] = 0
                ℓ₂_choice[i] = ℓ
            end
        end
    end
    
    return (m=m_choice, ℓ₁=ℓ₁_choice, o=o_choice, ℓ₂=ℓ₂_choice)
end

function experiment3_myopic(data)
    m_choice = zeros(Int, p.N)
    ℓ₁_choice = zeros(Int, p.N)
    
    for i in 1:p.N
        best_util = -Inf
        for m in 1:p.M, ℓ₁ in 1:p.L
            U = p.α_major[m] + p.α_college[ℓ₁] + p.β_college * college_moving_cost(m, data.ℓ₀[i], ℓ₁) + 
                data.ν[i, m, ℓ₁]
            if U > best_util
                best_util = U
                m_choice[i] = m
                ℓ₁_choice[i] = ℓ₁
            end
        end
    end
    
    ℓ_star = [rand(Categorical(p.λ[m_choice[i], :])) for i in 1:p.N]
    o_choice = zeros(Int, p.N)
    ℓ₂_choice = zeros(Int, p.N)
    
    for i in 1:p.N
        m = m_choice[i]
        best_util = -Inf
        
        u = period2_utility(m, 1, ℓ_star[i], data.ℓ₀[i], ℓ₁_choice[i], data.ε[i, 2, ℓ_star[i]])
        if u > best_util
            best_util = u
            o_choice[i] = 1
            ℓ₂_choice[i] = ℓ_star[i]
        end
        
        for ℓ in 1:p.L
            u = period2_utility(m, 0, ℓ, data.ℓ₀[i], ℓ₁_choice[i], data.ε[i, 1, ℓ])
            if u > best_util
                best_util = u
                o_choice[i] = 0
                ℓ₂_choice[i] = ℓ
            end
        end
    end
    
    return (m=m_choice, ℓ₁=ℓ₁_choice, o=o_choice, ℓ₂=ℓ₂_choice)
end

# Run all simulations
baseline = simulate_baseline()
exp1 = experiment1_uniform(baseline)
exp2 = experiment2_zero_costs(baseline)
exp3 = experiment3_myopic(baseline)

# Extract results
major_shares(data) = [mean(data.m .== m) for m in 1:p.M]
away_rate(data) = mean(data.ℓ₁ .!= data.ℓ₀)
related_rate(data) = mean(data.o .== 1)
moved_rate(data) = mean(data.ℓ₂ .!= data.ℓ₁)

# Build tables
df_majors = DataFrame(
    Major = p.major_names,
    Baseline = major_shares(baseline),
    Uniform = major_shares((ℓ₀=baseline.ℓ₀, m=exp1.m, ℓ₁=exp1.ℓ₁, o=exp1.o, ℓ₂=exp1.ℓ₂)),
    ZeroMC = major_shares((ℓ₀=baseline.ℓ₀, m=exp2.m, ℓ₁=exp2.ℓ₁, o=exp2.o, ℓ₂=exp2.ℓ₂)),
    Myopic = major_shares((ℓ₀=baseline.ℓ₀, m=exp3.m, ℓ₁=exp3.ℓ₁, o=exp3.o, ℓ₂=exp3.ℓ₂))
)

df_outcomes = DataFrame(
    Outcome = ["Away from home", "Related job rate", "Moved from college"],
    Baseline = [away_rate(baseline), related_rate(baseline), moved_rate(baseline)] .* 100,
    Uniform = [away_rate((ℓ₀=baseline.ℓ₀, m=exp1.m, ℓ₁=exp1.ℓ₁, o=exp1.o, ℓ₂=exp1.ℓ₂)), 
               related_rate((ℓ₀=baseline.ℓ₀, m=exp1.m, ℓ₁=exp1.ℓ₁, o=exp1.o, ℓ₂=exp1.ℓ₂)),
               moved_rate((ℓ₀=baseline.ℓ₀, m=exp1.m, ℓ₁=exp1.ℓ₁, o=exp1.o, ℓ₂=exp1.ℓ₂))] .* 100,
    ZeroMC = [away_rate((ℓ₀=baseline.ℓ₀, m=exp2.m, ℓ₁=exp2.ℓ₁, o=exp2.o, ℓ₂=exp2.ℓ₂)),
                related_rate((ℓ₀=baseline.ℓ₀, m=exp2.m, ℓ₁=exp2.ℓ₁, o=exp2.o, ℓ₂=exp2.ℓ₂)),
                moved_rate((ℓ₀=baseline.ℓ₀, m=exp2.m, ℓ₁=exp2.ℓ₁, o=exp2.o, ℓ₂=exp2.ℓ₂))] .* 100,
    Myopic = [away_rate((ℓ₀=baseline.ℓ₀, m=exp3.m, ℓ₁=exp3.ℓ₁, o=exp3.o, ℓ₂=exp3.ℓ₂)),
              related_rate((ℓ₀=baseline.ℓ₀, m=exp3.m, ℓ₁=exp3.ℓ₁, o=exp3.o, ℓ₂=exp3.ℓ₂)),
              moved_rate((ℓ₀=baseline.ℓ₀, m=exp3.m, ℓ₁=exp3.ℓ₁, o=exp3.o, ℓ₂=exp3.ℓ₂))] .* 100
)

df_changes = DataFrame(
    Experiment = ["Uniform", "ZeroMC", "Myopic"],
    ΔCS = collect(Vector(df_majors[1, 3:5]) .- df_majors[1, 2]) .* 100,
    ΔEducation = collect(Vector(df_majors[5, 3:5]) .- df_majors[5, 2]) .* 100,
    ΔAway = Vector(df_outcomes[1, 3:5]) .- df_outcomes[1, 2],
    ΔRelated = Vector(df_outcomes[2, 3:5]) .- df_outcomes[2, 2]
)

# Print LaTeX
println("% Table 1: Major Choice Distribution")
pretty_table(df_majors, backend=Val(:latex), formatters=ft_printf("%.4f", 2:5))

println("\n% Table 2: Key Outcomes (percent)")
pretty_table(df_outcomes, backend=Val(:latex), formatters=ft_printf("%.2f", 2:5))

println("\n% Table 3: Changes from Baseline (pp)")
pretty_table(df_changes, backend=Val(:latex), formatters=ft_printf("%.2f", 2:5))