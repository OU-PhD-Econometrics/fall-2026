version 19.5
clear all
capture log close
log using "demo_uncorrelated.log", replace
set seed 12345
set obs 100000

* ---- True parameters ----
scalar y_int = 2.5
scalar alpha_true = 0.5
scalar f_load = 0.25
scalar unique_var = 0.5
scalar y_noise_var = 1.0
scalar d_noise_var = 0.1
scalar d_load = 0.1
scalar y_load = 0.5

* ---- Latent factor for correlation ----
gen F = rnormal()

* ---- Generate 5 W's and 5 X's, correlated via F ----
forvalues i=1/5 {
    gen W`i' = sqrt(unique_var)*rnormal()
}
forvalues i=1/5 {
    gen X`i' = f_load*F + sqrt(unique_var)*rnormal()
}

* ---- Unobserved and observed factors ----
* First-stage:
* True d depends on both observed (X) and unobserved (W) variables
gen u_d = sqrt(d_noise_var)*rnormal()
gen d = d_load*X1 + d_load*X2 + d_load*X3 + d_load*X4 + d_load*X5 + ///
        d_load*W1 + d_load*W2 + d_load*W3 + d_load*W4 + d_load*W5 + u_d

* ---- True model: y = αd + Xβ + Wγ + noise ----
* Second-stage:
gen u_y = sqrt(y_noise_var)*rnormal()
gen y = y_int + alpha_true*d + ///
        y_load*X1 + y_load*X2 + y_load*X3 + y_load*X4 + y_load*X5 + ///
        y_load*W1 + y_load*W2 + y_load*W3 + y_load*W4 + y_load*W5 + u_y

*-------------------------------------------------------------------------------
* Program to calibrate rho_k's
*-------------------------------------------------------------------------------
capture program drop calc_rho_k
program define calc_rho_k
    syntax varlist(min=2) [if] [in], Treatment(varname) [NOIsily SORT]
    
    marksample touse
    
    // Separate treatment variable from covariates
    local covars : list varlist - treatment
    
    // Step 1: Run selection regression (Treatment on Controls)
    if "`noisily'" != "" {
        regress `treatment' `covars' if `touse'
    }
    else {
        quietly regress `treatment' `covars' if `touse'
    }
    
    // Step 2: Get full predicted index
    tempvar full_index
    quietly predict `full_index' if `touse', xb
    
    // Step 3: Store results in temporary frame or matrices
    tempname results
    matrix `results' = J(`: word count `covars'', 3, .)
    matrix colnames `results' = coefficient sd_k rho_k
    
    local i = 1
    foreach k of local covars {
        
        // Get coefficient for W_1k
        local pi_k = _b[`k']
        
        // Calculate sqrt(Var(pi_k * W_1k)) - the numerator
        tempvar index_k
        quietly gen `index_k' = `pi_k' * `k' if `touse'
        quietly summarize `index_k' if `touse'
        local sd_k = sqrt(r(Var))
        
        // Calculate sqrt(Var(pi_-k' * W_-k)) - the denominator
        tempvar index_minus_k
        quietly gen `index_minus_k' = `full_index' - `pi_k' * `k' if `touse'
        quietly summarize `index_minus_k' if `touse'
        local sd_minus_k = sqrt(r(Var))
        
        // Calculate rho_k
        local rho_k = `sd_k' / `sd_minus_k'
        
        // Store in matrix
        matrix `results'[`i', 1] = `pi_k'
        matrix `results'[`i', 2] = `sd_k'
        matrix `results'[`i', 3] = `rho_k'
        
        // Clean up temporary variables
        drop `index_k' `index_minus_k'
        
        local i = `i' + 1
    }
    
    // Step 4: Create dataset with results and optionally sort
    preserve
    clear
    svmat `results', names(col)
    gen varname = ""
    local i = 1
    foreach k of local covars {
        replace varname = "`k'" if _n == `i'
        local i = `i' + 1
    }
    
    if "`sort'" != "" {
        gsort -rho_k
    }
    
    // Step 5: Display results
    di ""
    di as text "{hline 70}"
    di as text "Rho_k Calculations: Relative Importance of Each Covariate"
    if "`sort'" != "" {
        di as text "(sorted by rho_k in descending order)"
    }
    di as text "{hline 70}"
    di as text "Covariate" _col(30) "Coefficient" _col(45) "Rho_k"
    di as text "{hline 70}"
    
    forvalues i = 1/`=_N' {
        local k = varname[`i']
        local pi_k = coefficient[`i']
        local rho_k = rho_k[`i']
        di as text "`k'" _col(30) as result %9.4f `pi_k' _col(45) as result %9.4f `rho_k'
    }
    
    di as text "{hline 70}"
    di as text "Note: Rho_k measures importance of variable k relative to all other"
    di as text "      observed covariates in the selection equation."
    di as text "      Compare breakdown point r_X^bp against these rho_k values."
    di as text "{hline 70}"
    di ""
    
    restore
    
end


*-------------------------------------------------------------------------------
* Analysis: short, medium, and long regressions and regsensitivity bounds
*-------------------------------------------------------------------------------
local covars X1 X2 X3 X4 X5
* basic regressions
regress y d       
regress y d `covars'
* infeasible regression
regress y d `covars' W1-W5

* calibrating rho_k's
reg d `covars'
calc_rho_k d `covars', t(d) noisily sort

* regsensitivity
regsensitivity breakdown y d `covars', compare(`covars')
regsensitivity breakdown y d `covars', compare(`covars') cbar(0(.1)1)
//regsensitivity breakdown y d `covars', compare(`covars') rybar(=rxbar) cbar(0(.1)1)
//regsensitivity breakdown y d `covars', compare(`covars') rybar(=rxbar) cbar(0(.1)1) beta(0.5)
regsensitivity breakdown y d `covars', compare(`covars') rybar(=rxbar) cbar(0) beta(0.5)
//regsensitivity plot
regsensitivity bounds    y d `covars', compare(`covars') rybar(=rxbar) cbar(1)
regsensitivity bounds    y d `covars', compare(`covars') rybar(=rxbar) cbar(0)
regsensitivity bounds    y d `covars', compare(`covars')
//regsensitivity plot

log close
