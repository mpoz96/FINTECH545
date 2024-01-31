using Distributions
using StatsBase
using DataFrames
using CSV
using Plots
using PlotThemes
using Printf
using JuMP
using Ipopt
using StateSpaceModels
using LinearAlgebra
using GLM

#Problem 1
function first4Moments(sample)

    n = size(sample,1)

    #mean
    μ_hat = sum(sample)/n

    #remove the mean from the sample
    data1.x_corrected = sample .- μ_hat
    cm2 = data1.x_corrected'*data1.x_corrected/n

    #variance
    σ2_hat = data1.x_corrected'*data1.x_corrected/(n-1)

    #skew
    skew_hat = sum(data1.x_corrected.^3)/n/sqrt(cm2*cm2*cm2)

    #kurtosis
    kurt_hat = sum(data1.x_corrected.^4)/n/cm2^2

    excessKurt_hat = kurt_hat - 3

    return μ_hat, σ2_hat, skew_hat, excessKurt_hat
end

data1 = CSV.read("problem1.csv", DataFrame)

#biased estimate
m, s2, sk, k = first4Moments(data1.x)

#out of the box estimates
jm = mean(data1.x)
js2 = var(data1.x)
jsk = skewness(data1.x)
jk = kurtosis(data1.x)

print("Mean diff = $(m-jm)\n")
print("var diff = $(s2-js2)\n")
print("skew diff = $(sk-jsk)\n")
print("Kurtosis diff = $(k-jk)\n")

#Answer
print("skewness and kurtosis are biased in julia function\n")

#Bessels Correction for variance bias:
n = size(data1,1)
print("with Bessels Correction, $(s2*(n/(n-1))) is the corrected variance. Therefore the Julia variance is biased\n")


# Question 2

data2 = CSV.read("problem2.csv", DataFrame)

ols = lm(@formula(y~x), data2)
println(ols)

n = size(data2.x, 1)
y = data2.y
x = hcat(ones(n), data2.x)  # Adding a column for intercept

function myll(σ, β...)
    beta = collect(β)
    e = y - x * beta
    s2 = σ * σ
    ll = -n / 2 * log(2 * π * s2) - sum(e .^ 2) / (2 * s2)
    return ll
end

mle = Model(Ipopt.Optimizer)
set_silent(mle)

@variable(mle, β[1:2], start=0)
@variable(mle, σ >= 0.0, start=1.0)

register(mle, :myll, 3, myll; autodiff=true)

@NLobjective(mle, Max, myll(σ, β...))
optimize!(mle)

println("Betas: ", value.(β))
β_estimated = value.(β)
σ_estimated = value(σ)
residuals = y - x * β_estimated
σ²_estimated = σ_estimated^2

# Compute the variance-covariance matrix of the coefficients
var_cov_matrix = inv(x'x) * σ²_estimated

# Standard errors are the square roots of the diagonal elements
std_errors = sqrt.(diag(var_cov_matrix))

println("Standard Errors: ", std_errors)